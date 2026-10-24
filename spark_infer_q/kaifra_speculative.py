"""K2 greedy DFlash2 verify/commit on DeepInfer's real Scheduler/MTPVarlen.

The native scheduler owns plans and output accounting. TensorFold owns device
state. This adapter supplies native MTP information without a DeepSeek sampler
or MTPInfoPool; CPU metadata slots are consumed by the actual Scheduler.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import time
from pathlib import Path

from .deepinfer_bridge import BridgeError, SchedulerBridge, WorkRow
from .kaifra_runtime import KaifraEngine
from .speculative import greedy_result
from .verification_budget import LaggedAcceptance


def conditional_rates(cumulative):
    result, previous = [], 1.0
    for value in cumulative:
        if not math.isfinite(value) or not 0 <= value <= previous:
            raise ValueError('acceptance rates must be finite cumulative probabilities')
        result.append(value/previous if previous else 0.0)
        previous = value
    return result


@dataclass
class DraftState:
    target: object
    policy: object
    context: object = None
    output_tokens: int = 0
    next_width: int = 1
    current_drafts: tuple = (0,)*7
    cumulative: tuple = ()


@dataclass(frozen=True)
class VerifiedStep:
    outputs: tuple[int, ...]
    keep: int
    next_drafts: tuple[int, ...]
    conditional: tuple[float, ...]
    draft_ms: float
    verify_ms: float
    commit_ms: float
    head_fallback: bool = False
    head_union_count: int | None = None
    cumulative: tuple = ()
    accepted: int = 0
    observed: int = 0


class PostDraftBudget:
    """Existing E17 controller: current cumulative rates in both MTP fields."""
    def __init__(self, artifact, module=None):
        if module is None:
            import spark_q_e3_varlen as module
            digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            if digest != '722b6ea835fa3e7e349af7f22ac727045a65af487a35f6709f3838e066bd678e':
                raise ValueError('unvalidated BudgetProbe binary')
        self.probe = module.BudgetProbe()
        self.artifact = artifact

    def choose(self, cumulative, context_rows, remaining):
        from types import SimpleNamespace
        limit = min(8, remaining)
        if limit <= 1:
            return 1
        costs = self.artifact['costs']['long' if context_rows >= 768 else 'short']
        paid = costs['native_draft_ms'] + costs['feature_increment_ms'] + self.artifact['cpu_predict_ms']
        self.probe.curve([(r, v+paid) for r,v in enumerate(costs['verify_commit_ms'],1)])
        rates = list(cumulative[:limit-1])
        conditional_rates(rates)  # validate the same cumulative probability contract
        params = SimpleNamespace(temperature=0.,top_p=1.,top_k=-1,seq_seed=0,top_logprobs=0,
            prompt_logprobs=False,first_n_token_no_eos=0,eos_id=248044,ignore_eos=False,max_tokens=64)
        entry = dict(draft_tokens=[1]*len(rates),accept_rates=rates,prev_accept_rates=rates,
                     process_cnt=2,schedule_cnt=2)
        width = self.probe.schedule(params,[entry])[0]
        if not 1 <= width <= limit:
            raise BridgeError('post-draft budget exceeded remaining output cap')
        return width


class DFlashProvider:
    """Borrow loaded Q4 weights; optional existing F0/F1 head, no fitting."""
    def __init__(self, worker, *, model=None, feature_group='f1', budget=None):
        self.worker, self.model, self.feature_group = worker, model, feature_group
        self.max_context = worker.max_context
        self.budget = budget
        self.deferred_draft = False

    def propose(self, handle, state, anchor, remaining):
        if self.budget is not None and remaining == 1:
            state.next_width = 1
            state.cumulative = ()
            return [0]*7, [0.0]*7
        worker = self.worker
        if hasattr(worker, 'begin') and worker.request != handle:
            # Its projection cache has one owner. On interleaving, reset the
            # cache before rebuilding from this request's committed context.
            worker.begin(handle)
        tokens, info = worker.propose(state.context, anchor)
        rates = None
        if self.model is not None:
            from .confidence_features import feature_vector
            features = feature_vector(info['signals'], len(state.context),
                state.output_tokens, remaining, False, self.feature_group)
            cumulative = self.model.predict(features)
            state.cumulative = tuple(cumulative)
            rates = conditional_rates(cumulative)
            if self.budget is not None:
                state.next_width = self.budget.choose(cumulative, len(state.context), remaining)
        return tokens, rates


class SpeculativeBackend:
    def __init__(self, target, provider, *, eos_ids, depth, prior,
                 prefill_rows=8, observer=None, prefill_executor=None):
        if prefill_rows not in (1,2,4,8) or any(r not in target.buffers for r in range(1,9)):
            raise ValueError('K2 requires validated eight-row target buffers')
        self.target, self.provider, self.depth = target, provider, depth
        self.eos_ids, self.prior = frozenset(eos_ids), tuple(prior)
        if len(self.prior) != 7:
            raise ValueError('seven-depth conditional prior required')
        LaggedAcceptance(self.prior)
        self.prefill_rows, self.observer, self.states = prefill_rows, observer, {}
        if prefill_executor is not None and observer is not None:
            raise ValueError('per-row prompt observers require the rows8 path')
        self.prefill_executor=prefill_executor

    def create(self, handle):
        with self.target.torch.inference_mode():
            state = DraftState(self.target.state(), LaggedAcceptance(self.prior))
        self.states[handle] = state
        return state

    def position(self, state):
        return state.target.pos

    def release(self, handle, state):
        if self.states.get(handle) is not state:
            raise BridgeError('foreign or released speculative state')
        release=getattr(self.target,'release_state',None)
        if release is not None:
            release(state.target)
        del self.states[handle]
        state.context = None

    def synchronize(self):
        self.target.torch.cuda.synchronize()

    def advance(self, row, state, max_tokens):
        if self.states.get(row.handle) is not state or state.target.pos != row.position:
            raise BridgeError('speculative state ownership/position mismatch')
        target, torch = self.target, self.target.torch
        verify_ms = commit_ms = draft_ms = 0.0
        head_info={}
        accepted=observed=0
        with torch.inference_mode():
            if row.prompt:
                if self.prefill_executor is not None:
                    t=time.perf_counter()
                    result=self.prefill_executor.advance(state.target,row.token_ids,capture=True)
                    logits,features=result.logits,result.features
                    state.context=features if state.context is None else torch.cat([state.context,features])
                    verify_ms+=(time.perf_counter()-t)*1000
                for begin in (() if self.prefill_executor is not None else range(0,len(row.token_ids),self.prefill_rows)):
                    ids = list(row.token_ids[begin:begin+self.prefill_rows])
                    t = time.perf_counter()
                    logits, features = target.step(state.target, ids, capture=True)
                    verify_ms += (time.perf_counter()-t)*1000
                    state.context = features if state.context is None else torch.cat([state.context,features])
                    if self.observer:
                        self.observer(row.handle, row.position+begin, ids, logits, len(ids), state.target)
                outputs = (int(logits[-1].argmax()),) if row.last_prompt_chunk else ()
                keep = len(row.token_ids)
            else:
                t = time.perf_counter()
                predict=getattr(target,'run_predictions',None)
                if predict is None:
                    logits, features, buffers = target.run(state.target, list(row.token_ids))
                    predicted = logits.argmax(-1).tolist()
                else:
                    pred,features,buffers,hidden,head_info=predict(state.target,list(row.token_ids),diagnostic=self.observer is not None)
                    predicted=pred.tolist()
                    logits=None
                    if self.observer is not None:
                        logits=target.screen.original(hidden) if hidden is not None else head_info['full_logits']
                        if predicted!=logits.argmax(-1).tolist():
                            raise BridgeError('screened predictions differ from original head')
                        if not head_info['fallback'] and not torch.equal(head_info['exact'],logits[:,head_info['ids']]):
                            raise BridgeError('screened subset differs from original BF16 logits')
                verify_ms = (time.perf_counter()-t)*1000
                outputs, keep, _, _ = greedy_result(row.token_ids[0], list(row.token_ids[1:]),
                    predicted, self.eos_ids, max_tokens-state.output_tokens)
                from .draft_signals import observed_labels
                labels = observed_labels(row.token_ids[1:], predicted, len(outputs), self.eos_ids)
                observed=len(labels);accepted=sum(v for _,v in labels)
                state.policy.observe(observed,accepted)
                t = time.perf_counter()
                target.commit(state.target, buffers, len(row.token_ids), keep)
                state.context = torch.cat([state.context,features[:keep]])
                commit_ms = (time.perf_counter()-t)*1000
                if self.observer:
                    self.observer(row.handle,row.position,list(row.token_ids),logits,keep,state.target)
                outputs = tuple(outputs)
            state.output_tokens += len(outputs)
            if len(state.context) != state.target.pos:
                raise BridgeError('draft context differs from committed target position')
            next_drafts = [0]*self.depth
            conditional = [0.0]*self.depth
            remaining = max_tokens-state.output_tokens
            if (not getattr(self.provider,'deferred_draft',False) and outputs and
                    remaining > 0 and outputs[-1] not in self.eos_ids):
                t = time.perf_counter()
                proposed, rates = self.provider.propose(row.handle,state,outputs[-1],remaining)
                if len(proposed) < self.depth:
                    raise BridgeError('drafter returned too few proposals')
                next_drafts = proposed[:self.depth]
                conditional = (rates if rates is not None else
                    [a/b for a,b in zip(state.policy.success,state.policy.trials)])[:self.depth]
                torch.cuda.synchronize()
                draft_ms = (time.perf_counter()-t)*1000
            return VerifiedStep(tuple(outputs), keep, tuple(next_drafts), tuple(conditional),
                                draft_ms, verify_ms, commit_ms, bool(head_info.get('fallback',False)),
                                head_info.get('union_count'),state.cumulative,accepted,observed)


class SpeculativeSchedulerBridge(SchedulerBridge):
    """Native MTP counters/publishing and accepted-token flat-layout adapter."""
    def __init__(self, *args, curve=None, postdraft=False, independent_postdraft=False, **kwargs):
        if not postdraft and curve is None:
            raise ValueError('a dynamic verification budget is required')
        super().__init__(*args,max_drafts=7,**kwargs)
        self.depth = 7
        self.postdraft = postdraft
        if independent_postdraft and (not postdraft or not getattr(self.backend,'gpu_batch',False) or self.max_requests!=2):
            raise ValueError('independent post-draft budgets require the two-request batch backend')
        if postdraft and self.max_requests != 1 and not independent_postdraft:
            raise ValueError('post-draft mode requires a single-request measured cost curve')
        if curve is not None:
            if self.max_requests != 1:
                raise ValueError('dynamic mode requires a measured single-request cost curve')
            points = list(curve)
            if ([r for r,_ in points] != list(range(1,9)) or
                    any(not math.isfinite(c) or c<=0 for _,c in points)):
                raise ValueError('positive measured cost curve for rows 1..8 required')
            self.native.decode_perf_curve = points
        # These native calls synchronously copy/read CPU values; they do not
        # retain Tensor storage. A bridge admits only one pending plan, so its
        # metadata workspace can be reused after complete(). NumPy writes avoid
        # tiny per-row Tensor allocations and dispatcher copy/index operations.
        import numpy as np
        self._cpu_arrays = dict(
            proposals=np.zeros((self.max_requests, self.depth), dtype=np.int32),
            rates=np.zeros((self.max_requests, self.depth), dtype=np.float32),
            tokens=np.zeros(self.max_requests, dtype=np.int32),
            accepted=np.zeros(self.max_requests*(self.depth+1), dtype=np.int32),
            widths=np.ones(self.max_requests, dtype=np.int32),
            positions=np.full(self.max_requests, -1, dtype=np.int32))
        self._cpu_views = {
            name: tuple(self.torch.from_numpy(array[:n]) for n in range(len(array)+1))
            for name, array in self._cpu_arrays.items()}

    def prepare(self):
        plan = super().prepare()
        if self.postdraft and plan.rows and not plan.rows[0].prompt:
            # Native scheduling reserves max NEXTN capacity. Its supported ZOS
            # finalization hook updates metadata and process_outputs' seqlens_q;
            # the admission token count remains a conservative reservation.
            try:
                plan.draft_ms=[0.]*len(plan.rows)
                batch_propose=getattr(self.backend.provider,'propose_batch',None)
                if batch_propose is not None:
                    active=[];requests=[]
                    for i,row in enumerate(plan.rows):
                        entry=self._entry(row.handle)
                        if entry['cancelled']:continue
                        state=entry['state'];active.append(i)
                        requests.append((row.handle,state,row.token_ids[0],
                            entry['seq'].sampling_params.max_tokens-state.output_tokens))
                    t=time.perf_counter()
                    with self.torch.inference_mode():
                        results=batch_propose(requests)
                        self.torch.cuda.synchronize()
                    if len(results)!=len(requests):raise BridgeError('batch drafter returned wrong request count')
                    elapsed=(time.perf_counter()-t)*1000
                    drafted=sum(r[3]>1 for r in requests)
                    plan.draft_batch_ms=elapsed;plan.draft_batch_size=drafted
                    for i,request,(proposed,rates) in zip(active,requests,results,strict=True):
                        if len(proposed)!=7:raise BridgeError('batch drafter returned wrong proposal count')
                        request[1].current_drafts=tuple(proposed)
                        # Additive attribution of one shared wall interval; never
                        # charge the entire batch time to each request.
                        plan.draft_ms[i]=elapsed/max(1,drafted) if request[3]>1 else 0.
                else:
                    for i,row in enumerate(plan.rows):
                        entry=self._entry(row.handle)
                        if not getattr(self.backend.provider,'deferred_draft',False) or entry['cancelled']:continue
                        state=entry['state']
                        remaining=entry['seq'].sampling_params.max_tokens-state.output_tokens
                        t=time.perf_counter()
                        with self.torch.inference_mode():
                            proposed,_=self.backend.provider.propose(row.handle,state,row.token_ids[0],remaining)
                            state.current_drafts=tuple(proposed[:7])
                            self.torch.cuda.synchronize()
                        plan.draft_ms[i]=(time.perf_counter()-t)*1000
                widths = self._postdraft_widths(plan)
                if any(not 1 <= q <= len(r.token_ids) for q,r in zip(widths,plan.rows)):
                    raise BridgeError('final post-draft width exceeds native reservation')
                self._cpu_arrays['widths'][:len(widths)] = widths
                plan.native_output.update_metadata_for_sampling_zos(plan.metadata,
                    self._cpu_views['widths'][len(widths)],
                    self._cpu_views['positions'][len(widths)],7)
                plan.rows = tuple(self._work_row(m,seq,self._entry(row.handle))
                    for m,seq,row in zip(plan.metadata,plan.native_output.scheduled_seqs,plan.rows,strict=True))
            except Exception:
                self.poisoned = True
                raise
        return plan

    def _postdraft_widths(self,plan):
        return [self._entry(r.handle)['state'].next_width for r in plan.rows]

    def _work_row(self, meta, seq, entry):
        if meta.is_prompt:
            return super()._work_row(meta,seq,entry)
        start = seq.num_cached_tokens
        data = seq.data
        drafts=(entry['state'].current_drafts if getattr(self.backend.provider,'deferred_draft',False)
                else meta.last_next_n_token)
        ids = (data.get_last_token_id(),) + tuple(drafts[:meta.seqlen_q-1])
        if (data.get_len()-start != 1 or len(ids)!=meta.seqlen_q or
                not 1<=len(ids)<=self.depth+1 or meta.seqlen_k!=start+len(ids)):
            self.poisoned=True
            raise BridgeError('native speculative token/position metadata mismatch')
        return WorkRow(entry['handle'],start,ids,False,False)

    def execute(self, plan):
        self._plan(plan)
        if plan.executed:
            raise BridgeError('plan already executed')
        results = []
        try:
            for row in plan.rows:
                entry = self._entry(row.handle)
                if entry['cancelled']:
                    results.append(None)
                    continue
                result = self.backend.advance(row,entry['state'],entry['seq'].sampling_params.max_tokens)
                if self.backend.position(entry['state']) != row.position+result.keep:
                    raise BridgeError('speculative backend committed wrong position')
                if any(type(t) is not int or not 0<=t<self.vocab_size
                       for t in result.outputs+result.next_drafts):
                    raise BridgeError('invalid speculative token ID')
                if (len(result.next_drafts)!=self.depth or len(result.conditional)!=self.depth or
                        any(not math.isfinite(p) or not 0<=p<=1 for p in result.conditional)):
                    raise BridgeError('invalid speculative draft metadata')
                results.append(result)
            plan.results, plan.executed = tuple(results), True
        except Exception:
            self.poisoned=True
            raise

    def complete(self, plan):
        self._plan(plan)
        if not plan.executed:
            raise BridgeError('backend fence not completed')
        torch = self.torch
        try:
            n = len(plan.rows)
            proposals = self._cpu_views['proposals'][n]
            rates = self._cpu_views['rates'][n]
            self._cpu_arrays['proposals'][:n].fill(0)
            self._cpu_arrays['rates'][:n].fill(0)
            flat, outputs, metrics = [], [], []
            for i,(row,result) in enumerate(zip(plan.rows,plan.results,strict=True)):
                cancelled = self._entry(row.handle)['cancelled']
                live = result is not None and not cancelled
                if live:
                    self._cpu_arrays['proposals'][i] = result.next_drafts
                    self._cpu_arrays['rates'][i] = result.conditional
                out = result.outputs if live else ()
                outputs.append(out[-1] if out else 0)
                accepted = list(out[:-1])
                flat.extend(accepted+[-1]*(len(row.token_ids)-len(accepted)))
                metrics.append(dict(request_id=row.handle.request_id,prompt=row.prompt,
                    verified_rows=len(row.token_ids),emitted=len(out),cancelled=cancelled,
                    accepted=result.accepted if live else 0,observed=result.observed if live else 0,
                    context_rows=row.position,rates=list(result.cumulative) if live else [],
                    head_fallback=result.head_fallback if live else False,
                    head_union_count=result.head_union_count if live else None,
                    draft_ms=(result.draft_ms+getattr(plan,'draft_ms',[0.]*len(plan.rows))[i]) if live else 0.0,
                    draft_batch_ms=getattr(plan,'draft_batch_ms',None),
                    draft_batch_size=getattr(plan,'draft_batch_size',None),
                    draft_ms_is_amortized=hasattr(plan,'draft_batch_ms'),
                    verify_ms=result.verify_ms if live else 0.0,
                    commit_ms=result.commit_ms if live else 0.0))
            # Publish the new draft while native bookkeeping still describes the
            # input plan. This advances native process_cnt even for prompt chunks
            # and cancelled rows, pairing every native schedule_cnt exactly once.
            self.core.process_MTPInfo(plan.native_output,True,proposals,rates)
            flat = flat if any(not r.prompt for r in plan.rows) else []
            self._cpu_arrays['accepted'][:len(flat)] = flat
            self._cpu_arrays['tokens'][:n] = outputs
            accepted = self._cpu_views['accepted'][len(flat)]
            tokens = self._cpu_views['tokens'][n]
            _, pushed = self.native.process_outputs(plan.native_output,accepted,tokens,None)
            pushed = list(map(int,pushed))
            for row,result,count in zip(plan.rows,plan.results,pushed,strict=True):
                entry = self._entry(row.handle)
                expected = () if entry['cancelled'] or result is None else result.outputs
                if count!=len(expected):
                    raise BridgeError('native published wrong speculative prefix length')
                if not entry['cancelled']:
                    if entry['seq'].num_cached_tokens!=self.backend.position(entry['state']):
                        raise BridgeError('native/device position diverged after speculation')
                    if expected and expected[-1] in self.eos_ids:
                        entry['seq'].status=self.core.SequenceStatus.FINISHED_STOPPED
                self.events.append(dict(request_id=row.handle.request_id,published=count))
            if plan.rows and not plan.rows[0].prompt and self.native.decode_perf_curve:
                self.native.update_mtp_runtime(len(plan.rows),sum(len(r.token_ids) for r in plan.rows),
                    sum(m['draft_ms']+m['verify_ms']+m['commit_ms'] for m in metrics),0.0)
            plan.metrics = tuple(metrics)
            self.pool.release(plan.metadata)
            self.pending=None
            self._retire()
            return pushed
        except Exception:
            self.poisoned=True
            raise


def speculative_engine(native, target, provider, *, eos_ids, prior, mode='adaptive',
                       capacity=256,max_requests=1,token_budget=32,prefill_rows=8,
                       tokenizer=None,curve=None,observer=None):
    if getattr(native,'max_drafts',None)!=7:
        raise ValueError('speculation requires a native max_drafts=7 context')
    if mode not in ('lagged','confidence','adaptive'):
        raise ValueError('unsupported speculative mode')
    if capacity+14>target.capacity or capacity+8>provider.max_context:
        raise ValueError('insufficient target/draft lookahead capacity')
    if (mode in ('confidence','adaptive')) != (provider.model is not None):
        raise ValueError('confidence mode requires a predictor; other modes use depth priors')
    if (mode=='adaptive') != (getattr(provider,'budget',None) is not None):
        raise ValueError('adaptive mode requires the current post-draft budget controller')
    dynamic = mode in ('lagged','confidence')
    if dynamic and curve is None:
        raise ValueError('dynamic mode requires an explicit measured cost curve')
    depth = 7
    backend=SpeculativeBackend(target,provider,eos_ids=eos_ids,depth=depth,prior=prior,
                               prefill_rows=prefill_rows,observer=observer)
    bridge=SpeculativeSchedulerBridge(native.core,native.torch,native.group,native.sampling_class,
        backend,vocab_size=int(target.w.head.b.weight.shape[0]),eos_ids=eos_ids,capacity=capacity,
        max_requests=max_requests,token_budget=token_budget,curve=curve if dynamic else None,postdraft=mode=='adaptive')
    engine=KaifraEngine(bridge,tokenizer=tokenizer)
    engine.capabilities=dict(engine.capabilities,target_only=False,speculative=True,
                            draft_depth=depth,verification_budget=('current_postdraft_e17' if mode=='adaptive' else
                                'native_mtp_varlen'))
    engine.capabilities.update(getattr(target,'capabilities',{}))
    return engine


def session_provider(session, mode, *, model_path=None, cost_bucket='short'):
    """Reuse frozen session weights and measured costs; never train/download."""
    if mode in ('confidence','adaptive'):
        from .confidence_features import ConfidenceDraft
        from .local_policy import ProgressModel
        path=Path(model_path) if model_path is not None else session.root/('artifacts/e17-selected-model.json' if mode=='adaptive' else 'artifacts/e16-selected-model.json')
        payload=path.read_bytes();artifact=json.loads(payload)
        approved = (artifact.get('candidate')=='unweighted_f1' and artifact.get('experimental') is True
                    if mode=='adaptive' else artifact.get('gate_pass') is True)
        if (not approved or artifact['feature_group'] not in ('f0','f1') or
                len(artifact['metadata']['heads'])!=7):
            raise ValueError('a gated frozen seven-depth model is required')
        worker=ConfidenceDraft.attach(session.draft)
        worker.feature_group=artifact['feature_group']
        provider=DFlashProvider(worker,model=ProgressModel(artifact['metadata']),
                                feature_group=artifact['feature_group'],
                                budget=PostDraftBudget(artifact) if mode=='adaptive' else None)
        costs=artifact['costs'][cost_bucket]
        paid=costs['native_draft_ms']+costs['feature_increment_ms']+artifact['cpu_predict_ms']
        curve=[(r,v+paid) for r,v in enumerate(costs['verify_commit_ms'],1)]
        metadata=dict(model_sha256=hashlib.sha256(payload).hexdigest(),cost_bucket=cost_bucket,
                      feature_group=artifact['feature_group'],paid_draft_cost_includes_width1=True,
                      budget_policy='current_postdraft' if mode=='adaptive' else 'native_lagged',
                      context_costs='short_below_768_else_long' if mode=='adaptive' else cost_bucket)
    else:
        from .draft_context import CachedContextDraft
        provider=DFlashProvider(CachedContextDraft.attach(session.draft))
        paid=session.calibration['draft_median_ms']['8']
        curve=[(r,v+paid) for r,v in session.calibration['verify_commit_curve_ms']]
        metadata=dict(model_sha256=None,paid_draft_cost_includes_width1=True)
    return provider,curve,metadata
