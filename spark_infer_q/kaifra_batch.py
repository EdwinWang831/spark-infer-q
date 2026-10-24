"""Native two-request batching with current per-request E17 post-draft budgets."""
import copy
import time
from dataclasses import dataclass
from types import SimpleNamespace

from .deepinfer_bridge import BridgeError,SchedulerBridge
from .kaifra_runtime import KaifraEngine
from .kaifra_speculative import DFlashProvider,SpeculativeBackend,SpeculativeSchedulerBridge,VerifiedStep
from .speculative import greedy_result
from .draft_signals import observed_labels


class RequestDraftProvider:
    """Share immutable packed weights; each admitted generation owns its cache."""
    deferred_draft=True
    def __init__(self,provider):
        if not provider.deferred_draft or provider.model is None or provider.budget is None:
            raise ValueError('batching requires current E17 post-draft policy')
        self.base=provider;self.model=provider.model;self.budget=provider.budget
        self.profile_metadata=getattr(provider,'profile_metadata',None)
        self.max_context=provider.max_context;self.providers={}

    def propose(self,handle,state,anchor,remaining):
        provider=self.providers.get(handle)
        if provider is None:
            base=self.base.worker;worker=copy.copy(base);torch=worker.torch
            for name in ('_context_rows','_keys','_values','_pad_features','_pad_context'):
                setattr(worker,name,torch.empty_like(getattr(base,name)))
            worker.used=set(base.used);worker.request=None
            worker.cached_rows=worker.appended_rows=worker.reused_rows=0
            provider=DFlashProvider(worker,model=self.model,feature_group=self.base.feature_group,budget=self.budget)
            provider.deferred_draft=True;self.providers[handle]=provider
        return provider.propose(handle,state,anchor,remaining)

    def release(self,handle):
        self.providers.pop(handle,None)


class BatchedRequestDraftProvider(RequestDraftProvider):
    """Owned batch backbone; keep frozen E17 and per-request budget semantics.

    The caller supplies a verified 1..32-row W8 provider. Existing packed
    weights and the resident worker remain unchanged. Opt-in until measured.
    """
    def __init__(self,provider,matmul,*,kernel_identity,max_requests=4):
        from .draft_batch import BatchedDFlash
        from .draft_backbone import PackedLinear
        if not provider.deferred_draft or provider.model is None or provider.budget is None:
            raise ValueError('batch draft requires current E17 post-draft policy')
        self.base=provider;self.model=provider.model;self.budget=provider.budget
        self.feature_group=provider.feature_group;self.max_context=provider.max_context
        worker=copy.copy(provider.worker)
        worker.weights=worker.weights.copy();worker.fused=worker.fused.copy();worker.used=set(worker.used)
        for mapping in (worker.weights,worker.fused):
            for name,weight in mapping.items():
                if isinstance(weight,PackedLinear):
                    mapping[name]=PackedLinear(weight.packed,matmul)
        self.backend=BatchedDFlash(worker,max_requests=max_requests)
        self.providers=self.backend.caches
        self.profile_metadata=dict(getattr(provider,'profile_metadata',{}) or {},
            draft_execution='owned_batch_v1',draft='batched W8 MLP / Q4 output head / BF16 attention and selector',
            batch_w8_kernel=dict(kernel_identity),
            budget_cost_status='legacy E29 control; batch draft costs not recalibrated')
        # Old joint tables bind the original W8 binary/source. Extra metadata
        # alone would not invalidate them: validate_runtime compares only the
        # table's recorded keys. Replace those identities, never inherit them.
        self.profile_metadata['w8_binary_sha256']=kernel_identity['binary_sha256']
        self.profile_metadata.pop('w8_source_sha256',None)
        if 'source_sha256' in kernel_identity:
            self.profile_metadata['w8_source_sha256']=kernel_identity['source_sha256']

    def release(self,handle):
        self.backend.release(handle)

    def propose(self,handle,state,anchor,remaining):
        return self.propose_batch([(handle,state,anchor,remaining)])[0]

    def propose_batch(self,requests):
        from .draft_batch import DraftInput
        from .confidence_features import feature_vector
        from .kaifra_speculative import conditional_rates
        requests=tuple(requests)
        if not requests:return []
        if len(requests)>self.backend.max_requests or len({r[0] for r in requests})!=len(requests):
            raise ValueError('duplicate or excessive draft batch requests')
        if any(type(r[3]) is not int or r[3]<1 for r in requests):
            raise ValueError('invalid remaining output budget')
        active=[i for i,r in enumerate(requests) if r[3]>1]
        inputs=[DraftInput(requests[i][0],requests[i][1].context,requests[i][2]) for i in active]
        outputs=self.backend.propose_batch(inputs) if inputs else []
        decisions={}
        for i,(tokens,info) in zip(active,outputs,strict=True):
            handle,state,anchor,remaining=requests[i]
            features=feature_vector(info['signals'],len(state.context),state.output_tokens,
                remaining,False,self.feature_group)
            cumulative=tuple(self.model.predict(features));rates=conditional_rates(cumulative)
            width=self.budget.choose(cumulative,len(state.context),remaining)
            if len(tokens)!=7 or not 1<=width<=min(8,remaining):
                raise BridgeError('invalid batched draft result or budget')
            decisions[i]=(tokens,rates,cumulative,width)
        # Publish policy state only after validating every batch result.
        result=[]
        for i,(handle,state,anchor,remaining) in enumerate(requests):
            tokens,rates,cumulative,width=decisions.get(i,([0]*7,[0.]*7,(),1))
            state.cumulative=cumulative;state.next_width=width
            result.append((tokens,rates))
        return result


@dataclass
class Execution:
    rows: tuple
    entries: tuple
    target_ticket: object
    results: tuple = ()
    event: object = None
    phase: str = 'verify'
    committed: float = 0.


@dataclass
class SplitExecution:
    rows: tuple
    entries: tuple
    current: object
    index: int = 0
    results: tuple = ()
    phase: str = 'split'


class BatchBackend(SpeculativeBackend):
    gpu_batch=True
    joint_decode=True
    def release(self,handle,state):
        super().release(handle,state)
        release=getattr(self.provider,'release',None)
        if release is not None:release(handle)

    def synchronize(self):
        self.target.torch.cuda.synchronize()
        if self.target.inflight is not None:self.target.discard(self.target.inflight)

    def launch(self,rows,entries):
        if not self.joint_decode and len(rows)>1:
            return SplitExecution(rows,entries,self._launch(rows[:1],entries[:1]))
        return self._launch(rows,entries)

    def _launch(self,rows,entries):
        active=[]
        for row,entry in zip(rows,entries,strict=True):
            state=entry['state']
            if self.states.get(row.handle) is not state or state.target.pos!=row.position:
                raise BridgeError('batch device ownership/position mismatch')
            if not entry['cancelled']:active.append((state.target,row.token_ids))
        if not active:return Execution(rows,entries,None,results=(None,)*len(rows),phase='done')
        with self.target.torch.inference_mode():
            ticket=self.target.launch(active,diagnostic=self.observer is not None)
        return Execution(rows,entries,ticket)

    def poll(self,execution):
        if isinstance(execution,SplitExecution):
            if not self._poll(execution.current):return False
            execution.results+=execution.current.results;execution.index+=1
            if execution.index==len(execution.rows):return True
            i=execution.index
            execution.current=self._launch(execution.rows[i:i+1],execution.entries[i:i+1]);return False
        return self._poll(execution)

    def _poll(self,execution):
        if execution.phase=='done':return True
        target,torch=self.target,self.target.torch;ticket=execution.target_ticket
        with torch.inference_mode():
            if execution.phase=='verify':
                if all(e['cancelled'] for e in execution.entries):
                    if not ticket.event.query():return False
                    target.discard(ticket);execution.results=(None,)*len(execution.rows);execution.phase='done';return True
                if not target.poll(ticket):return False
                windows={id(w.state):w for w in ticket.windows};results=[];keeps={}
                elapsed=(time.perf_counter()-ticket.started)*1000
                for row,entry in zip(execution.rows,execution.entries,strict=True):
                    if entry['cancelled']:results.append(None);continue
                    state=entry['state'];win=windows[id(state.target)];q=len(row.token_ids)
                    predicted=ticket.cpu_predictions[win.at:win.at+q].tolist()
                    if self.observer is not None:
                        full=ticket.full_logits[win.at:win.at+q]
                        if predicted!=full.argmax(-1).tolist():raise BridgeError('batch screened argmax differs')
                        info=ticket.head_info
                        if not info['fallback'] and not torch.equal(info['exact'][win.at:win.at+q],full[:,info['ids']]):
                            raise BridgeError('batch BF16 subset differs')
                    outputs,keep,_,_=greedy_result(row.token_ids[0],list(row.token_ids[1:]),predicted,
                        self.eos_ids,entry['seq'].sampling_params.max_tokens-state.output_tokens)
                    labels=observed_labels(row.token_ids[1:],predicted,len(outputs),self.eos_ids)
                    accepted=sum(v for _,v in labels);observed=len(labels)
                    state.policy.observe(observed,accepted);state.output_tokens+=len(outputs)
                    state.context=torch.cat([state.context,ticket.features[win.at:win.at+keep]])
                    keeps[id(state.target)]=keep
                    # Shared verify wall is allocated equally, not an isolated
                    # per-request measurement. Total throughput uses wall time.
                    results.append(VerifiedStep(tuple(outputs),keep,(0,)*self.depth,(0.,)*self.depth,
                        0.,elapsed/len(execution.rows),0.,bool(ticket.head_info['fallback']),
                        ticket.head_info.get('union_count'),state.cumulative,accepted,observed))
                execution.results=tuple(results);execution.committed=time.perf_counter()
                execution.event=target.commit(ticket,keeps);execution.phase='commit';return False
            if execution.phase=='commit':
                if not execution.event.query():return False
                target.retire(ticket)
                for row,entry,result in zip(execution.rows,execution.entries,execution.results,strict=True):
                    if result is None:continue
                    state=entry['state']
                    if target._slot(state.target)<0 or state.target.pos!=row.position+result.keep or len(state.context)!=state.target.pos:
                        raise BridgeError('joint commit/context position mismatch')
                    if self.observer is not None and not entry['cancelled']:
                        win=next(w for w in ticket.windows if w.state is state.target)
                        self.observer(row.handle,row.position,list(row.token_ids),
                            ticket.full_logits[win.at:win.at+len(row.token_ids)],result.keep,state.target)
                execution.phase='done';return True
        raise BridgeError('unknown asynchronous batch phase')


class AsyncBatchMixin:
    """One pending native plan; never publish before its GPU event completes."""
    def execute_async(self,plan):
        self._plan(plan)
        if plan.executed or hasattr(plan,'execution'):raise BridgeError('plan already dispatched')
        try:
            if not plan.rows or plan.rows[0].prompt:
                super().execute(plan);return
            entries=tuple(self._entry(row.handle) for row in plan.rows)
            plan.execution=self.backend.launch(plan.rows,entries)
        except Exception:
            self.poisoned=True;raise

    def poll(self,plan):
        self._plan(plan)
        if plan.executed:return True
        try:
            if not self.backend.poll(plan.execution):return False
            plan.results=plan.execution.results
            for row,entry,result in zip(plan.rows,plan.execution.entries,plan.results,strict=True):
                if entry['cancelled'] or result is None:continue
                if any(type(t) is not int or not 0<=t<self.vocab_size for t in result.outputs):
                    raise BridgeError('invalid joint output token')
            plan.predictions=tuple(r.outputs[-1] if r and r.outputs else 0 for r in plan.results)
            if not hasattr(plan,'metrics'):
                plan.metrics=tuple(dict(request_id=row.handle.request_id,prompt=False,
                    verified_rows=len(row.token_ids),emitted=len(r.outputs) if r else 0)
                    for row,r in zip(plan.rows,plan.results,strict=True))
            plan.executed=True;return True
        except Exception:
            self.poisoned=True;raise

    def execute(self,plan):
        self.execute_async(plan)
        while not self.poll(plan):
            if not self.wait_for_device(plan):time.sleep(.0005)

    def wait_for_device(self,plan):
        """Wait only for this plan's next event; poll still validates/publishes.

        Fixed sleeps between GPU phases delayed the following eager DFlash
        submission on Spark. Event completion retains the same fences without
        a process-wide CUDA synchronize or a Python busy-poll loop.
        """
        self._plan(plan)
        if plan.executed:return False
        execution=getattr(plan,'execution',None)
        while isinstance(execution,SplitExecution):execution=execution.current
        if execution is None or execution.phase=='done':return False
        if execution.phase=='verify':
            ticket=execution.target_ticket
            event=None if ticket is None else ticket.event
        elif execution.phase=='commit':event=execution.event
        else:raise BridgeError('unknown device wait phase')
        if event is None:return False
        try:event.synchronize()
        except Exception:
            self.poisoned=True;raise
        return True


class BatchSpeculativeBridge(AsyncBatchMixin,SpeculativeSchedulerBridge):
    def __init__(self,*args,joint_budget=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.joint_budget=joint_budget

    def _postdraft_widths(self,plan):
        widths=super()._postdraft_widths(plan)
        if self.joint_budget is None:return widths
        from .joint_budget import BudgetRequest
        active=[];requests=[]
        for i,row in enumerate(plan.rows):
            entry=self._entry(row.handle)
            if entry['cancelled']:continue
            state=entry['state'];active.append(i)
            requests.append(BudgetRequest(self.backend.target._slot(state.target),len(state.context),
                entry['seq'].sampling_params.max_tokens-state.output_tokens,tuple(state.cumulative),
                widths[i],len(row.token_ids)))
        decision=self.joint_budget.choose(requests)
        for i,q in zip(active,decision['widths'],strict=True):
            widths[i]=q;self._entry(plan.rows[i].handle)['state'].next_width=q
        plan.joint_budget=decision
        return widths


class BatchTargetBridge(AsyncBatchMixin,SchedulerBridge):
    # Prompt execution needs the speculative backend's max_tokens argument.
    def execute_async(self,plan):
        if plan.rows and plan.rows[0].prompt:
            self._plan(plan)
            if plan.executed:raise BridgeError('plan already executed')
            try:
                results=[]
                for row in plan.rows:
                    e=self._entry(row.handle)
                    results.append(None if e['cancelled'] else self.backend.advance(row,e['state'],e['seq'].sampling_params.max_tokens))
                plan.results=tuple(results);plan.predictions=tuple(r.outputs[-1] if r and r.outputs else 0 for r in results)
                plan.executed=True
            except Exception:self.poisoned=True;raise
        else:super().execute_async(plan)


def batch_engine(native,target,provider,*,eos_ids,prior,capacity=1536,tokenizer=None,observer=None,target_only=False,joint_decode=True,joint_budget=None):
    depth=0 if target_only else 7
    if native.max_drafts!=depth:raise ValueError('native NEXTN context differs from batch mode')
    if joint_budget is not None:
        if target_only or not joint_decode:raise ValueError('joint budgets require adaptive joint execution')
        metadata=getattr(provider,'profile_metadata',None)
        if metadata is None:raise ValueError('joint budgets require a fingerprinted execution profile')
        joint_budget.validate_runtime(metadata,target)
    if capacity+2*depth>target.capacity:raise ValueError('insufficient physical lookahead')
    if not target_only:
        if capacity+8>provider.max_context:raise ValueError('insufficient draft context')
        if not isinstance(provider,RequestDraftProvider):provider=RequestDraftProvider(provider)
    else:provider=SimpleNamespace(deferred_draft=True)
    backend=BatchBackend(target,provider,eos_ids=eos_ids,depth=depth,prior=prior,observer=observer)
    backend.joint_decode=joint_decode
    args=(native.core,native.torch,native.group,native.sampling_class,backend)
    kwargs=dict(vocab_size=int(target.w.head.b.weight.shape[0]),eos_ids=eos_ids,capacity=capacity,max_requests=2,token_budget=32)
    bridge=BatchTargetBridge(*args,**kwargs) if target_only else BatchSpeculativeBridge(*args,postdraft=True,independent_postdraft=True,joint_budget=joint_budget,**kwargs)
    engine=KaifraEngine(bridge,tokenizer=tokenizer)
    engine.capabilities=dict(engine.capabilities,gpu_batch=True,asynchronous=True,max_requests=2,
        speculative=not target_only,target_only=target_only,verification_budget='per_request_current_e17' if not target_only else 'none',
        global_budget_optimized=joint_budget is not None)
    if joint_budget is not None:
        engine.capabilities.update(verification_budget='joint_current_e17',batch_cost_sha256=joint_budget.sha256)
    engine.capabilities.update(target.capabilities)
    engine.capabilities['joint_decode']=joint_decode
    engine.capabilities.update(draft_gpu_batch=isinstance(provider,BatchedRequestDraftProvider),
        draft_execution='owned_batch_v1' if isinstance(provider,BatchedRequestDraftProvider) else 'per_request',
        draft_cost_calibrated=False if isinstance(provider,BatchedRequestDraftProvider) else None)
    return engine
