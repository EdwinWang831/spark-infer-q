"""Full target/Q4 DFlash2 verification through native scheduling, no reload."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']).resolve();manifest=json.loads((root/'manifest.json').read_text())
    for name,sha in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    tag='_kaifra_k2_gpu_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    try:
        spec=importlib.import_module(tag+'.kaifra_speculative')
        native=importlib.import_module(tag+'.native_scheduler')
        result=validate(session,spec,native)
        result.update(source_sha256=manifest,model_reloaded=False)
        return result
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)


def validate(session,spec,native_module):
    torch,target=session.torch,session.target
    identities=id(session.target),id(session.draft)
    prompts=dict(arithmetic='What is 17 + 25? Answer with only the integer.',
        zh='用两句话解释什么是缓存，以及为什么重复读取可能更快。',
        code='Write a Python function sum_positive(xs) that sums positive values. Output only code.')
    references={};cases=[];rows=states=0;shadows={};kinds={}
    for name,text in prompts.items():
        messages=[dict(role='user',content=text)]
        ids=session.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                                  enable_thinking=False,return_dict=False)
        assert len(ids)+48<=256
        state=target.state();logits=[];tokens=[];inputs=list(ids)
        for token in ids:
            lg,_=target.step(state,[token],capture=False);logits.append(lg.cpu())
        for i in range(48):
            token=int(lg[-1].argmax());tokens.append(token)
            if token in session.eos or i==47:break
            inputs.append(token);lg,_=target.step(state,[token],capture=False);logits.append(lg.cpu())
        references[name]=dict(ids=ids,messages=messages,logits=logits,tokens=tokens,inputs=inputs)
        del state,lg
    # Reproduce the existing E17 post-draft loop directly, outside the native
    # adapter. Compare every selected width, not only final greedy output.
    from spark_infer_q.confidence_features import ConfidenceDraft,feature_vector
    from spark_infer_q.local_policy import ProgressModel
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.speculative import greedy_result
    from e3_mechanism_checks import params,entry
    import spark_q_e3_varlen
    artifact=json.loads((session.root/'artifacts/e17-selected-model.json').read_text())
    predictor=ProgressModel(artifact['metadata'])
    legacy_widths={}
    for kind,ref in references.items():
        st,ctx,lg=prompt_state(target,ref['ids'],chunk_size=8)
        tokens=[int(lg[-1].argmax())];widths=[]
        draft=ConfidenceDraft.attach(session.draft);draft.feature_group='f1';draft.begin(('legacy',kind))
        budget=spark_q_e3_varlen.BudgetProbe()
        while len(tokens)<32 and tokens[-1] not in session.eos:
            remaining=32-len(tokens);limit=min(8,remaining);width=1;proposals=[]
            if limit>1:
                proposals,info=draft.propose(ctx,tokens[-1])
                rates=predictor.predict(feature_vector(info['signals'],len(ctx),len(tokens),remaining,False,'f1'))[:limit-1]
                c=artifact['costs']['long' if len(ctx)>=768 else 'short']
                paid=c['native_draft_ms']+c['feature_increment_ms']+artifact['cpu_predict_ms']
                budget.curve([(r,v+paid) for r,v in enumerate(c['verify_commit_ms'],1)])
                width=budget.schedule(params(),[entry(rates,rates,process=len(widths)+2,schedule=len(widths)+2)])[0]
            lg,ft,buf=target.run(st,[tokens[-1]]+proposals[:width-1])
            emitted,keep,_,_=greedy_result(tokens[-1],proposals[:width-1],lg.argmax(-1).tolist(),session.eos,remaining)
            target.commit(st,buf,width,keep);ctx=torch.cat([ctx,ft[:keep]])
            tokens.extend(emitted);widths.append(width)
        assert tokens==ref['tokens'][:32]
        legacy_widths[kind]=widths
        del st,ctx,lg,ft,buf,draft
    def observe(handle,position,ids,lg,keep,state):
        nonlocal rows,states
        ref=references[kinds[handle.request_id]]
        assert ids[:keep]==ref['inputs'][position:position+keep]
        if handle not in shadows:shadows[handle]=target.state()
        for j,token in enumerate(ids[:keep]):
            assert torch.equal(lg[j:j+1].cpu(),ref['logits'][position+j])
            target.step(shadows[handle],[token],capture=False);rows+=1
        assert not target.state_difference(state,shadows[handle]);states+=1
    e1=session.root.parent
    class ControlledProvider:
        """Oracle-only diagnostic; excluded from natural/throughput evidence."""
        def __init__(self,base,fault):
            self.base,self.fault=base,fault
            self.max_context,self.model=base.max_context,base.model
        def propose(self,handle,state,anchor,remaining):
            original,rates=self.base.propose(handle,state,anchor,remaining)
            ref=references[kinds[handle.request_id]]['tokens']
            tokens=ref[state.output_tokens:state.output_tokens+7]
            tokens=tokens+[0]*(7-len(tokens))
            if self.fault is not None and remaining>self.fault+1:
                tokens[self.fault]=(tokens[self.fault]+1)%int(target.w.head.b.weight.shape[0])
            return tokens,rates
    with native_module.NativeScheduler(e1/'DeepInfer',e1/'build-release/lib/deepinfer',max_drafts=7) as native:
        def make(mode='fixed7',max_requests=1,fault='natural'):
            provider,curve,metadata=spec.session_provider(session,mode)
            if fault!='natural':provider=ControlledProvider(provider,fault)
            return spec.speculative_engine(native,target,provider,eos_ids=session.eos,
                prior=session.calibration['conditional_prior'],mode=mode,capacity=256,
                max_requests=max_requests,tokenizer=session.tokenizer,curve=curve,observer=observe),metadata
        def submit(e,name,kind,limit=32):
            kinds[name]=kind
            return e.submit(name,references[kind]['messages'],max_tokens=limit)
        def finish(e,name,events,handles,metadata=None,diagnostic=False):
            for handle,kind,limit in handles:
                output=[t for ev in events if ev.handle==handle for t in ev.token_ids]
                assert output==references[kind]['tokens'][:limit],(name,kind)
            assert not e.active_requests and not e.bridge.backend.states
            assert not e.bridge.native.has_unfinished_seqs()
            assert e.bridge.blocks.get_num_free_blocks()==e.bridge.block_count-1
            cases.append(dict(case=name,status='PASS',oracle_proposals=diagnostic,
                output_tokens=sum(len(x.token_ids) for x in events),metadata=metadata or {}))
            e.close();shadows.clear()
            print(json.dumps(dict(k2_case=name,status='PASS')),flush=True)
        for mode in ('fixed1','fixed3','fixed7','lagged','confidence','adaptive'):
            for kind in prompts:
                e,metadata=make(mode);h=submit(e,kind,kind)
                events=[];widths=[]
                while e.active_requests:
                    events+=e.step()
                    widths += [m['verified_rows'] for m in e.last_step_metrics if not m['prompt']]
                finish(e,mode+'_'+kind,events,[(h,kind,32)],metadata)
                cases[-1]['decode_widths']=widths
                if mode=='adaptive':
                    assert widths==legacy_widths[kind],(kind,widths,legacy_widths[kind])
                    cases[-1]['legacy_budget_widths_exact']=True
        for fault in (None,0,3,6):
            e,metadata=make(fault=fault);h=submit(e,'controlled','code',32)
            events=list(e.drain())
            finish(e,'controlled_fault_'+str(fault),events,[(h,'code',32)],diagnostic=True)
        e,_=make(fault=None);h=submit(e,'controlled_eos','arithmetic',32)
        events=list(e.drain());assert events[-1].finish_reason=='stop'
        finish(e,'controlled_eos_in_proposal',events,[(h,'arithmetic',32)],diagnostic=True)
        e,_=make(max_requests=2)
        a,b=submit(e,'a','zh',24),submit(e,'b','code',24)
        events=list(e.drain());finish(e,'two_requests_fixed7',events,[(a,'zh',24),(b,'code',24)])
        e,_=make('adaptive');h=submit(e,'one','zh',1);events=list(e.drain())
        assert events[-1].processed_position==len(references['zh']['ids'])
        finish(e,'max_tokens_one',events,[(h,'zh',1)])
        for phase in ('waiting','prepared_prompt','prepared_decode'):
            e,_=make('adaptive');h=submit(e,'reuse','zh',24)
            if phase=='prepared_decode':
                while not e.step():pass
            if phase!='waiting':p=e.bridge.prepare();e.bridge.execute(p)
            e.cancel(h)
            if phase!='waiting':e.bridge.complete(p)
            events=list(e.drain())
            assert not any(ev.token_ids for ev in events) and events[-1].finish_reason=='cancelled'
            new=submit(e,'reuse','arithmetic',32)
            assert new.generation>h.generation
            more=list(e.drain());events+=more
            finish(e,'cancel_'+phase+'_reuse',events,[(new,'arithmetic',32)])
        native_metadata=native.metadata.copy()
    assert identities==(id(session.target),id(session.draft))
    return dict(status='PASS_K2_NATIVE_GPU',native=native_metadata,cases=cases,
        legacy_postdraft_widths=legacy_widths,full_logit_rows=rows,live_state_checks=states,same_target_object=True,same_draft_object=True,
        limits=['synchronous greedy, separate GPU execution per request',
                'controlled oracle proposals are state tests only, not drafter quality or timing',
                'same Target serial reference, not independent full-model numerical proof'])
