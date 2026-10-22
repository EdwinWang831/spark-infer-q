"""Predeclared new-prompt A/B, callable only after the E15 offline gate passes."""
import hashlib
import json
from pathlib import Path
import time

from e16_ablation_tasks import TASKS
CONTRACT=dict(modes=['fixed','native_cached','f0','f1'],repetitions=3,capacity=2048,max_depth=7,
              scope='independent bilingual F0/F1 ablation with E15 frozen heads; full draft/feature/transfer/prediction included')



def run(session,arguments):
    from e3_wide_target import WideTarget
    from e3_mechanism_checks import entry,params
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_context import CachedContextDraft
    from spark_infer_q.confidence_features import ConfidenceDraft,feature_vector
    from spark_infer_q.local_policy import ProgressModel
    from spark_infer_q.verification_budget import LaggedAcceptance
    from spark_infer_q.draft_signals import observed_labels
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.speculative import greedy_result
    import spark_q_e3_varlen
    assert hashlib.sha256(Path(spark_q_e3_varlen.__file__).read_bytes()).hexdigest()=='722b6ea835fa3e7e349af7f22ac727045a65af487a35f6709f3838e066bd678e'
    root=session.root;torch=session.torch
    trained=json.loads((root/'artifacts/e15-training.json').read_text())
    assert trained['status']=='PASS_E15_TRAINING' and trained['gate_pass'],'offline gate must pass'
    artifact=root/'artifacts/e15-selected-model.json';model_info=json.loads(artifact.read_text())
    assert model_info['gate_pass'] and model_info['feature_group']==trained['selected_model']
    models={g:ProgressModel(trained['models'][g]['metadata']) for g in ('f0','f1')}
    group='both_e15_frozen_heads'
    target=WideTarget(session.weights,capacity=2048)
    base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    native=CachedContextDraft.attach(base);learned=ConfidenceDraft.attach(base);learned.feature_group='f1'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    names=['e16_ablation.py','e16_ablation_tasks.py','spark_infer_q/confidence_features.py','spark_infer_q/confidence_head.py',
           'spark_infer_q/local_policy.py','spark_infer_q/draft_context.py','spark_infer_q/verification_budget.py',
           'spark_infer_q/speculative.py','e3_target.py','e3_wide_target.py','e3_mechanism_checks.py']
    hashes={n:sha(root/n) for n in names}
    report=dict(status='RUNNING',contract=CONTRACT,tasks=[],feature_group=group,fixed_rows=model_info['fixed_rows'],
                source_sha256=hashes,model_sha256=sha(artifact),started_unix=time.time(),
                target_object_id=id(session.weights),draft_object_id=id(session.draft))
    output=root/'artifacts/e16-ablation.json';assert not output.exists()
    def save():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def generate(task,start,context,first,mode,oracle=None):
        state=start.clone();ctx=context.clone();tokens=[first];steps=[];logs=[];stop=first in session.eos
        worker=learned if mode in models else native
        if mode in models:worker.feature_group=mode
        model=models.get(mode)
        if mode!='serial':worker.begin((task['id'],time.monotonic_ns()))
        c=model_info['costs']['long' if start.pos>=768 else 'short'];curve=c['verify_commit_ms']
        budget=spark_q_e3_varlen.BudgetProbe();policy=LaggedAcceptance(session.calibration['conditional_prior'])
        draft=c['native_draft_ms']
        if mode=='native_cached':
            budget.curve([(r,v+(draft if r>1 else 0.)) for r,v in enumerate(curve,1)])
            for _ in range(5):budget.runtime(1,1,curve[0],0.)
        else:
            paid=draft+c['feature_increment_ms']+(trained['models'][mode]['cpu_predict_ms'] if mode in models else 0.)
            budget.curve([(r,v+paid) for r,v in enumerate(curve,1)])
        torch.cuda.synchronize();t0=time.perf_counter()
        while len(tokens)<task['output_cap'] and not stop:
            anchor=tokens[-1];offset=len(tokens)-1
            if mode=='serial':
                lg,_=target.step(state,[anchor],capture=False);logs.append(lg.cpu())
                nxt=int(lg[-1].argmax());tokens.append(nxt);stop=nxt in session.eos;continue
            limit=min(8,task['output_cap']-len(tokens));width=1;rates=[]
            t=time.perf_counter()
            if mode=='native_cached':
                if limit>1:
                    rates=policy.current[:limit-1]
                    width=budget.schedule(params(),[entry(rates,policy.previous[:limit-1],process=len(steps)+2,schedule=len(steps)+2)])[0]
                proposals,info=worker.propose(ctx,anchor) if width>1 else ([],{})
            elif mode=='fixed':
                width=min(report['fixed_rows'],limit)
                proposals,info=worker.propose(ctx,anchor) if limit>1 else ([],{})
            else:
                proposals,info=worker.propose(ctx,anchor) if limit>1 else ([],{})
                if limit>1:
                    x=feature_vector(info['signals'],len(ctx),len(tokens),task['output_cap']-len(tokens),task['thinking'],mode)
                    rates=model.predict(x)[:limit-1]
                    width=budget.schedule(params(),[entry(rates,rates,process=len(steps)+2,schedule=len(steps)+2)])[0]
            proposals=proposals[:width-1];torch.cuda.synchronize();dm=(time.perf_counter()-t)*1000
            t=time.perf_counter();lg,ft,buf=target.run(state,[anchor]+proposals);vm=(time.perf_counter()-t)*1000
            pred=lg.argmax(-1).tolist()
            out,keep,_,stop=greedy_result(anchor,proposals,pred,session.eos,task['output_cap']-len(tokens))
            labels=observed_labels(proposals,pred,len(out),session.eos)
            t=time.perf_counter();target.commit(state,buf,width,keep);cm=(time.perf_counter()-t)*1000
            ctx=torch.cat([ctx,ft[:keep]]);tokens.extend(out)
            assert len(ctx)==state.pos==start.pos+len(tokens)-1
            if oracle is not None:
                assert out==oracle['tokens'][offset+1:offset+1+len(out)]
                for j in range(keep):assert torch.equal(lg[j:j+1].cpu(),oracle['logits'][offset+j])
            policy.observe(len(labels),sum(v for _,v in labels))
            if mode=='native_cached':budget.runtime(1,width,dm+vm+cm,0.)
            steps.append(dict(context_rows=state.pos-keep,width=width,emitted=keep,accepted=sum(v for _,v in labels),
                              observed=len(labels),rates=rates,draft_feature_budget_ms=dm,verify_ms=vm,commit_ms=cm))
        torch.cuda.synchronize();elapsed=(time.perf_counter()-t0)*1000
        return dict(tokens=tokens,state=state,logits=logs,steps=steps,decode_ms=elapsed)
    save()
    try:
        for ti,task in enumerate(TASKS):
            ids=session.tokenizer.apply_chat_template([dict(role='user',content=task['prompt'])],tokenize=True,
                add_generation_prompt=True,enable_thinking=task['thinking'],return_dict=False)
            if 'target_prompt_tokens' in task:
                n=task['target_prompt_tokens'];assert len(ids)>n;ids=ids[:n-128]+ids[-128:]
            assert len(ids)+task['output_cap']+8<=2048
            start,ctx,lg=prompt_state(target,ids,chunk_size=8);first=int(lg[-1].argmax())
            golden=generate(task,start,ctx,first,'serial')
            rec=dict(id=task['id'],prompt_tokens=len(ids),completion_tokens=len(golden['tokens']),language=task['language'],family=task['family'],eos=golden['tokens'][-1] in session.eos,output_cap=task['output_cap'],checks=[],timings=[],
                     token_sha256=hashlib.sha256(json.dumps(golden['tokens']).encode()).hexdigest())
            report['tasks'].append(rec)
            for mode in CONTRACT['modes']:
                result=generate(task,start,ctx,first,mode,oracle=golden)
                assert result['tokens']==golden['tokens'] and not target.state_difference(result['state'],golden['state'])
                rec['checks'].append(dict(mode=mode,committed_logits_exact=True,live_state_exact=True,logit_rows=len(golden['tokens'])-1))
                del result
            for rep in range(CONTRACT['repetitions']):
                modes=CONTRACT['modes'][:]
                if rep%2:modes.reverse()
                shift=(ti+rep)%len(modes);modes=modes[shift:]+modes[:shift]
                for slot,mode in enumerate(modes):
                    result=generate(task,start,ctx,first,mode)
                    assert result['tokens']==golden['tokens'] and not target.state_difference(result['state'],golden['state'])
                    rec['timings'].append(dict(mode=mode,repetition=rep,order=slot,decode_ms=result['decode_ms'],steps=result['steps'],exact=True))
                    del result
                save();print(json.dumps(dict(e16_ablation=task['id'],repetition=rep)),flush=True)
            del start,ctx,golden
        for n,h in hashes.items():assert sha(root/n)==h,n
        assert sha(artifact)==report['model_sha256']
        report.update(status='PASS_E16_ABLATION',finished_unix=time.time());save()
        return dict(status=report['status'],artifact=output.name,target_object_id=id(session.weights),draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');save();raise
