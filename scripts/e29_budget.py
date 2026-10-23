"""Contemporaneous fixed-seven versus frozen-linear budget, BF16 draft/eager target."""
import hashlib
import json
import random
import time

from e28_probe import sha

MODES=('fixed7','linear')
CONTRACT=dict(prompts=8,repetitions=2,modes=list(MODES),capacity=2048,output_cap=384,
    policy='fixed seven drafts versus frozen E17 NumPy linear using E29 calibration-only frozen costs',
    execution='identical BF16 draft, Q4 head, eager target; no graph or W8 payload',
    timing='decode after first token, includes drafting/F1/CPU prediction/budget/verify/commit/loop',
    limits=['same eight reused engineering roots; two repetitions, not new independent traffic',
        'prefill, state clone, oracle/checks outside timing; no default promotion or retraining'])


def order(ti,rep):return list(MODES) if (ti+rep)%2==0 else list(reversed(MODES))


def summarize(tasks,samples=4000):
    assert len(tasks)==len({t['group'] for t in tasks})==8
    totals=[]
    for t in tasks:
        assert len(t['timings'])==4
        row={}
        for m in MODES:
            ts=[v for v in t['timings'] if v['mode']==m]
            assert sorted(v['repetition'] for v in ts)==[0,1] and all(v['exact'] for v in ts)
            row[m]=sum(v['decode_ms'] for v in ts)
        totals.append(row)
    def rates(indices):
        n=sum((tasks[i]['completion_tokens']-1)*2 for i in indices)
        return {m:n*1000/sum(totals[i][m] for i in indices) for m in MODES}
    point=rates(list(range(8)));rng=random.Random(29281023)
    draws=[]
    for _ in range(samples):
        v=rates(rng.choices(range(8),k=8));draws.append(v['linear']/v['fixed7']-1)
    draws.sort();trace={}
    for m in MODES:
        runs=[v for t in tasks for v in t['timings'] if v['mode']==m];steps=[s for v in runs for s in v['steps']]
        trace[m]=dict(decode_ms=sum(v['decode_ms'] for v in runs),rounds=len(steps),verified_rows=sum(s['width'] for s in steps),
            emitted=sum(s['emitted'] for s in steps),accepted=sum(s['accepted'] for s in steps),observed=sum(s['observed'] for s in steps),
            phase_ms={k:sum(s[k] for s in steps) for k in ('draft_budget_ms','verify_ms','commit_ms')},
            width_counts={str(w):sum(s['width']==w for s in steps) for w in range(1,9)})
    return dict(decode_tps=point,relative_gain=point['linear']/point['fixed7']-1,
        prompt_bootstrap_gain_95=[draws[int(.025*(samples-1))],draws[int(.975*(samples-1))]],trace=trace,
        prompts=8,timed_runs=32,timed_tokens_per_arm=sum((t['completion_tokens']-1)*2 for t in tasks),
        per_repetition=[dict(repetition=r,relative_gain=
            sum(v['decode_ms'] for t in tasks for v in t['timings'] if v['mode']=='fixed7' and v['repetition']==r)/
            sum(v['decode_ms'] for t in tasks for v in t['timings'] if v['mode']=='linear' and v['repetition']==r)-1) for r in range(2)])


def run(session,arguments):
    from e24_run import expected_prompt
    from e3_wide_target import WideTarget
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.confidence_features import ConfidenceDraft,feature_vector
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.confidence_head import choose_rows
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.draft_signals import observed_labels
    root,torch=session.root,session.torch;assert not arguments
    seal_path=root/'artifacts/e29-budget-seal.json';seal=json.loads(seal_path.read_text());assert seal['contract']==CONTRACT
    def check():
        for kind in ('source_sha256','input_sha256'):
            for n,h in seal[kind].items():assert sha(root/n)==h,n
        for n,h in seal['external_source_sha256'].items():assert sha(root.parent/'e2-load/reference/tensorfold'/n)==h,n
    check();path=root/'artifacts/e29-budget.json';assert not path.exists()
    report=dict(status='RUNNING',contract=CONTRACT,started_unix=time.time(),seal_sha256=sha(seal_path),tasks=[],
        target_object_id=id(session.weights),draft_object_id=id(session.draft),cpu_threads=torch.get_num_threads())
    def save():
        p=path.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(path)
    save();target=WideTarget(session.weights,capacity=2048)
    base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048;engine=ConfidenceDraft.attach(base)
    model=NumpyLinearProgressModel(json.loads((root/'artifacts/e27-linear-model.json').read_text())['metadata'])
    costs=json.loads((root/'artifacts/e29-frozen-costs.json').read_text())['modes']['bf16']
    reference={t['id']:t for t in json.loads((root/'artifacts/e29-online.json').read_text())['tasks']}
    tasks=json.loads((root/'e28_test_tasks.json').read_text())['tasks']
    def generate(task,start,context,first,mode,oracle=None):
        state=start.clone();ctx=context.clone();tokens=[first];steps=[];logs=[];features=[];stop=first in session.eos
        engine.begin((task['id'],mode,time.monotonic_ns()));torch.cuda.synchronize();t0=time.perf_counter()
        while len(tokens)<task['output_cap'] and not stop:
            anchor=tokens[-1];offset=len(tokens)-1
            if mode=='serial':
                lg,ft=target.step(state,[anchor]);logs.append(lg.cpu());features.append(ft.cpu());nxt=int(lg[-1].argmax());tokens.append(nxt);stop=nxt in session.eos;continue
            limit=min(8,task['output_cap']-len(tokens));t=time.perf_counter()
            proposals,info=engine.propose(ctx,anchor) if limit>1 else ([],{});rates=[];width=limit
            if mode=='linear' and limit>1:
                x=feature_vector(info['signals'],len(ctx),len(tokens),task['output_cap']-len(tokens),False,'f1');rates=model.predict(x)
                c=costs['long' if len(ctx)>=768 else 'short'];width=choose_rows(rates,[v+c['paid_ms'] for v in c['verify_commit_ms']],limit)
            proposals=proposals[:width-1];torch.cuda.synchronize();dm=(time.perf_counter()-t)*1000
            t=time.perf_counter();lg,ft,b=target.run(state,[anchor]+proposals);vm=(time.perf_counter()-t)*1000
            pred=lg.argmax(-1).tolist();emitted,keep,_,stop=greedy_result(anchor,proposals,pred,session.eos,task['output_cap']-len(tokens))
            labels=observed_labels(proposals,pred,keep,session.eos)
            t=time.perf_counter();target.commit(state,b,width,keep);cm=(time.perf_counter()-t)*1000
            ctx=torch.cat([ctx,ft[:keep]]);tokens.extend(emitted)
            assert len(ctx)==state.pos==start.pos+len(tokens)-1
            if oracle is not None:
                assert emitted==oracle['tokens'][offset+1:offset+1+len(emitted)]
                for j in range(keep):
                    assert torch.equal(lg[j:j+1].cpu(),oracle['logits'][offset+j]) and torch.equal(ft[j:j+1].cpu(),oracle['features'][offset+j])
            steps.append(dict(context_rows=state.pos-keep,width=width,emitted=keep,accepted=sum(v for _,v in labels),observed=len(labels),rates=rates,
                draft_budget_ms=dm,verify_ms=vm,commit_ms=cm))
        torch.cuda.synchronize();elapsed=(time.perf_counter()-t0)*1000
        return dict(tokens=tokens,state=state,logits=logs,features=features,steps=steps,decode_ms=elapsed)
    for ti,task in enumerate(tasks):
        report['active']=dict(index=ti,id=task['id'],phase='prefill');save()
        ids,h=expected_prompt(session,task);start,ctx,lg=prompt_state(target,ids,chunk_size=8);first=int(lg[-1].argmax())
        assert h==reference[task['id']]['prompt_sha256'];report['active']['phase']='serial_reference';save();golden=generate(task,start,ctx,first,'serial')
        token_sha=hashlib.sha256(json.dumps(golden['tokens']).encode()).hexdigest();assert token_sha==reference[task['id']]['token_sha256']
        rec=dict(id=task['id'],group=task['group'],family=task['family'],language=task['language'],prompt_tokens=len(ids),prompt_sha256=h,
            completion_tokens=len(golden['tokens']),token_sha256=token_sha,checks=[],timings=[]);report['tasks'].append(rec)
        for mode in MODES:
            report['active'].update(phase='correctness_and_warmup',mode=mode);save();result=generate(task,start,ctx,first,mode,golden)
            assert result['tokens']==golden['tokens'] and not target.state_difference(result['state'],golden['state'])
            rec['checks'].append(dict(mode=mode,tokens_exact=True,committed_logits_exact=True,committed_features_exact=True,live_state_exact=True));del result
        for rep in range(2):
            for slot,mode in enumerate(order(ti,rep)):
                report['active'].update(phase='timing',mode=mode,repetition=rep,slot=slot);save();result=generate(task,start,ctx,first,mode)
                assert result['tokens']==golden['tokens'] and not target.state_difference(result['state'],golden['state'])
                rec['timings'].append(dict(mode=mode,repetition=rep,order=slot,decode_ms=result['decode_ms'],steps=result['steps'],exact=True));del result
            save()
        del start,ctx,golden
    check();report.update(status='PASS_E29_BUDGET',summary=summarize(report['tasks']),finished_unix=time.time());report.pop('active',None);save()
    return dict(status=report['status'],target_object_id=id(session.weights),draft_object_id=id(session.draft))
