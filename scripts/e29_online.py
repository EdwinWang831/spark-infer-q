"""Three-arm steady-state target verification A/B, with frozen calibration costs."""
import hashlib
import json
import random
import time

from e28_probe import sha,backend

MODES=('eager_bf16','graph_bf16','graph_w8')
CONTRACT=dict(prompts=8,repetitions=4,modes=list(MODES),capacity=2048,output_cap=384,
    policy='frozen E17 NumPy linear; costs frozen from separate E29 calibration roots; no online fitting',
    primary='graph_bf16 versus eager_bf16: identical draft, probabilities and frozen costs',
    secondary='graph_w8 versus eager_bf16: combined draft quantization, calibrated costs and graph execution',
    order='each pair has two forward and two reverse orders within each prompt',
    timing='steady-state decode after first token; all draft/F1/predict/budget/verify/commit/loop included',
    exclusions='prefill, first token, initial state copy, graph capture/warmup, correctness checks and disk writes',
    limits=['eight reused E28 engineering test roots, not fresh independent traffic',
        'single request, greedy; no batching, cold-start, random sampling or default promotion',
        'calibration hindsight oracle never used as online signal; BF16 controls coexist'])


def order(ti,rep):
    base=list(MODES);offset=(ti+rep//2)%3;base=base[offset:]+base[:offset]
    return base if rep%2==0 else list(reversed(base))


def summarize(tasks,samples=4000):
    assert len(tasks)==len({t['group'] for t in tasks})==8
    totals=[]
    for t in tasks:
        assert len(t['timings'])==12
        row={}
        for mode in MODES:
            ts=[v for v in t['timings'] if v['mode']==mode]
            assert sorted(v['repetition'] for v in ts)==list(range(4)) and all(v['exact'] for v in ts)
            row[mode]=sum(v['decode_ms'] for v in ts)
        totals.append(row)
    def rates(indices):
        tokens=sum((tasks[i]['completion_tokens']-1)*4 for i in indices)
        return {m:tokens*1000/sum(totals[i][m] for i in indices) for m in MODES}
    point=rates(list(range(8)));rng=random.Random(29261023);draws={m:[] for m in MODES[1:]}
    for _ in range(samples):
        v=rates(rng.choices(range(8),k=8))
        for m in draws:draws[m].append(v[m]/v['eager_bf16']-1)
    comparisons={}
    for m,v in draws.items():
        v.sort();comparisons[m]=dict(relative_gain=point[m]/point['eager_bf16']-1,
            prompt_bootstrap_gain_95=[v[int(.025*(samples-1))],v[int(.975*(samples-1))]])
    trace={}
    for m in MODES:
        runs=[v for t in tasks for v in t['timings'] if v['mode']==m];steps=[s for v in runs for s in v['steps']]
        trace[m]=dict(decode_ms=sum(v['decode_ms'] for v in runs),rounds=len(steps),
            emitted=sum(s['emitted'] for s in steps),verified_rows=sum(s['width'] for s in steps),
            accepted=sum(s['accepted'] for s in steps),observed=sum(s['observed'] for s in steps),
            phase_ms={k:sum(s[k] for s in steps) for k in ('draft_budget_ms','verify_ms','commit_ms')},
            width_counts={str(w):sum(s['width']==w for s in steps) for w in range(1,9)})
    return dict(decode_tps=point,versus_eager_bf16=comparisons,trace=trace,
        prompts=8,timed_runs=96,timed_tokens_per_arm=sum((t['completion_tokens']-1)*4 for t in tasks),
        per_repetition=[dict(repetition=r,relative_gain={m:
            sum(v['decode_ms'] for t in tasks for v in t['timings'] if v['mode']=='eager_bf16' and v['repetition']==r)/
            sum(v['decode_ms'] for t in tasks for v in t['timings'] if v['mode']==m and v['repetition']==r)-1 for m in MODES[1:]}) for r in range(4)])


def run(session,arguments):
    from e24_run import expected_prompt
    from e3_wide_target import WideTarget
    from e29_graph import FeatureGraphTarget
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.confidence_features import ConfidenceDraft,feature_vector
    from spark_infer_q.draft_backbone import attach_mlp_w8
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.confidence_head import choose_rows
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.draft_signals import observed_labels
    from spark_infer_q.prompt import prompt_state
    root,torch=session.root,session.torch;assert not arguments
    seal_path=root/'artifacts/e29-online-seal.json';seal=json.loads(seal_path.read_text());assert seal['contract']==CONTRACT
    def check():
        for kind in ('source_sha256','input_sha256'):
            for n,h in seal[kind].items():assert sha(root/n)==h,n
        for n,h in seal['external_source_sha256'].items():
            assert sha(root.parent/'e2-load/reference/tensorfold'/n)==h,n
    check();path=root/'artifacts/e29-online.json';assert not path.exists()
    report=dict(status='RUNNING',started_unix=time.time(),seal_sha256=sha(seal_path),contract=CONTRACT,tasks=[],
        target_object_id=id(session.weights),draft_object_id=id(session.draft),cpu_threads=torch.get_num_threads())
    def save():
        p=path.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(path)
    save();eager=WideTarget(session.weights,capacity=2048);graph=FeatureGraphTarget(session.weights,capacity=2048)
    ext,identity=backend(root);report['kernel_identity']=identity
    base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    engines=dict(bf16=ConfidenceDraft.attach(base),w8=attach_mlp_w8(base,lambda x,p:ext.matmul(x,p.values,p.scales)))
    model=NumpyLinearProgressModel(json.loads((root/'artifacts/e27-linear-model.json').read_text())['metadata'])
    costs=json.loads((root/'artifacts/e29-frozen-costs.json').read_text());assert costs['source_sha256']==sha(root/'artifacts/e29-sweep.json')
    old=json.loads((root/'artifacts/e28-online.json').read_text());references={t['id']:t for t in old['tasks']}
    tasks=json.loads((root/'e28_test_tasks.json').read_text())['tasks'];assert len(tasks)==8
    def generate(task,start,context,first,mode,oracle=None):
        target=eager if mode in ('serial','eager_bf16') else graph
        state=start.clone() if target is eager else graph.bind(start)
        ctx=context.clone();tokens=[first];steps=[];logs=[];features=[];stop=first in session.eos
        engine=None if mode=='serial' else engines['w8' if mode=='graph_w8' else 'bf16']
        if engine is not None:engine.begin((task['id'],mode,time.monotonic_ns()))
        initial_captures=graph.captures
        torch.cuda.synchronize();t0=time.perf_counter()
        while len(tokens)<task['output_cap'] and not stop:
            anchor=tokens[-1];offset=len(tokens)-1
            if mode=='serial':
                lg,ft=target.step(state,[anchor],capture=True);logs.append(lg.cpu());features.append(ft.cpu());nxt=int(lg[-1].argmax());tokens.append(nxt);stop=nxt in session.eos;continue
            limit=min(8,task['output_cap']-len(tokens));t=time.perf_counter()
            proposals,info=engine.propose(ctx,anchor) if limit>1 else ([],{})
            rates=[];width=1
            if limit>1:
                x=feature_vector(info['signals'],len(ctx),len(tokens),task['output_cap']-len(tokens),False,'f1');rates=model.predict(x)
                c=costs['modes']['w8' if mode=='graph_w8' else 'bf16']['long' if len(ctx)>=768 else 'short']
                width=choose_rows(rates,[v+c['paid_ms'] for v in c['verify_commit_ms']],limit)
            proposals=proposals[:width-1];torch.cuda.synchronize();dm=(time.perf_counter()-t)*1000
            t=time.perf_counter();lg,ft,b=target.run(state,[anchor]+proposals);vm=(time.perf_counter()-t)*1000
            pred=lg.argmax(-1).tolist();emitted,keep,_,stop=greedy_result(anchor,proposals,pred,session.eos,task['output_cap']-len(tokens))
            labels=observed_labels(proposals,pred,len(emitted),session.eos)
            t=time.perf_counter();target.commit(state,b,width,keep);cm=(time.perf_counter()-t)*1000
            ctx=torch.cat([ctx,ft[:keep]]);tokens.extend(emitted)
            assert len(ctx)==state.pos==start.pos+len(tokens)-1
            if oracle is not None:
                assert emitted==oracle['tokens'][offset+1:offset+1+len(emitted)]
                for j in range(keep):
                    assert torch.equal(lg[j:j+1].cpu(),oracle['logits'][offset+j])
                    assert torch.equal(ft[j:j+1].cpu(),oracle['features'][offset+j])
            steps.append(dict(context_rows=state.pos-keep,width=width,emitted=keep,accepted=sum(v for _,v in labels),observed=len(labels),
                rates=rates,draft_budget_ms=dm,verify_ms=vm,commit_ms=cm))
        torch.cuda.synchronize();elapsed=(time.perf_counter()-t0)*1000
        if mode!='serial':assert graph.captures==initial_captures,'unexpected capture within decode'
        return dict(tokens=tokens,state=state,logits=logs,features=features,steps=steps,decode_ms=elapsed)
    warm=False
    for ti,task in enumerate(tasks):
        report['active']=dict(index=ti,id=task['id'],phase='prefill');save()
        ids,prompt_sha=expected_prompt(session,task);assert prompt_sha==references[task['id']]['prompt_sha256']
        start,ctx,lg=prompt_state(eager,ids,chunk_size=8);first=int(lg[-1].argmax())
        if not warm:
            report['active']['phase']='graph_warmup';save();torch.cuda.synchronize();t=time.perf_counter();graph.warm(start)
            report['graph_preparation']=dict(ms=(time.perf_counter()-t)*1000,captures=graph.captures,
                allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved());assert graph.captures==16;warm=True
        report['active']['phase']='serial_reference';save();golden=generate(task,start,ctx,first,'serial')
        token_sha=hashlib.sha256(json.dumps(golden['tokens']).encode()).hexdigest();assert token_sha==references[task['id']]['token_sha256']
        rec=dict(id=task['id'],group=task['group'],language=task['language'],family=task['family'],prompt_tokens=len(ids),
            prompt_sha256=prompt_sha,completion_tokens=len(golden['tokens']),token_sha256=token_sha,checks=[],timings=[])
        report['tasks'].append(rec)
        for mode in MODES:
            report['active'].update(phase='correctness_and_warmup',mode=mode);save();result=generate(task,start,ctx,first,mode,golden)
            assert result['tokens']==golden['tokens'] and not eager.state_difference(result['state'],golden['state'])
            rec['checks'].append(dict(mode=mode,tokens_exact=True,committed_logits_exact=True,committed_features_exact=True,live_state_exact=True));del result
        for rep in range(4):
            for slot,mode in enumerate(order(ti,rep)):
                report['active'].update(phase='timing',mode=mode,repetition=rep,slot=slot);save()
                result=generate(task,start,ctx,first,mode)
                assert result['tokens']==golden['tokens'] and not eager.state_difference(result['state'],golden['state'])
                rec['timings'].append(dict(mode=mode,repetition=rep,order=slot,decode_ms=result['decode_ms'],steps=result['steps'],exact=True));del result
            save();print(json.dumps(dict(e29_online=ti,repetition=rep)),flush=True)
        del start,ctx,golden
    check();report.update(status='PASS_E29_ONLINE',summary=summarize(report['tasks']),finished_unix=time.time(),captures=graph.captures);report.pop('active',None);save()
    return dict(status=report['status'],target_object_id=id(session.weights),draft_object_id=id(session.draft))
