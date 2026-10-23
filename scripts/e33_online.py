"""Matched W8/linear full-decode comparison: original head vs fused screened head."""
import hashlib
import json
import random
import time
from e28_probe import sha,backend

MODES=('original','screened')
CONTRACT=dict(prompts=8,repetitions=4,capacity=2048,modes=list(MODES),manifest='e28_test_tasks.json',
    baseline='same E28 W8 draft, Q4 draft head, E17 NumPy linear and frozen E29 W8 fees; original full target Graph',
    candidate='original backbone/final mixer Graph returns hidden; fused FP64-bound Q8 screen + original BF16 subset/fallback outside backbone Graph',
    order='AB/BA alternating, four repetitions per prompt',
    timing='steady-state actual decode; all draft/predict/verify/screen/sync/commit/loop included; preparation/oracle/checks excluded',
    gate='all tokens/logits/features/live state/decision traces exact; >=1% decode gain; positive root95% lower bound and every repetition positive',
    scope='eight reused engineering roots, separate from four E33 component roots; greedy single request, no default/service promotion')


def summarize(tasks):
    assert len(tasks)==8
    total=[]
    for task in tasks:
        assert len(task['timings'])==8
        for mode in MODES:assert sorted(t['repetition'] for t in task['timings'] if t['mode']==mode)==list(range(4))
        total.append({m:sum(t['decode_ms'] for t in task['timings'] if t['mode']==m) for m in MODES})
    tokens=sum((t['completion_tokens']-1)*4 for t in tasks)
    tps={m:tokens*1000/sum(t[m] for t in total) for m in MODES};rng=random.Random(33331023);draws=[]
    for _ in range(4000):
        picks=rng.choices(range(8),k=8);draws.append(sum(total[i]['original'] for i in picks)/sum(total[i]['screened'] for i in picks)-1)
    draws.sort();rep=[]
    for r in range(4):
        sums={m:sum(t['decode_ms'] for p in tasks for t in p['timings'] if t['mode']==m and t['repetition']==r) for m in MODES}
        rep.append(dict(repetition=r,speed_gain=sums['original']/sums['screened']-1))
    trace={}
    for m in MODES:
        steps=[s for p in tasks for t in p['timings'] if t['mode']==m for s in t['steps']]
        trace[m]=dict(rounds=len(steps),verified_rows=sum(s['width'] for s in steps),emitted=sum(s['emitted'] for s in steps),
            accepted=sum(s['accepted'] for s in steps),observed=sum(s['observed'] for s in steps),
            phase_ms={k:sum(s[k] for s in steps) for k in ('draft_budget_ms','verify_ms','commit_ms')},
            fallbacks=sum(s.get('fallback',False) for s in steps),union_counts=[s['union_count'] for s in steps if s.get('union_count') is not None])
    return dict(decode_tps=tps,speed_gain=tps['screened']/tps['original']-1,root_bootstrap_gain_95=[draws[99],draws[3899]],per_repetition=rep,
        timed_tokens_per_arm=tokens,timed_runs=64,trace=trace)


def run(session,arguments):
    from e24_run import expected_prompt
    from e3_wide_target import WideTarget
    from e29_graph import FeatureGraphTarget
    from e33_target import ScreenedGraphTarget
    from e33_fused_head import FusedScreenedHead
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_backbone import attach_mlp_w8
    from spark_infer_q.confidence_features import feature_vector
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.confidence_head import choose_rows
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.draft_signals import observed_labels
    root,torch=session.root,session.torch;assert not arguments
    seal_path=root/'artifacts/e33-online-seal.json';seal=json.loads(seal_path.read_text());assert seal['contract']==CONTRACT
    def check():
        for key in ('source_sha256','input_sha256'):
            for n,h in seal[key].items():assert sha(root/n)==h,n
        for n,h in seal['external_source_sha256'].items():assert sha(root.parent/'e2-load/reference/tensorfold'/n)==h,n
    check();path=root/'artifacts/e33-online.json';assert not path.exists()
    report=dict(status='RUNNING',contract=CONTRACT,seal_sha256=sha(seal_path),started_unix=time.time(),tasks=[],target_object_id=id(session.weights),draft_object_id=id(session.draft))
    def save():
        p=path.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(path)
    save();screen=FusedScreenedHead(session.weights.head);assert screen.static_ok;report['head_preparation']=screen.preparation
    for rows in range(1,9):screen.prepare(rows)
    eager=WideTarget(session.weights,capacity=2048);original=FeatureGraphTarget(session.weights,capacity=2048);screened=ScreenedGraphTarget(session.weights,screen,capacity=2048)
    ext,identity=backend(root);report['w8_identity']=identity;base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048;draft=attach_mlp_w8(base,lambda x,p:ext.matmul(x,p.values,p.scales))
    model=NumpyLinearProgressModel(json.loads((root/'artifacts/e27-linear-model.json').read_text())['metadata']);costs=json.loads((root/'artifacts/e29-frozen-costs.json').read_text())['modes']['w8']
    tasks=json.loads((root/'e28_test_tasks.json').read_text())['tasks'];component_ids={t['id'] for t in json.loads((root/'e30_profile_tasks.json').read_text())['tasks']};assert not component_ids.intersection(t['id'] for t in tasks)
    references={t['id']:t for t in json.loads((root/'artifacts/e29-online.json').read_text())['tasks']}
    def policy(steps):return [{k:s[k] for k in ('context_rows','width','emitted','accepted','observed','rates')} for s in steps]
    def generate(task,start,context,first,mode,oracle=None):
        target=eager if mode=='serial' else original if mode=='original' else screened
        state=start.clone() if mode=='serial' else target.bind(start);ctx=context.clone();tokens=[first];steps=[];logs=[];features=[];stop=first in session.eos
        if mode!='serial':draft.begin((task['id'],mode,time.monotonic_ns()))
        initial=(original.captures,screened.captures,len(screen.cache),set(screen.kernel_sha256))
        torch.cuda.synchronize();t0=time.perf_counter()
        while len(tokens)<task['output_cap'] and not stop:
            anchor=tokens[-1];offset=len(tokens)-1
            if mode=='serial':
                lg,ft=eager.step(state,[anchor]);logs.append(lg.cpu());features.append(ft.cpu());nxt=int(lg[-1].argmax());tokens.append(nxt);stop=nxt in session.eos;continue
            limit=min(8,task['output_cap']-len(tokens));t=time.perf_counter()
            proposals,info=draft.propose(ctx,anchor) if limit>1 else ([],{});rates=[];width=1
            if limit>1:
                rates=model.predict(feature_vector(info['signals'],len(ctx),len(tokens),task['output_cap']-len(tokens),False,'f1'))
                c=costs['long' if len(ctx)>=768 else 'short'];width=choose_rows(rates,[v+c['paid_ms'] for v in c['verify_commit_ms']],limit)
            proposals=proposals[:width-1];torch.cuda.synchronize();dm=(time.perf_counter()-t)*1000;t=time.perf_counter()
            head_info={};hidden=None
            if mode=='original':lg,ft,buf=target.run(state,[anchor]+proposals)
            else:pred,ft,buf,hidden,head_info=target.run_predictions(state,[anchor]+proposals,diagnostic=oracle is not None)
            vm=(time.perf_counter()-t)*1000;predictions=lg.argmax(-1).tolist() if mode=='original' else pred.tolist()
            emitted,keep,_,stop=greedy_result(anchor,proposals,predictions,session.eos,task['output_cap']-len(tokens));labels=observed_labels(proposals,predictions,len(emitted),session.eos)
            t=time.perf_counter();target.commit(state,buf,width,keep);cm=(time.perf_counter()-t)*1000;ctx=torch.cat([ctx,ft[:keep]]);tokens.extend(emitted)
            assert len(ctx)==state.pos==start.pos+len(tokens)-1
            if oracle is not None:
                if mode=='screened':
                    lg=screen.original(hidden);assert predictions==lg.argmax(-1).tolist()
                    if not head_info['fallback']:assert torch.equal(head_info['exact'],lg[:,head_info['ids']])
                assert emitted==oracle['tokens'][offset+1:offset+1+len(emitted)]
                for j in range(keep):
                    assert torch.equal(lg[j:j+1].cpu(),oracle['logits'][offset+j])
                    assert torch.equal(ft[j:j+1].cpu(),oracle['features'][offset+j])
            steps.append(dict(context_rows=state.pos-keep,width=width,emitted=keep,accepted=sum(v for _,v in labels),observed=len(labels),rates=rates,
                draft_budget_ms=dm,verify_ms=vm,commit_ms=cm,fallback=bool(head_info.get('fallback',False)),union_count=head_info.get('union_count')))
        torch.cuda.synchronize();elapsed=(time.perf_counter()-t0)*1000
        if mode!='serial':assert (original.captures,screened.captures,len(screen.cache))==initial[:3]
        # Correctness warmup may JIT new dynamic subset counts; timed runs may not.
        if oracle is None and mode!='serial':assert set(screen.kernel_sha256)==initial[3],'new subset JIT inside timing'
        return dict(tokens=tokens,state=state,logits=logs,features=features,steps=steps,decode_ms=elapsed)
    warmed=False
    for ti,task in enumerate(tasks):
        report['active']=dict(index=ti,id=task['id'],phase='prefill');save();ids,prompt_sha=expected_prompt(session,task);assert prompt_sha==references[task['id']]['prompt_sha256']
        start,ctx,lg=prompt_state(eager,ids,chunk_size=8);first=int(lg[-1].argmax())
        if not warmed:
            t=time.perf_counter();original.warm(start);screened.warm(start);report['graph_preparation_ms']=(time.perf_counter()-t)*1000;warmed=True
        report['active']['phase']='serial_oracle';save();gold=generate(task,start,ctx,first,'serial');token_sha=hashlib.sha256(json.dumps(gold['tokens']).encode()).hexdigest();assert token_sha==references[task['id']]['token_sha256']
        record=dict(id=task['id'],group=task['group'],family=task['family'],language=task['language'],prompt_tokens=len(ids),prompt_sha256=prompt_sha,
            completion_tokens=len(gold['tokens']),token_sha256=token_sha,checks=[],timings=[]);report['tasks'].append(record)
        traces={}
        for mode in MODES:
            report['active'].update(phase='correctness_warmup',mode=mode);save();result=generate(task,start,ctx,first,mode,gold)
            assert result['tokens']==gold['tokens'] and not eager.state_difference(result['state'],gold['state']);traces[mode]=policy(result['steps'])
            record['checks'].append(dict(mode=mode,tokens_exact=True,committed_logits_exact=True,committed_features_exact=True,live_state_exact=True,head_argmax_exact=True));del result
        assert traces['original']==traces['screened'];record['decision_trace_exact']=True
        for rep in range(4):
            order=MODES if (ti+rep)%2==0 else tuple(reversed(MODES))
            for slot,mode in enumerate(order):
                report['active'].update(phase='timing',mode=mode,repetition=rep,slot=slot);save();result=generate(task,start,ctx,first,mode)
                assert result['tokens']==gold['tokens'] and not eager.state_difference(result['state'],gold['state']) and policy(result['steps'])==traces['original']
                record['timings'].append(dict(mode=mode,repetition=rep,order=slot,decode_ms=result['decode_ms'],steps=result['steps'],exact=True));del result
            save();print(json.dumps(dict(e33_online=ti,repetition=rep)),flush=True)
        del start,ctx,gold
    check();summary=summarize(report['tasks']);passed=summary['speed_gain']>=.01 and summary['root_bootstrap_gain_95'][0]>0 and all(r['speed_gain']>0 for r in summary['per_repetition'])
    report.update(status='PASS_E33_ONLINE_GAIN' if passed else 'PASS_E33_ONLINE_NO_CONFIRMED_GAIN',summary=summary,finished_unix=time.time(),kernel_sha256=screen.kernel_sha256,captures=dict(original=original.captures,screened=screened.captures));report.pop('active',None);save()
    return dict(status=report['status'],target_object_id=id(session.weights),draft_object_id=id(session.draft))
