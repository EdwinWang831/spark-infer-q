"""E33's eight roots: same optimized pipeline in legacy vs native Scheduler."""
import hashlib,importlib,json,random,sys,time
from pathlib import Path
from types import ModuleType

MODES=('legacy','native')
CONTRACT=dict(prompts=8,repetitions=4,modes=list(MODES),physical_capacity=2048,logical_capacity=1536,
    manifest='e28_test_tasks.json',profile='screened exact BF16 head + Graph + E28 W8 draft/Q4 head + E17 NumPy + E29 W8 costs',
    timing='decode after first token, including initial draft and all policy/verify/head/sync/commit/native bookkeeping/events',
    order='AB/BA alternates per root and repetition',scope='same eight engineering roots as E33, single-request greedy')


def summarize(tasks):
    assert len(tasks)==8 and len({t['group'] for t in tasks})==8
    totals=[]
    for task in tasks:
        assert len(task['timings'])==8
        for mode in MODES:
            assert sorted(t['repetition'] for t in task['timings'] if t['mode']==mode)==list(range(4))
        totals.append({m:sum(t['decode_ms'] for t in task['timings'] if t['mode']==m) for m in MODES})
    tokens=sum((t['completion_tokens']-1)*4 for t in tasks)
    rates={m:tokens*1000/sum(t[m] for t in totals) for m in MODES}
    rng=random.Random(2026102333);draws=[]
    for _ in range(4000):
        indices=rng.choices(range(8),k=8)
        draws.append(sum(totals[i]['legacy'] for i in indices)/sum(totals[i]['native'] for i in indices)-1)
    draws.sort()
    phases={}
    for mode in MODES:
        runs=[t for root in tasks for t in root['timings'] if t['mode']==mode]
        steps=[s for t in runs for s in t['steps']]
        phases[mode]=dict(rounds=len(steps),verified_rows=sum(s['width'] for s in steps),
            emitted=sum(s['emitted'] for s in steps),accepted=sum(s['accepted'] for s in steps),observed=sum(s['observed'] for s in steps),
            head_fallbacks=sum(s['fallback'] for s in steps),union_counts=[s['union_count'] for s in steps],
            request_ms=sum(t['request_ms'] for t in runs),ttft_ms=sum(t['ttft_ms'] for t in runs),
            phase_ms={k:sum(s[k] for s in steps) for k in ('draft_budget_ms','verify_ms','commit_ms')})
    return dict(decode_tps=rates,speed_gain=rates['native']/rates['legacy']-1,
        root_bootstrap_gain_95=[draws[99],draws[3899]],timed_tokens_per_arm=tokens,timed_runs=64,trace=phases,
        per_repetition=[dict(repetition=r,speed_gain=
            sum(t['decode_ms'] for root in tasks for t in root['timings'] if t['mode']=='legacy' and t['repetition']==r)/
            sum(t['decode_ms'] for root in tasks for t in root['timings'] if t['mode']=='native' and t['repetition']==r)-1) for r in range(4)])


def summarize_overhead(tasks):
    modes=('legacy','native_before','native')
    assert len(tasks)==8 and len({t['group'] for t in tasks})==8
    for task in tasks:
        assert len(task['timings'])==12
        assert {t['mode'] for t in task['timings']}==set(modes)
        for mode in modes:
            assert sorted(t['repetition'] for t in task['timings'] if t['mode']==mode)==list(range(4))
    # Reuse the audited two-arm accounting for each paired question. Bootstrap
    # resamples roots, keeping all repetitions/arms of a root together.
    result={}
    for label,reference in (('versus_legacy','legacy'),('versus_before','native_before')):
        paired=[]
        for task in tasks:
            timings=[dict(t,mode='legacy' if t['mode']==reference else 'native')
                     for t in task['timings'] if t['mode'] in (reference,'native')]
            paired.append(dict(task,timings=timings))
        result[label]=summarize(paired)
    tokens=result['versus_legacy']['timed_tokens_per_arm']
    result.update(timed_tokens_per_arm=tokens,timed_runs=96,decode_tps={mode:tokens*1000/sum(
        t['decode_ms'] for task in tasks for t in task['timings'] if t['mode']==mode) for mode in modes})
    return result


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    tag='_e33_online_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    candidate=None;legacy_target=None
    try:
        spec=importlib.import_module(tag+'.kaifra_speculative');opt=importlib.import_module(tag+'.kaifra_optimized')
        native_module=importlib.import_module(tag+'.native_scheduler')
        with session.torch.inference_mode():
            provider,metadata=opt.optimized_provider(session);candidate=opt.optimized_target(session);candidate.warmup()
        result=validate(session,spec,native_module,candidate,provider,root,probe=arguments.get('probe',False))
        result.update(source_sha256=manifest,metadata=metadata,model_reloaded=False)
        return result
    finally:
        if candidate is not None:
            if candidate.leased:candidate.release_state(candidate.workspace)
            candidate.close()
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)


def validate(session,spec,native_module,candidate,provider,output_root,*,probe=False,baseline=None):
    from e24_run import expected_prompt
    from e3_wide_target import WideTarget
    from e33_target import ScreenedGraphTarget
    from e33_fused_head import FusedScreenedHead
    from e28_probe import backend,sha
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_backbone import attach_mlp_w8
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.confidence_features import feature_vector
    from spark_infer_q.confidence_head import choose_rows
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.draft_signals import observed_labels
    torch,root=session.torch,session.root
    seal_path=root/'artifacts/e33-online-seal.json';seal=json.loads(seal_path.read_text())
    def check():
        for kind in ('source_sha256','input_sha256'):
            for name,digest in seal[kind].items():assert sha(root/name)==digest,name
        for name,digest in seal['external_source_sha256'].items():assert sha(root.parent/'e2-load/reference/tensorfold'/name)==digest,name
    check()
    eager=WideTarget(session.weights,capacity=2048)
    screen=FusedScreenedHead(session.weights.head)
    assert screen.static_ok
    for rows in range(1,9):screen.prepare(rows)
    legacy_target=ScreenedGraphTarget(session.weights,screen,capacity=2048)
    ext,w8_identity=backend(root);base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    draft=attach_mlp_w8(base,lambda x,p:ext.matmul(x,p.values,p.scales))
    model=NumpyLinearProgressModel(json.loads((root/'artifacts/e27-linear-model.json').read_text())['metadata'])
    costs=json.loads((root/'artifacts/e29-frozen-costs.json').read_text())['modes']['w8']
    tasks=json.loads((root/'e28_test_tasks.json').read_text())['tasks']
    previous={r['id']:r for r in json.loads((root/'artifacts/e33-online.json').read_text())['tasks']}
    assert len(tasks)==8
    if probe:tasks=[tasks[i] for i in (0,3)]
    modes=('legacy','native_before','native') if baseline is not None else MODES
    repetitions=1 if probe else 4
    identity=id(session.weights),id(session.draft)
    report=dict(status='RUNNING',contract=CONTRACT,tasks=[],started_unix=time.time(),
        legacy_seal_sha256=sha(seal_path),legacy_input_sha256=seal['input_sha256'],
        legacy_source_sha256=seal['source_sha256'],legacy_external_source_sha256=seal['external_source_sha256'],
        w8_identity={k:v for k,v in w8_identity.items() if k!='binary'},native=None,
        preparation=dict(native_graph=candidate.capabilities,legacy_head=screen.preparation),
        full_logit_rows=0,full_feature_rows=0,prefill_feature_rows=0,final_state_checks=0)
    if baseline is not None:report['contract']=dict(CONTRACT,modes=list(modes),order='rotate three arms by root/repetition, reverse on odd repetitions')
    if probe:report['contract']=dict(CONTRACT,prompts=2,repetitions=1,scope='instrumented two-root diagnostic; no promotion claim')
    def start_profile(gold):
        if not probe or gold is not None:return None
        import cProfile
        profiler=cProfile.Profile();profiler.enable();return profiler
    def finish_profile(profiler):
        if profiler is None:return {}
        profiler.disable()
        import pstats
        stats=pstats.Stats(profiler)
        rows=[dict(file=Path(key[0]).name,line=key[1],function=key[2],calls=v[1],self_ms=v[2]*1000,total_ms=v[3]*1000)
              for key,v in stats.stats.items()]
        return dict(profile=sorted(rows,key=lambda r:r['total_ms'],reverse=True)[:160])
    path=output_root/'progress.json'
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    def policy(steps):return [{k:s[k] for k in ('context_rows','width','emitted','accepted','observed','rates')} for s in steps]
    def serial(ids,cap):
        start,ctx,lg=prompt_state(eager,ids,chunk_size=8);state=start.clone();out=[int(lg[-1].argmax())];logs=[];fts=[]
        while len(out)<cap and out[-1] not in session.eos:
            lg,ft=eager.step(state,[out[-1]])
            logs.append(lg.cpu());fts.append(ft.cpu());out.append(int(lg[-1].argmax()))
        return dict(start=start,context=ctx,first=out[0],tokens=out,state=state,logits=logs,features=fts)
    def legacy(task,ids,gold=None):
        request_start=time.perf_counter();start,ctx,lg=prompt_state(eager,ids,chunk_size=8)
        state=legacy_target.bind(start);ctx=ctx.clone();out=[int(lg[-1].argmax())];steps=[]
        draft.begin((task['id'],'legacy',time.monotonic_ns()))
        initial=(legacy_target.captures,len(screen.cache),set(screen.kernel_sha256))
        torch.cuda.synchronize();profiler=start_profile(gold);t0=time.perf_counter()
        while len(out)<task['output_cap'] and out[-1] not in session.eos:
            anchor=out[-1];offset=len(out)-1;remaining=task['output_cap']-len(out);limit=min(8,remaining)
            proposals=[];rates=[];width=1;t=time.perf_counter()
            if limit>1:
                proposals,info=draft.propose(ctx,anchor)
                rates=model.predict(feature_vector(info['signals'],len(ctx),len(out),remaining,False,'f1'))
                c=costs['long' if len(ctx)>=768 else 'short'];width=choose_rows(rates,[v+c['paid_ms'] for v in c['verify_commit_ms']],limit)
            proposals=proposals[:width-1];torch.cuda.synchronize();dm=(time.perf_counter()-t)*1000;t=time.perf_counter()
            pred,ft,buf,hidden,info=legacy_target.run_predictions(state,[anchor]+proposals,diagnostic=gold is not None)
            vm=(time.perf_counter()-t)*1000;predictions=pred.tolist()
            emitted,keep,_,_=greedy_result(anchor,proposals,predictions,session.eos,remaining)
            labels=observed_labels(proposals,predictions,len(emitted),session.eos)
            t=time.perf_counter();legacy_target.commit(state,buf,width,keep);cm=(time.perf_counter()-t)*1000
            ctx=torch.cat([ctx,ft[:keep]]);out.extend(emitted)
            if gold is not None:
                full=screen.original(hidden);assert predictions==full.argmax(-1).tolist()
                if not info['fallback']:assert torch.equal(info['exact'],full[:,info['ids']])
                assert emitted==gold['tokens'][offset+1:offset+1+len(emitted)]
                for j in range(keep):
                    assert torch.equal(full[j:j+1].cpu(),gold['logits'][offset+j]);report['full_logit_rows']+=1
                    assert torch.equal(ft[j:j+1].cpu(),gold['features'][offset+j]);report['full_feature_rows']+=1
            steps.append(dict(context_rows=state.pos-keep,width=width,emitted=len(emitted),accepted=sum(v for _,v in labels),observed=len(labels),rates=rates,
                draft_budget_ms=dm,verify_ms=vm,commit_ms=cm,fallback=bool(info['fallback']),union_count=info.get('union_count')))
        torch.cuda.synchronize();end=time.perf_counter();profile_data=finish_profile(profiler)
        assert (legacy_target.captures,len(screen.cache))==initial[:2]
        if gold is None:assert set(screen.kernel_sha256)==initial[2],'new legacy subset JIT inside timing'
        return dict(**profile_data,tokens=out,state=state,steps=steps,decode_ms=(end-t0)*1000,request_ms=(end-request_start)*1000,ttft_ms=(t0-request_start)*1000)
    save()
    try:
        with native_module.NativeScheduler(root.parent/'DeepInfer',root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
            report['native']=native.metadata
            def generate_native(task,ids,gold=None,*,implementation=None):
                impl,target,source_provider=implementation or (spec,candidate,provider)
                e=None;captured=[]
                def observe(handle,pos,input_ids,lg,keep,state):
                    context=e.bridge._entry(handle)['state'].context
                    n=len(ids)
                    if pos<n:
                        assert pos+keep<=n and torch.equal(context[pos:pos+keep],gold['context'][pos:pos+keep])
                        report['prefill_feature_rows']+=keep
                    else:
                        offset=pos-n
                        for j in range(keep):
                            assert torch.equal(lg[j:j+1].cpu(),gold['logits'][offset+j]);report['full_logit_rows']+=1
                            assert torch.equal(context[pos+j:pos+j+1].cpu(),gold['features'][offset+j]);report['full_feature_rows']+=1
                e=impl.speculative_engine(native,target,source_provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],
                    mode='adaptive',capacity=1536,token_budget=32,observer=observe if gold is not None else None)
                release=e.bridge.backend.release
                def record_release(h,state):
                    captured.append(state.target)
                    release(h,state)
                e.bridge.backend.release=record_release
                initial=(target.captures,len(target.screen.cache),set(target.screen.kernel_sha256))
                try:
                    request_start=time.perf_counter();e.submit_tokens(task['id'],ids,max_tokens=task['output_cap'])
                    out=[];steps=[]
                    while not out:
                        for ev in e.step():out.extend(ev.token_ids)
                        assert not any(m['draft_ms'] for m in e.last_step_metrics)
                    assert len(out)==1
                    if gold is not None:
                        live=e.bridge.requests[task['id']]['state']
                        assert out==[gold['first']] and torch.equal(live.context,gold['context'])
                        assert not eager.state_difference(live.target,gold['start'])
                    torch.cuda.synchronize();profiler=start_profile(gold);t0=time.perf_counter()
                    while e.active_requests:
                        for ev in e.step():out.extend(ev.token_ids)
                        steps.extend(dict(context_rows=m['context_rows'],width=m['verified_rows'],emitted=m['emitted'],accepted=m['accepted'],observed=m['observed'],rates=m['rates'],
                            draft_budget_ms=m['draft_ms'],verify_ms=m['verify_ms'],commit_ms=m['commit_ms'],fallback=m['head_fallback'],union_count=m['head_union_count'])
                            for m in e.last_step_metrics if not m['prompt'])
                    torch.cuda.synchronize();end=time.perf_counter();profile_data=finish_profile(profiler)
                    assert len(captured)==1 and not target.leased
                    assert not e.bridge.backend.states and not e.bridge.native.has_unfinished_seqs()
                    assert (target.captures,len(target.screen.cache))==initial[:2]
                    if gold is None:assert set(target.screen.kernel_sha256)==initial[2],'new native subset JIT inside timing'
                    return dict(**profile_data,tokens=out,state=captured[0],steps=steps,decode_ms=(end-t0)*1000,
                        request_ms=(end-request_start)*1000,ttft_ms=(t0-request_start)*1000)
                finally:
                    e.close();del e.bridge.backend.release;e.bridge.backend.observer=None
            def generate(mode,task,ids,gold=None):
                if mode=='legacy':return legacy(task,ids,gold)
                return generate_native(task,ids,gold,implementation=baseline if mode=='native_before' else None)
            warmed=False
            for ti,task in enumerate(tasks):
                report['active']=dict(index=ti,id=task['id'],phase='prefill_oracle');save()
                ids,prompt_sha=expected_prompt(session,task);assert prompt_sha==previous[task['id']]['prompt_sha256']
                assert len(ids)+task['output_cap']<=1536
                gold=serial(ids,task['output_cap']);token_sha=hashlib.sha256(json.dumps(gold['tokens']).encode()).hexdigest()
                assert token_sha==previous[task['id']]['token_sha256']
                if not warmed:
                    t=time.perf_counter();legacy_target.warm(gold['start']);report['preparation']['legacy_graph_ms']=(time.perf_counter()-t)*1000;warmed=True
                record=dict(id=task['id'],group=task['group'],family=task['family'],language=task['language'],prompt_tokens=len(ids),prompt_sha256=prompt_sha,
                    completion_tokens=len(gold['tokens']),token_sha256=token_sha,checks=[],timings=[])
                report['tasks'].append(record);traces={}
                for mode in modes:
                    report['active'].update(phase='correctness_warmup',mode=mode);save()
                    result=generate(mode,task,ids,gold)
                    assert result['tokens']==gold['tokens'] and not eager.state_difference(result['state'],gold['state'])
                    report['final_state_checks']+=1;traces[mode]=policy(result['steps'])
                    record['checks'].append(dict(mode=mode,tokens_exact=True,committed_logits_exact=True,committed_features_exact=True,live_state_exact=True));del result
                assert all(traces['legacy']==traces[m] for m in modes),(task['id'],'budget trace mismatch')
                record['decision_trace_exact']=True
                for repetition in range(repetitions):
                    if baseline is None:
                        order=modes if (ti+repetition)%2==0 else tuple(reversed(modes))
                    else:
                        offset=(ti+repetition)%len(modes);order=modes[offset:]+modes[:offset]
                        if repetition%2:order=tuple(reversed(order))
                    for slot,mode in enumerate(order):
                        report['active'].update(phase='timing',mode=mode,repetition=repetition,slot=slot);save()
                        result=generate(mode,task,ids)
                        assert result['tokens']==gold['tokens'] and not eager.state_difference(result['state'],gold['state'])
                        assert policy(result['steps'])==traces['legacy']
                        report['final_state_checks']+=1
                        record['timings'].append(dict(mode=mode,repetition=repetition,order=slot,exact=True,
                            **{k:result[k] for k in ('decode_ms','request_ms','ttft_ms','steps','profile') if k in result}));del result
                    save();print(json.dumps(dict(k3e33_task=ti,repetition=repetition,status='PASS')),flush=True)
                del gold
            check();assert identity==(id(session.weights),id(session.draft))
            summary={} if probe else summarize_overhead(report['tasks']) if baseline is not None else summarize(report['tasks'])
            report.update(status='PASS_K3E33_PROFILE' if probe else 'PASS_K3E33_OVERHEAD_ONLINE' if baseline is not None else 'PASS_K3E33_MATCHED_ONLINE',summary=summary,finished_unix=time.time(),
                captures=dict(legacy=legacy_target.captures,native=candidate.captures),graph=candidate.capabilities,
                kernel_sha256=dict(legacy=screen.kernel_sha256,native=candidate.screen.kernel_sha256))
            if baseline is not None:
                report['captures']['native_before']=baseline[1].captures
                report['kernel_sha256']['native_before']=baseline[1].screen.kernel_sha256
            report.pop('active',None);save();return report
    finally:
        legacy_target.graphs.clear();legacy_target.outputs.clear();legacy_target.bound=None;screen.cache.clear()
