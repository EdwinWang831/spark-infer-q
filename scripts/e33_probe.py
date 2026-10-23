"""Resident feasibility gate for a screened target head; no target dispatch replacement."""
import json
import random
import statistics
import time
from e28_probe import sha,backend

CONTRACT=dict(roots=4,rounds=8,capacity=2048,repetitions=6,primitive_calls=10,max_union=2048,
    baseline='frozen W8 draft / E17 NumPy linear / E29 W8 fees / full original target Graph',
    candidate='one per-output symmetric INT8 RTN helper, BF16 hidden, FP32 scale, FP64 outward bounds, original BF16 subset',
    bound='L2 quantization + both gamma(2K+1,u=2^-23) accumulation paths + BF16 output rounding; finite magnitude [2^-48,2^16] or zero',
    gate='all bounds/subset/argmax checks; stop if Q8-only aggregate not faster; else complete implemented head >=10% time reduction, positive root-bootstrap lower bound, no root >2% slower',
    timing='primitive CUDA-event Graph replay separately; full head uses synchronized host wall, includes dynamic checks/nonzero/sync/subset/fallback/argmax',
    scope='32 early real hidden windows on four reused calibration roots, greedy component evidence, not full decode TPS')


def summarize(cases,area,modes):
    roots=sorted({c['id'] for c in cases});assert len(roots)==4
    totals=[]
    for r in roots:
        totals.append({m:sum(statistics.median(t['ms'] for t in c[area] if t['mode']==m) for c in cases if c['id']==r) for m in modes})
    a=sum(t[modes[0]] for t in totals);b=sum(t[modes[1]] for t in totals);rng=random.Random(33261023);draws=[]
    for _ in range(4000):
        picks=rng.choices(range(4),k=4);draws.append(sum(totals[i][modes[0]] for i in picks)/sum(totals[i][modes[1]] for i in picks)-1)
    draws.sort();return dict(sum_median_ms={m:sum(t[m] for t in totals) for m in modes},speed_gain=a/b-1,time_reduction=1-b/a,
        root_bootstrap_gain_95=[draws[99],draws[3899]],per_root=[dict(id=r,**t,speed_gain=t[modes[0]]/t[modes[1]]-1) for r,t in zip(roots,totals)])


def run(session,arguments):
    from e24_run import expected_prompt
    from e3_wide_target import WideTarget
    from e29_graph import FeatureGraphTarget
    from e33_head_screen import ScreenedHead
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_backbone import attach_mlp_w8
    from spark_infer_q.confidence_features import feature_vector
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.confidence_head import choose_rows
    from spark_infer_q.speculative import greedy_result
    root,torch=session.root,session.torch;assert not arguments
    seal_path=root/'artifacts/e33-seal.json';seal=json.loads(seal_path.read_text());assert seal['contract']==CONTRACT
    def check():
        for key in ('source_sha256','input_sha256'):
            for n,h in seal[key].items():assert sha(root/n)==h,n
        for n,h in seal['external_source_sha256'].items():assert sha(root.parent/'e2-load/reference/tensorfold'/n)==h,n
    check();path=root/'artifacts/e33-probe.json';assert not path.exists()
    report=dict(status='RUNNING',contract=CONTRACT,started_unix=time.time(),seal_sha256=sha(seal_path),cases=[],preflight=[],
        target_object_id=id(session.weights),draft_object_id=id(session.draft),torch=torch.__version__)
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    def finish(status):
        check();report.update(status=status,finished_unix=time.time(),kernel_sha256=screen.kernel_sha256);report.pop('active',None);save()
        return dict(status=status,target_object_id=id(session.weights),draft_object_id=id(session.draft))
    save();assert session.weights.head.kernel=='b16' and session.weights.comm is None
    t=time.perf_counter();screen=ScreenedHead(session.weights.head);report['preparation']=screen.preparation;save()
    if not screen.static_ok:return finish('REJECT_E33_STATIC_RANGE')
    generator=torch.Generator(device='cuda').manual_seed(33261023)
    fixed=torch.tensor([0,1,63,64,127,screen.n//2,screen.n-1],dtype=torch.int64,device='cuda')
    for rows in range(1,9):
        x=torch.randn((rows,screen.k),device='cuda',dtype=torch.bfloat16,generator=generator);gold=screen.original(x)
        subset=screen.subset(x,fixed);assert torch.equal(subset,gold[:,fixed])
        pred,info=screen.select(x,diagnostic=True);assert torch.equal(pred,gold.argmax(-1))
        _,_,outputs=screen.cache[rows];approx,bounds,mask,_,counts,valid=outputs
        assert bool(valid);error=(gold.double()-approx.double()).abs();assert bool((error<=bounds).all())
        if not info['fallback']:assert torch.equal(info['exact'],gold[:,info['ids']])
        report['preflight'].append(dict(rows=rows,subset_exact=True,argmax_exact=True,bound_max_ratio=float((error/bounds).max()),fallback=info['fallback'],union_count=info['union_count']))
        save()
    for mode in ('forced_capacity','zero_ties','nonfinite'):
        x=torch.randn((1,screen.k),device='cuda',dtype=torch.bfloat16,generator=generator)
        if mode=='zero_ties':x.zero_()
        if mode=='nonfinite':x[0,0]=float('nan')
        gold=screen.original(x).argmax(-1);pred,info=screen.select(x,force_cap=0 if mode=='forced_capacity' else None)
        assert info['fallback'] and torch.equal(pred,gold)
        if mode=='zero_ties':assert int(pred[0])==0 and info['union_count']==screen.n
        report['preflight'].append(dict(mode=mode,argmax_exact=True,**info));save()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x=torch.randn((3,screen.k),device='cuda',dtype=torch.bfloat16,generator=generator);gold=screen.original(x);pred,info=screen.select(x,diagnostic=True)
    stream.synchronize();assert torch.equal(pred,gold.argmax(-1));report['separate_stream_exact']=True;save()
    eager=WideTarget(session.weights,capacity=2048);target=FeatureGraphTarget(session.weights,capacity=2048)
    ext,identity=backend(root);report['w8_identity']=identity;base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    draft=attach_mlp_w8(base,lambda x,p:ext.matmul(x,p.values,p.scales))
    model=NumpyLinearProgressModel(json.loads((root/'artifacts/e27-linear-model.json').read_text())['metadata']);costs=json.loads((root/'artifacts/e29-frozen-costs.json').read_text())['modes']['w8']
    tasks=json.loads((root/'e30_profile_tasks.json').read_text())['tasks'];saved_inputs=[];warmed=False
    for ti,task in enumerate(tasks):
        report['active']=dict(index=ti,phase='prefill');save();ids,prompt_sha=expected_prompt(session,task);start,ctx,lg=prompt_state(eager,ids,chunk_size=8)
        if not warmed:target.warm(start);warmed=True
        state=target.bind(start);tokens=[int(lg[-1].argmax())];draft.begin((task['id'],time.monotonic_ns()))
        for step in range(8):
            if tokens[-1] in session.eos or len(tokens)>=task['output_cap']:break
            proposals,signals=draft.propose(ctx,tokens[-1]);rates=model.predict(feature_vector(signals['signals'],len(ctx),len(tokens),task['output_cap']-len(tokens),False,'f1'))
            c=costs['long' if len(ctx)>=768 else 'short'];width=choose_rows(rates,[v+c['paid_ms'] for v in c['verify_commit_ms']],min(8,task['output_cap']-len(tokens)));proposals=proposals[:width-1]
            lg,ft,buf=target.run(state,[tokens[-1]]+proposals);x=buf.mixed[:width].clone();gold=screen.original(x);assert torch.equal(gold,lg)
            pred,info=screen.select(x,diagnostic=True);assert torch.equal(pred,gold.argmax(-1))
            _,_,outputs=screen.cache[width];approx,bounds,mask,_,counts,valid=outputs
            if not bool(valid):return finish('REJECT_E33_REAL_RANGE')
            error=(gold.double()-approx.double()).abs();violation=int((error>bounds).sum())
            if violation:
                report['failure']=dict(id=task['id'],step=step,bound_violations=violation,max_ratio=float((error/bounds).max()));return finish('REJECT_E33_BOUND')
            exact=info['fallback'] or torch.equal(info['exact'],gold[:,info['ids']]);assert exact
            kept=mask.gather(1,gold.argmax(-1)[:,None]);assert bool(kept.all())
            record=dict(id=task['id'],family=task['family'],language=task['language'],step=step,context_rows=state.pos,rows=width,prompt_tokens=len(ids),prompt_sha256=prompt_sha,
                bound_max_ratio=float((error/bounds).max()),bound_violations=0,argmax_exact=True,original_winner_retained=True,subset_checked=not info['fallback'],subset_exact=bool(exact),
                union_count=info['union_count'],per_row=counts.cpu().tolist(),fallback=info['fallback'],fallback_reason=info.get('reason'),primitive_timings=[])
            report['cases'].append(record);save();saved_inputs.append((record,x,gold.argmax(-1).clone()))
            graphs={}
            for mode in ('bf16','q8'):
                fn=screen.original if mode=='bf16' else screen.approx
                fn(x);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g,capture_error_mode='thread_local'):out=fn(x)
                graphs[mode]=(g,out);g.replay()
            torch.cuda.synchronize()
            for rep in range(6):
                for slot,mode in enumerate(('bf16','q8') if rep%2==0 else ('q8','bf16')):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
                    for _ in range(10):graphs[mode][0].replay()
                    end.record();end.synchronize();record['primitive_timings'].append(dict(mode=mode,repetition=rep,order=slot,ms=begin.elapsed_time(end)/10))
            save();emitted,keep,_,stop=greedy_result(tokens[-1],proposals,lg.argmax(-1).tolist(),session.eos,task['output_cap']-len(tokens));target.commit(state,buf,width,keep);ctx=torch.cat([ctx,ft[:keep]]);tokens.extend(emitted)
            print(json.dumps(dict(e33_case=len(report['cases']),rows=width,candidates=record['union_count'],fallback=record['fallback'])),flush=True)
            if stop:break
        del start,state,ctx
    assert len(report['cases'])==32
    report['primitive_summary']=summarize(report['cases'],'primitive_timings',('bf16','q8'));save()
    if report['primitive_summary']['speed_gain']<=0:return finish('REJECT_E33_APPROX_COST')
    for record,x,oracle in saved_inputs:
        # Warm every dynamic subset shape before full implemented-path timings.
        for _ in range(3):pred,_=screen.select(x);assert torch.equal(pred,oracle)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,capture_error_mode='thread_local'):baseline=screen.original(x).argmax(-1)
        for _ in range(3):graph.replay()
        torch.cuda.synchronize();record['full_timings']=[]
        for rep in range(6):
            for slot,mode in enumerate(('bf16','screened') if rep%2==0 else ('screened','bf16')):
                torch.cuda.synchronize();t=time.perf_counter()
                if mode=='bf16':graph.replay();pred=baseline;info={}
                else:pred,info=screen.select(x)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
                assert torch.equal(pred,oracle)
                record['full_timings'].append(dict(mode=mode,repetition=rep,order=slot,ms=elapsed,fallback=info.get('fallback',False)))
        save()
    report['full_summary']=summarize(report['cases'],'full_timings',('bf16','screened'));gate=report['full_summary']
    passed=gate['time_reduction']>=.1 and gate['root_bootstrap_gain_95'][0]>0 and min(r['speed_gain'] for r in gate['per_root'])>=-.02
    return finish('PASS_E33_HEAD_GATE' if passed else 'REJECT_E33_FULL_COST')
