"""Exact real-input MoE overlap gate and separately instrumented stream evidence."""
import copy
import json
import random
import statistics
import time
from e28_probe import sha,backend

CONTRACT=dict(roots=4,rounds=2,layers=[0,23,47],capacity=2048,repetitions=6,
    modes=['serial','overlap'],regimes=['warm_reuse','evict_128MiB'],
    calls_per_interval=dict(warm_reuse=10,evict_128MiB=1),
    baseline='E33 screened target Graph / W8 draft / E17 NumPy / frozen E29 fees',
    change='shared expert on held side stream, routed branch on main; fork/join, same kernels and original output slots',
    gate='all exact/guard/state gates; >=5% evicted complete-MoE time reduction; root95 lower>0; no root slower>2%',
    timing='paired whole-MoE CUDA Graph event intervals; all stream dependencies included, capture/check/eviction excluded',
    scope='24 real early-window MoEs on four reused roots; not full decode speed; no weight byte reduction claimed')


def summarize(cases):
    roots=sorted({c['id'] for c in cases})
    assert len(roots)==4 and len(cases)==24
    result={}
    for regime in CONTRACT['regimes']:
        totals=[{m:sum(statistics.median(t['ms'] for t in c['timings'] if t['mode']==m and t['regime']==regime)
            for c in cases if c['id']==r) for m in CONTRACT['modes']} for r in roots]
        a=sum(t['serial'] for t in totals);b=sum(t['overlap'] for t in totals)
        rng=random.Random(35261023);draws=[]
        for _ in range(4000):
            picks=rng.choices(range(4),k=4)
            draws.append(sum(totals[i]['serial'] for i in picks)/sum(totals[i]['overlap'] for i in picks)-1)
        draws.sort()
        result[regime]=dict(sum_median_ms={m:sum(t[m] for t in totals) for m in CONTRACT['modes']},
            speed_gain=a/b-1,time_reduction=1-b/a,root_bootstrap_gain_95=[draws[99],draws[3899]],
            per_root=[dict(id=r,**t,speed_gain=t['serial']/t['overlap']-1) for r,t in zip(roots,totals)])
    return result


def run(session,arguments):
    from e24_run import expected_prompt
    from e3_wide_target import WideTarget
    from e33_target import ScreenedGraphTarget
    from e33_fused_head import FusedScreenedHead
    from e32_probe_v2 import clone_module
    from e35_overlap import OverlappedMoE
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_backbone import attach_mlp_w8
    from spark_infer_q.confidence_features import feature_vector
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.confidence_head import choose_rows
    from spark_infer_q.speculative import greedy_result
    from tensorfold.cuda import moe as moe_mod
    from tensorfold.families.qwen4_exp.cuda import nvfp4_moe
    from tensorfold.families.qwen4_exp.cuda.state import _MoECfg
    torch,root=session.torch,session.root
    assert not arguments
    seal_path=root/'artifacts/e35-seal.json'
    seal=json.loads(seal_path.read_text());assert seal['contract']==CONTRACT
    def check():
        for key in ('source_sha256','input_sha256'):
            for n,h in seal[key].items():assert sha(root/n)==h,n
        for n,h in seal['external_source_sha256'].items():
            assert sha(root.parent/'e2-load/reference/tensorfold'/n)==h,n
    check()
    path=root/'artifacts/e35-probe.json';assert not path.exists()
    report=dict(status='RUNNING',contract=CONTRACT,seal_sha256=sha(seal_path),started_unix=time.time(),
        target_object_id=id(session.weights),draft_object_id=id(session.draft),preflight=[],cases=[],trajectory=[],traces=[])
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    save()
    overlap=OverlappedMoE(nvfp4_moe)
    cfg=_MoECfg(session.weights.cfg)
    def snapshots(buf,rows):
        return {k:getattr(buf,k)[:rows].clone() for k in ('act','y','pick','wts','logits')}
    def exact(buf,gold,rows):
        for k,v in gold.items():assert torch.equal(getattr(buf,k)[:rows],v),k
    def call(mode,x,xs,router,ex,buf,cfg):
        return (nvfp4_moe.moe if mode=='serial' else overlap)(x,xs,router,ex,buf,cfg)
    generator=torch.Generator(device='cuda').manual_seed(35261023)
    for li in CONTRACT['layers']:
        layer=session.weights.layers[li]
        for rows in range(1,9):
            for strided in (False,True):
                for prefill in (False,True):
                    x=torch.randn((rows,session.weights.cfg.hidden+(8 if strided else 0)),device='cuda',dtype=torch.bfloat16,generator=generator)[:,:session.weights.cfg.hidden]
                    buf=moe_mod.MoEBuffers(rows+1,cfg,x.device,prefill=prefill)
                    for k in ('act','y','logits','wts'):getattr(buf,k).fill_(13)
                    buf.pick.fill_(-17)
                    call('serial',x,None,layer.moe.router,layer.moe.experts,buf,cfg)
                    torch.cuda.synchronize();gold=snapshots(buf,rows)
                    call('overlap',x,None,layer.moe.router,layer.moe.experts,buf,cfg)
                    torch.cuda.synchronize();exact(buf,gold,rows)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,capture_error_mode='thread_local'):
                        call('overlap',x,None,layer.moe.router,layer.moe.experts,buf,cfg)
                    for _ in range(2):graph.replay()
                    torch.cuda.synchronize();exact(buf,gold,rows)
                    for k in ('act','y','logits','wts'):assert bool((getattr(buf,k)[rows:]==13).all()),k
                    assert bool((buf.pick[rows:]==-17).all())
                    report['preflight'].append(dict(layer=li,rows=rows,strided=strided,prefill=prefill,outputs_graph_exact=True,guards_exact=True))
                    del graph,buf,gold
            save()
    # Reuse the helper from a caller stream distinct from the original main.
    buf=moe_mod.MoEBuffers(3,cfg,session.weights.device)
    x=torch.randn((3,session.weights.cfg.hidden),device='cuda',dtype=torch.bfloat16,generator=generator)
    call('serial',x,None,layer.moe.router,layer.moe.experts,buf,cfg);torch.cuda.synchronize();gold=snapshots(buf,3)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):call('overlap',x,None,layer.moe.router,layer.moe.experts,buf,cfg)
    stream.synchronize();exact(buf,gold,3);report['separate_caller_stream_exact']=True
    del buf,x,gold
    save()
    flush=torch.zeros(128*1024**2//4,dtype=torch.int32,device='cuda')
    current={}
    def probe(x,xs,router,ex,buf,cfg):
        if current['layer'] not in CONTRACT['layers']:return
        rows=x.shape[0];gold=snapshots(buf,rows)
        graphs={}
        for mode in CONTRACT['modes']:
            call(mode,x,xs,router,ex,buf,cfg);torch.cuda.synchronize();exact(buf,gold,rows)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,capture_error_mode='thread_local'):call(mode,x,xs,router,ex,buf,cfg)
            graphs[mode]=g
            for _ in range(3):g.replay()
            torch.cuda.synchronize();exact(buf,gold,rows)
        rec=dict(**current,rows=rows,input_stride=list(x.stride()),outputs_graph_exact=True,timings=[])
        report['cases'].append(rec)
        save()
        for regime in CONTRACT['regimes']:
            for rep in range(6):
                order=CONTRACT['modes'] if rep%2==0 else list(reversed(CONTRACT['modes']))
                for slot,mode in enumerate(order):
                    if regime=='evict_128MiB':flush.add_(1)
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();calls=CONTRACT['calls_per_interval'][regime]
                    for _ in range(calls):graphs[mode].replay()
                    b.record();b.synchronize();elapsed=a.elapsed_time(b)/calls
                    exact(buf,gold,rows)
                    rec['timings'].append(dict(regime=regime,mode=mode,repetition=rep,order=slot,calls=calls,ms=elapsed))
        save()
        if current['step']==0 and current['layer']==0:
            for mode in CONTRACT['modes']:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as prof:
                    with torch.profiler.record_function('e35.moe'):
                        graphs[mode].replay();torch.cuda.synchronize()
                trace=root/('artifacts/e35-'+str(len(report['traces']))+'-trace.json')
                assert not trace.exists();prof.export_chrome_trace(str(trace))
                report['traces'].append(dict(id=current['id'],mode=mode,file=trace.name,sha256=sha(trace)))
            exact(buf,gold,rows)
        print(json.dumps(dict(e35_cases=len(report['cases']),layer=current['layer'],rows=rows)),flush=True)
    eager=WideTarget(session.weights,capacity=2048)
    screen=FusedScreenedHead(session.weights.head)
    target=ScreenedGraphTarget(session.weights,screen,capacity=2048)
    diagnostic=ScreenedGraphTarget(session.weights,screen,capacity=2048)
    local_forward=clone_module(diagnostic.forward)
    local_moe=clone_module(local_forward.nvfp4_moe)
    def observed_moe(x,xs,router,ex,buf,cfg):
        nvfp4_moe.moe(x,xs,router,ex,buf,cfg)
        probe(x,xs,router,ex,buf,cfg)
        return buf
    local_moe.moe=observed_moe
    local_forward.nvfp4_moe=local_moe
    diagnostic.forward=local_forward
    def diagnostic_compute(state,rows):
        f,w=local_forward,session.weights;b=diagnostic.buffers[rows];segs=[(state,0,rows)]
        f._embed(w,b.ids[:rows],w.cfg.streams,b.h[:rows]);pending=None;features=[]
        for layer in w.layers:
            current['layer']=layer.index
            pending=f.layer_forward(layer,w,segs,b,rows,pending,context=2048)
            if layer.index in (5,15,25,35,45):
                shadow=b.h[:rows].clone();proxy=copy.copy(b);proxy.pss=torch.empty_like(b.pss)
                f._writeback(shadow,proxy,rows,w.cfg,pending)
                features.append(shadow.unflatten(-1,(4,2560)).mean(-2))
        return f.finish(w,w.mixer,b,rows,pending),torch.cat(features,-1)
    ext,identity=backend(root);report['w8_identity']=identity
    base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    draft=attach_mlp_w8(base,lambda x,p:ext.matmul(x,p.values,p.scales))
    model=NumpyLinearProgressModel(json.loads((root/'artifacts/e27-linear-model.json').read_text())['metadata'])
    costs=json.loads((root/'artifacts/e29-frozen-costs.json').read_text())['modes']['w8']
    tasks=json.loads((root/'e30_profile_tasks.json').read_text())['tasks'];warmed=False
    for task in tasks:
        ids,prompt_sha=expected_prompt(session,task);start,ctx,lg=prompt_state(eager,ids,chunk_size=8)
        if not warmed:target.warm(start);warmed=True
        state=target.bind(start);tokens=[int(lg[-1].argmax())]
        draft.begin((task['id'],time.monotonic_ns()))
        for step in range(2):
            assert tokens[-1] not in session.eos
            proposals,info=draft.propose(ctx,tokens[-1])
            rates=model.predict(feature_vector(info['signals'],len(ctx),len(tokens),task['output_cap']-len(tokens),False,'f1'))
            c=costs['long' if len(ctx)>=768 else 'short']
            width=choose_rows(rates,[v+c['paid_ms'] for v in c['verify_commit_ms']],min(8,task['output_cap']-len(tokens)))
            proposals=proposals[:width-1];input_ids=[tokens[-1]]+proposals
            other=state.clone();pred,ft,b,hidden,_=target.run_predictions(state,input_ids)
            logits=screen.original(hidden).clone();ft=ft.clone()
            current.update(id=task['id'],step=step,context_rows=state.pos,prompt_sha256=prompt_sha)
            db=diagnostic.buffers[width];local_forward.stage(session.weights,db,[(other,input_ids)])
            dl,df=diagnostic_compute(other,width);torch.cuda.synchronize()
            assert torch.equal(logits,dl) and torch.equal(ft,df)
            emitted,keep,_,stop=greedy_result(tokens[-1],proposals,pred.tolist(),session.eos,task['output_cap']-len(tokens))
            target.commit(state,b,width,keep);diagnostic.commit(other,db,width,keep)
            assert not eager.state_difference(state,other)
            report['trajectory'].append(dict(id=task['id'],step=step,rows=width,emitted=keep,logits_features_live_state_exact=True))
            save();ctx=torch.cat([ctx,ft[:keep]]);tokens.extend(emitted)
            if stop:break
        del start,state,other,ctx
    assert len(report['preflight'])==96 and len(report['cases'])==24 and len(report['trajectory'])==8
    result=summarize(report['cases']);gate=result['evict_128MiB']
    passed=gate['time_reduction']>=.05 and gate['root_bootstrap_gain_95'][0]>0 and min(v['speed_gain'] for v in gate['per_root'])>=-.02
    report.update(status='PASS_E35_MOE_GATE' if passed else 'REJECT_E35_MOE_COST',summary=result,finished_unix=time.time())
    check();save()
    return dict(status=report['status'],summary=result,target_object_id=id(session.weights),draft_object_id=id(session.draft))
