"""Check compact shared activation staging and screen real MXFP8 projections."""
import copy
import gc
import importlib
import json
import random
import statistics
import sys
import time
import types
from e28_probe import sha,backend

CONTRACT=dict(roots=4,rounds=2,layers=[0,23,47],capacity=2048,repetitions=6,
    calls_per_interval=dict(warm_reuse=10,evict_128MiB=1),
    change='MXFP8 BM16: store only real activation rows; unused ldmatrix rows alias row0; unchanged MMA/split-K/cluster/weights',
    baseline='frozen E28 W8 draft + E17 NumPy linear + E29 W8 costs + full target Graph',
    regimes=['warm_reuse','evict_128MiB'],timing='single-projection CUDA Graph replay; checks/capture/eviction outside events',
    gate='all bitwise/guard/diagnostic checks; >=10% evicted projection time reduction; positive root-bootstrap lower bound; no root >2% slower',
    scope='four reused calibration roots, early real routes, three layers; microbenchmark not decode TPS')


def clone_module(source):
    """Rebind Python functions into private globals; never mutate the upstream module."""
    module=types.ModuleType(source.__name__+'_e32_private');module.__dict__.update(source.__dict__)
    for name,value in source.__dict__.items():
        if isinstance(value,types.FunctionType) and value.__globals__ is source.__dict__:
            fn=types.FunctionType(value.__code__,module.__dict__,value.__name__,value.__defaults__,value.__closure__)
            fn.__kwdefaults__=value.__kwdefaults__;fn.__annotations__=value.__annotations__;module.__dict__[name]=fn
    return module


def summarize(cases):
    roots=sorted({c['id'] for c in cases});result={}
    assert len(roots)==4
    for regime in CONTRACT['regimes']:
        totals=[]
        for r in roots:
            totals.append({m:sum(statistics.median(t['ms'] for t in c['timings'] if t['mode']==m and t['regime']==regime) for c in cases if c['id']==r) for m in ('original','compact')})
        original=sum(t['original'] for t in totals);new=sum(t['compact'] for t in totals);rng=random.Random(32261023);draws=[]
        for _ in range(4000):
            picks=rng.choices(range(4),k=4);draws.append(sum(totals[i]['original'] for i in picks)/sum(totals[i]['compact'] for i in picks)-1)
        draws.sort();result[regime]=dict(sum_median_ms=dict(original=original,compact=new),speed_gain=original/new-1,
            time_reduction=1-new/original,root_bootstrap_gain_95=[draws[99],draws[3899]],
            per_root=[dict(id=r,**t,speed_gain=t['original']/t['compact']-1) for r,t in zip(roots,totals)])
    return result


def run(session,arguments):
    from e24_run import expected_prompt
    from e3_wide_target import WideTarget
    from e29_graph import FeatureGraphTarget
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_backbone import attach_mlp_w8
    from spark_infer_q.confidence_features import feature_vector
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.confidence_head import choose_rows
    from spark_infer_q.speculative import greedy_result
    from tensorfold.cuda.nvfp4 import linear
    from tensorfold.cuda.kernels import qmm
    torch,root=session.torch,session.root;assert not arguments
    seal_path=root/'artifacts/e32-seal.json';seal=json.loads(seal_path.read_text());assert seal['contract']==CONTRACT
    def check():
        for key in ('source_sha256','input_sha256'):
            for n,h in seal[key].items():assert sha(root/n)==h,n
        for n,h in seal['external_source_sha256'].items():assert sha(root.parent/'e2-load/reference/tensorfold'/n)==h,n
    check();build=json.loads((root/'artifacts/e32-build.json').read_text());assert sha(build['binary'])==build['binary_sha256']
    sys.path.insert(0,str(root/'build/e32/binary'));ext=importlib.import_module('spark_q_e32_compact');original=linear._ext()
    path=root/'artifacts/e32-probe.json';assert not path.exists()
    report=dict(status='RUNNING',contract=CONTRACT,started_unix=time.time(),seal_sha256=sha(seal_path),build=build,
        baseline_binary_sha256=sha(original.__file__),target_object_id=id(session.weights),draft_object_id=id(session.draft),preflight=[],cases=[],trajectory=[])
    def save():
        p=path.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(path)
    save()
    def call(mode,x,weight,out,sk=None,n=None):
        sk=qmm.split_k(weight.n,weight.k) if sk is None else sk;n=weight.n if n is None else n
        if mode=='original':original.qmmf(x,weight.w8,weight.bs,1.,out,None,2,n,sk,weight.npad,16,False)
        else:ext.qmmf(x,weight.w8,weight.bs,1.,out,n,sk,weight.npad,mode=='compact')
    layer=session.weights.layers[0]
    linears=[('in',layer.gdn.proj),('out',layer.gdn.out),('shared_gu',layer.moe.experts.shared.gu),('shared_down',layer.moe.experts.shared.down)]
    assert all(isinstance(w,linear.Mx8Linear) for _,w in linears)
    generator=torch.Generator(device='cuda').manual_seed(32261023)
    for label,weight in linears:
        for rows in range(1,9):
            for strided in (False,True):
                x=torch.randn((rows,weight.k+8 if strided else weight.k),device='cuda',dtype=torch.bfloat16,generator=generator)[:,:weight.k]
                for sk in (1,2,4,8):
                    if weight.k//64%sk:continue
                    n=weight.n-(3 if strided else 0)
                    storage=torch.full((rows*n+32,),13.,device='cuda',dtype=torch.bfloat16);out=storage[16:-16].view(rows,n)
                    call('original',x,weight,out,sk,n);ref=out.clone()
                    for mode in ('control','compact'):
                        out.fill_(13);call(mode,x,weight,out,sk,n);torch.cuda.synchronize()
                        assert torch.equal(out,ref),(label,rows,strided,sk,mode)
                        assert bool((storage[:16]==13).all()) and bool((storage[-16:]==13).all())
                    report['preflight'].append(dict(matrix=label,n=n,k=weight.k,rows=rows,strided=strided,sk=sk,control_compact_original_exact=True,guards_exact=True))
        save()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        weight=linears[1][1];x=torch.randn((3,weight.k),device='cuda',dtype=torch.bfloat16,generator=generator)
        out=torch.empty((3,weight.n),device='cuda',dtype=torch.bfloat16);call('original',x,weight,out);ref=out.clone();call('compact',x,weight,out)
    stream.synchronize();assert torch.equal(ref,out);report['separate_stream_exact']=True;save()
    del storage,out,ref,x
    flush=torch.zeros(128*1024**2//4,dtype=torch.int32,device='cuda');current={}
    def probe(weight,x,got,kind):
        if current['layer'] not in CONTRACT['layers'] or not isinstance(weight,linear.Mx8Linear):return
        assert got.dtype==torch.bfloat16 and x.shape[0]<=8
        rows=x.shape[0];storage=torch.full((rows*weight.n+32,),13.,device='cuda',dtype=torch.bfloat16);out=storage[16:-16].view(rows,weight.n)
        for mode in ('original','control','compact'):
            call(mode,x,weight,out);torch.cuda.synchronize();assert torch.equal(out,got),(current,kind,mode)
            assert bool((storage[:16]==13).all()) and bool((storage[-16:]==13).all())
        record=dict(**current,kind=kind,rows=rows,n=weight.n,k=weight.k,npad=weight.npad,input_stride=list(x.stride()),sk=qmm.split_k(weight.n,weight.k),outputs_exact=True,guards_exact=True,timings=[])
        report['cases'].append(record);save();graphs={}
        for mode in ('original','compact'):
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,capture_error_mode='thread_local'):call(mode,x,weight,out)
            graphs[mode]=graph;graph.replay();torch.cuda.synchronize();assert torch.equal(out,got)
        for _ in range(3):graphs['original'].replay();graphs['compact'].replay()
        for regime in CONTRACT['regimes']:
            for rep in range(6):
                for slot,mode in enumerate(('original','compact') if rep%2==0 else ('compact','original')):
                    if regime=='evict_128MiB':flush.add_(1)
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record();calls=10 if regime=='warm_reuse' else 1
                    for _ in range(calls):graphs[mode].replay()
                    end.record();end.synchronize();assert torch.equal(out,got)
                    assert bool((storage[:16]==13).all()) and bool((storage[-16:]==13).all())
                    record['timings'].append(dict(regime=regime,repetition=rep,order=slot,mode=mode,ms=begin.elapsed_time(end)/calls,calls=calls))
        save();print(json.dumps(dict(e32_case=len(report['cases']),layer=current['layer'],kind=kind,rows=rows)),flush=True)
    eager=WideTarget(session.weights,capacity=2048);target=FeatureGraphTarget(session.weights,capacity=2048);diagnostic=FeatureGraphTarget(session.weights,capacity=2048)
    original_forward=diagnostic.forward;local_forward=clone_module(original_forward);local_moe=clone_module(original_forward.nvfp4_moe)
    original_mm=local_forward._mm;original_shared=local_moe._shared
    def mm(x,weight,xs,out,buf,**kwargs):
        got=original_mm(x,weight,xs,out,buf,**kwargs);probe(weight,x,got,'block_projection');return got
    def shared(weight,x,prompt,f32):
        got=original_shared(weight,x,prompt,f32);probe(weight,x,got,'shared_projection');return got
    local_forward._mm=mm;local_moe._shared=shared;local_forward.nvfp4_moe=local_moe;diagnostic.forward=local_forward
    def diagnostic_compute(state,rows):
        f,w=local_forward,session.weights;buf=diagnostic.buffers[rows];segs=[(state,0,rows)];f._embed(w,buf.ids[:rows],w.cfg.streams,buf.h[:rows]);pending=None;features=[]
        for layer in w.layers:
            current['layer']=layer.index;pending=f.layer_forward(layer,w,segs,buf,rows,pending,context=2048)
            if layer.index in (5,15,25,35,45):
                shadow=buf.h[:rows].clone();proxy=copy.copy(buf);proxy.pss=torch.empty_like(buf.pss);f._writeback(shadow,proxy,rows,w.cfg,pending);features.append(shadow.unflatten(-1,(4,2560)).mean(-2))
        return f.finish(w,w.mixer,buf,rows,pending),torch.cat(features,-1)
    w8,_=backend(root);base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048;draft=attach_mlp_w8(base,lambda x,p:w8.matmul(x,p.values,p.scales))
    model=NumpyLinearProgressModel(json.loads((root/'artifacts/e27-linear-model.json').read_text())['metadata']);costs=json.loads((root/'artifacts/e29-frozen-costs.json').read_text())['modes']['w8']
    tasks=json.loads((root/'e30_profile_tasks.json').read_text())['tasks'];warmed=False
    try:
        for ti,task in enumerate(tasks):
            report['active']=dict(index=ti,phase='prefill');save();ids,prompt_sha=expected_prompt(session,task);start,ctx,lg=prompt_state(eager,ids,chunk_size=8)
            if not warmed:target.warm(start);warmed=True
            state=target.bind(start);tokens=[int(lg[-1].argmax())];draft.begin((task['id'],time.monotonic_ns()))
            for step in range(2):
                assert tokens[-1] not in session.eos and len(tokens)<task['output_cap']
                proposals,info=draft.propose(ctx,tokens[-1]);rates=model.predict(feature_vector(info['signals'],len(ctx),len(tokens),task['output_cap']-len(tokens),False,'f1'))
                c=costs['long' if len(ctx)>=768 else 'short'];width=choose_rows(rates,[v+c['paid_ms'] for v in c['verify_commit_ms']],min(8,task['output_cap']-len(tokens)));proposals=proposals[:width-1];input_ids=[tokens[-1]]+proposals
                other=state.clone();lg,ft,buf=target.run(state,input_ids);current.update(id=task['id'],family=task['family'],language=task['language'],step=step,context_rows=state.pos,prompt_tokens=len(ids),prompt_sha256=prompt_sha)
                db=diagnostic.buffers[width];local_forward.stage(session.weights,db,[(other,input_ids)]);dl,df=diagnostic_compute(other,width);torch.cuda.synchronize()
                assert torch.equal(lg,dl) and torch.equal(ft,df)
                emitted,keep,_,stop=greedy_result(tokens[-1],proposals,lg.argmax(-1).tolist(),session.eos,task['output_cap']-len(tokens));target.commit(state,buf,width,keep);diagnostic.commit(other,db,width,keep)
                assert not eager.state_difference(state,other);report['trajectory'].append(dict(id=task['id'],step=step,width=width,emitted=keep,logits_features_live_state_exact=True));save();ctx=torch.cat([ctx,ft[:keep]]);tokens.extend(emitted)
                if stop:break
            del start,state,other,ctx
        check();assert len(report['cases'])==96 and len(report['trajectory'])==8
        report['summary']=summarize(report['cases']);gate=report['summary']['evict_128MiB'];passed=gate['time_reduction']>=.1 and gate['root_bootstrap_gain_95'][0]>0 and min(v['speed_gain'] for v in gate['per_root'])>=-.02
        report.update(status='PASS_E32_COMPACT_GATE' if passed else 'REJECT_E32_COMPACT_PERFORMANCE',finished_unix=time.time());report.pop('active',None);save()
        return dict(status=report['status'],target_object_id=id(session.weights),draft_object_id=id(session.draft))
    finally:
        diagnostic.forward=original_forward;local_forward.__dict__.clear();local_moe.__dict__.clear();gc.collect()
