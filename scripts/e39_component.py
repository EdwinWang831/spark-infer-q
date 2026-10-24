"""Boundary/stream/Graph checks and real prefill complete-MoE cost gate."""
import gc
import json
from types import ModuleType


def component(session,source,manifest,helpers):
    digest,load,clone=helpers
    torch,w=session.torch,session.weights
    root=session.root
    build_job=root/'jobs/done/320-e39-build-v1.json'
    built=json.loads(build_job.read_text());assert built['status']=='PASS'
    built=built['result'];assert built['status']=='PASS_E39_BUILD'
    for name in ('scripts/e39_cta.cuh','scripts/e39_build.py','scripts/e39_bindings.cpp'):
        assert manifest[name]==built['source_sha256'][name],name
    binary=source.parent/'e39-prefill-20261024-build-v1/build/binary/spark_q_e39_prefill_v1.so'
    assert digest(binary)==built['binary_sha256'] and str(binary)==built['binary']
    geometry=root/'jobs/done/301-e38-geometry-v2.json'
    assert digest(geometry)=='f1f5af156ad3cb4e71c54d5aecf7f1f49085de4d7784486ba618afb523d6a9e5'
    geo=json.loads(geometry.read_text());assert geo['status']=='PASS' and geo['result']['status']=='PASS_E38_GEOMETRY'
    for name in ('scripts/e37_model.py','scripts/e36_prefill_candidate.py','scripts/e36_diagnose.py'):
        assert manifest[name]==geo['result']['source_sha256'][name],name
    for data in (built,geo['result']):
        assert data['target_object_id']==id(w) and data['draft_object_id']==id(session.draft)
    modules=[]
    tag='_e39_component_'+digest(source/'manifest.json')[:10]
    def owned(name,suffix):
        key=tag+'_'+suffix;modules.append(key);return load(source/name,key)
    adapter=owned('scripts/e36_prefill_candidate.py','adapter')
    helper=owned('scripts/e36_diagnose.py','helper');model=owned('scripts/e37_model.py','model')
    candidate_module=owned('scripts/e39_candidate.py','candidate');account=owned('scripts/e39_accounting.py','account')
    from tensorfold.families.qwen4_exp.cuda import nvfp4_moe,forward as upstream
    from tensorfold.families.qwen4_exp.cuda.state import _MoECfg
    from tensorfold.cuda import moe as moe_mod,experts as grouped
    original_moe=nvfp4_moe.moe;original_gu=nvfp4_moe.nvx.gate_up;original_down=nvfp4_moe.nvx.down
    cfg=_MoECfg(w.cfg);key=tag+'_private_moe';modules.append(key)
    candidate=candidate_module.Candidate(nvfp4_moe,clone,key,candidate_module.extension(binary))
    gc.collect();torch.cuda.synchronize();baseline=torch.cuda.memory_allocated()
    result=dict(status='RUNNING_E39_COMPONENT',source_sha256=manifest,
        build_job_sha256=digest(build_job),geometry_job_sha256=digest(geometry),binary_sha256=built['binary_sha256'],
        target_object_id=id(w),draft_object_id=id(session.draft),model_reloaded=False,tf32_disabled=True,
        baseline_bytes=baseline,preflight=[],cases=[],diagnostic_prompts=[],candidate_min_rows=256,
        timing_scope='whole_moe_graph_cuda_event',default_promoted=False)
    path=source/'result.json'
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    def snapshot(buf,rows):return {n:getattr(buf,n)[:rows].clone() for n in ('act','y','pick','wts','logits')}
    def exact(buf,gold,rows):
        for n,v in gold.items():assert torch.equal(getattr(buf,n)[:rows],v),n
    try:
        gen=torch.Generator(device='cuda').manual_seed(38261024)
        ex=w.layers[0].moe.experts.routed_experts
        for rows in (1,15,16,17,31,32,33,64,128,512):
            for strided in (False,True):
                for dtype in (torch.float32,torch.bfloat16):
                    result['active']=dict(preflight_rows=rows,strided=strided,dtype=str(dtype));save()
                    buf=moe_mod.MoEBuffers(rows+1,cfg,w.device,prefill=True)
                    buf.y=torch.full(buf.y.shape,13.,device=w.device,dtype=dtype);buf.act.fill_(13.)
                    x=torch.randn((rows,ex.dims+(8 if strided else 0)),device=w.device,dtype=torch.bfloat16,generator=gen)[:,:ex.dims]
                    picks=buf.pick[:rows];picks[:,:cfg.num_experts_per_tok]=torch.arange(cfg.num_experts_per_tok,device=w.device)
                    picks[:,cfg.num_experts_per_tok]=ex.count
                    grouped.route(picks,buf.plan,16)
                    original_gu(x,ex,buf.plan,buf.act.view(-1,ex.width),rows,skip=ex.count)
                    gu=buf.act.clone()
                    grouped.route(picks,buf.plan,32)
                    candidate.module.nvx.gate_up(x,ex,buf.plan,buf.act.view(-1,ex.width),rows,skip=ex.count)
                    assert torch.equal(buf.act,gu),'boundary gate/up'
                    grouped.route(picks,buf.plan,16)
                    original_down(buf.act.view(-1,ex.width),ex,buf.plan,buf.y.view(-1,ex.dims),rows,skip=ex.count)
                    down=buf.y.clone()
                    grouped.route(picks,buf.plan,32)
                    def kernels():
                        candidate.module.nvx.gate_up(x,ex,buf.plan,buf.act.view(-1,ex.width),rows,skip=ex.count)
                        candidate.module.nvx.down(buf.act.view(-1,ex.width),ex,buf.plan,buf.y.view(-1,ex.dims),rows,skip=ex.count)
                    kernels();torch.cuda.synchronize()
                    assert torch.equal(buf.act,gu) and torch.equal(buf.y,down),'boundary down'
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,capture_error_mode='thread_local'):kernels()
                    for _ in range(2):graph.replay()
                    torch.cuda.synchronize();assert torch.equal(buf.act,gu) and torch.equal(buf.y,down)
                    if rows==33 and strided and dtype==torch.float32:
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):kernels()
                        stream.synchronize();assert torch.equal(buf.act,gu) and torch.equal(buf.y,down)
                        result['separate_caller_stream_exact']=True
                    assert bool((buf.act[rows:]==13).all()) and bool((buf.y[rows:]==13).all())
                    result['preflight'].append(dict(rows=rows,strided=strided,dtype=str(dtype),exact=True,guards_exact=True,graph_exact=True))
                    del graph,buf,x,gu,down
        gc.collect();save()
        flush=torch.zeros(128*1024**2//4,dtype=torch.int32,device=w.device)
        current={}
        def probe(x,xs,router,ex,buf,cfg):
            rows=len(x);gold=snapshot(buf,rows);graphs={}
            for mode,fn in (('serial',original_moe),('candidate',candidate)):
                fn(x,xs,router,ex,buf,cfg);torch.cuda.synchronize();exact(buf,gold,rows)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,capture_error_mode='thread_local'):fn(x,xs,router,ex,buf,cfg)
                for _ in range(3):graph.replay()
                torch.cuda.synchronize();exact(buf,gold,rows);graphs[mode]=graph
            rec=dict(name=current['name'],layer=current['layer'],rows=rows,exact=True,timings=[])
            result['cases'].append(rec);save()
            for regime in ('warm','evict_128MiB'):
                for repetition in range(6):
                    order=('serial','candidate') if (repetition+current['layer'])%2==0 else ('candidate','serial')
                    for mode in order:
                        if regime=='evict_128MiB':flush.add_(1)
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        calls=3 if regime=='warm' else 1
                        a.record()
                        for _ in range(calls):graphs[mode].replay()
                        b.record();b.synchronize();ms=a.elapsed_time(b)/calls;exact(buf,gold,rows)
                        rec['timings'].append(dict(regime=regime,mode=mode,repetition=repetition,calls=calls,ms=ms))
            original_moe(x,xs,router,ex,buf,cfg);torch.cuda.synchronize();exact(buf,gold,rows);save()
        with adapter.using_forward(session,tag+'_forward') as (f,_):
            observed=ModuleType(tag+'_observed')
            def observed_moe(x,xs,router,ex,buf,cfg):
                original_moe(x,xs,router,ex,buf,cfg)
                if current['layer'] in (0,23,47):probe(x,xs,router,ex,buf,cfg)
                return buf
            observed.moe=observed_moe;f.nvfp4_moe=observed
            block=f.moe_block
            def observed_block(layer,*a,**kw):
                current['layer']=layer.index;return block(layer,*a,**kw)
            f.moe_block=observed_block;harness=model.PromptHarness(session,f)
            inputs=geo['result']['inputs']
            for c in inputs:
                current.update(name=c['name']);result['active']=c['name']+':real_moe';save()
                value=harness.prompt(c['ids']);out=helper.tensor_hash(value[0],value[1])
                assert out==next(v['output_sha256'] for v in geo['result']['cases'] if v['name']==c['name'])
                result['diagnostic_prompts'].append(dict(name=c['name'],output_sha256=out,exact=True));del value
            result['summary']=account.summarize(result['cases'])
        result.update(status='PASS_E39_COMPONENT',upstream_functions_unchanged=(
            nvfp4_moe.moe is original_moe and nvfp4_moe.nvx.gate_up is original_gu and nvfp4_moe.nvx.down is original_down))
        assert result['upstream_functions_unchanged'];result.pop('active',None);save();return result
    finally:
        if 'f' in locals():
            f.nvfp4_moe=nvfp4_moe
            if 'block' in locals():f.moe_block=block
        for name in modules:__import__('sys').modules.pop(name,None)
