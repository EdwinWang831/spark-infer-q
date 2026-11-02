"""Tile16 pipeline: boundary correctness and matched complete-MoE cost screen."""
import gc
import importlib
import json
from pathlib import Path


def component(session,source,tag,args,load):
    torch,w=session.torch,session.weights
    sha=load('scripts/e43_build.py').sha
    buildpath=session.root/'jobs/done'/args['build_job'];job=json.loads(buildpath.read_text())
    assert job['status']=='PASS' and job['load_count']==1
    built=job['result'];assert built['status']=='PASS_E43_BUILD'
    assert built['target_object_id']==id(w) and built['draft_object_id']==id(session.draft)
    for name in ('scripts/e43_compute.cuh','scripts/e43_build.py','scripts/e43_bindings.cpp'):
        assert sha(source/name)==built['source_sha256'][name],name
    build=source.parent/('e43-prefill-20261024-'+args['build_version'])
    binary=build/'build/binary/spark_q_e43_pipeline_v1.so';assert sha(binary)==built['binary_sha256']
    mod=load('scripts/e43_candidate.py');candidate=mod.Candidate(mod.extension(binary))
    account=load('scripts/e43_accounting.py')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    from e3_wide_target import WideTarget
    from tensorfold.cuda import experts as grouped,moe as moe_mod
    from tensorfold.cuda.nvfp4 import experts as nvx,linear
    from tensorfold.families.qwen4_exp.cuda import nvfp4_moe
    from tensorfold.families.qwen4_exp.cuda.state import _MoECfg
    cfg=_MoECfg(w.cfg);original=nvfp4_moe.moe;gu=nvx.gate_up;down=nvx.down
    baseline_binary=Path(linear._ext().__file__)
    assert sha(baseline_binary)==session.metadata['extensions_sha256']['spark_q_moe_linear']
    report=dict(status='RUNNING_E43_COMPONENT',build_job_sha256=sha(buildpath),binary_sha256=built['binary_sha256'],
        baseline_binary_sha256=sha(baseline_binary),preflight=[],cases=[],diagnostic_prompts=[],
        candidate_min_rows=256,default_promoted=False,timing_scope='whole_moe_graph_cuda_event')
    path=source/'result.json'
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    def snapshot(buf,rows):return {n:getattr(buf,n)[:rows].clone() for n in ('act','y','pick','wts','logits')}
    def exact(buf,gold,rows):
        for n,v in gold.items():assert torch.equal(getattr(buf,n)[:rows],v),n
    executors=[]
    try:
        with torch.inference_mode():
            gen=torch.Generator(device='cuda').manual_seed(43261024);ex=w.layers[0].moe.experts.routed_experts
            for rows in (1,15,16,17,31,32,33,64,128,512,1024):
                for strided in (False,True):
                    for dtype in (torch.float32,torch.bfloat16):
                        report['active']=dict(preflight_rows=rows,strided=strided,dtype=str(dtype));save()
                        buf=moe_mod.MoEBuffers(rows+1,cfg,w.device,prefill=True)
                        buf.y=torch.full(buf.y.shape,13.,device=w.device,dtype=dtype);buf.act.fill_(13.)
                        x=torch.randn((rows,ex.dims+(8 if strided else 0)),device=w.device,dtype=torch.bfloat16,generator=gen)[:,:ex.dims]
                        picks=buf.pick[:rows];picks[:,:cfg.num_experts_per_tok]=torch.arange(cfg.num_experts_per_tok,device=w.device)
                        picks[:,cfg.num_experts_per_tok]=ex.count;grouped.route(picks,buf.plan,16)
                        gu(x,ex,buf.plan,buf.act.view(-1,ex.width),rows,skip=ex.count);gold_gu=buf.act.clone()
                        down(buf.act.view(-1,ex.width),ex,buf.plan,buf.y.view(-1,ex.dims),rows,skip=ex.count);gold_down=buf.y.clone()
                        def kernels():
                            candidate.gate_up(x,ex,buf.plan,buf.act.view(-1,ex.width),rows,skip=ex.count)
                            candidate.down(buf.act.view(-1,ex.width),ex,buf.plan,buf.y.view(-1,ex.dims),rows,skip=ex.count)
                        kernels();torch.cuda.synchronize();assert torch.equal(buf.act,gold_gu) and torch.equal(buf.y,gold_down)
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,capture_error_mode='thread_local'):kernels()
                        for _ in range(2):graph.replay()
                        torch.cuda.synchronize();assert torch.equal(buf.act,gold_gu) and torch.equal(buf.y,gold_down)
                        if rows==33 and strided and dtype==torch.float32:
                            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):kernels()
                            stream.synchronize();assert torch.equal(buf.act,gold_gu) and torch.equal(buf.y,gold_down)
                            report['separate_caller_stream_exact']=True
                        assert bool((buf.act[rows:]==13).all()) and bool((buf.y[rows:]==13).all())
                        report['preflight'].append(dict(rows=rows,strided=strided,dtype=str(dtype),exact=True,guards_exact=True,graph_exact=True))
                        del graph,buf,x,gold_gu,gold_down
            del kernels;gc.collect();save()
            flush=torch.zeros(128*1024**2//4,dtype=torch.int32,device=w.device);current={}
            def probe(layer,w,b,rows):
                if layer.index not in (0,23,47):return
                x,xs,router,ex,buf=b.mixed[:rows],b.xs_mixed[:rows],layer.moe.router,layer.moe.experts,b.moe
                gold=snapshot(buf,rows);graphs={}
                for mode,fn in (('serial',original),('candidate',candidate)):
                    fn(x,xs,router,ex,buf,cfg);torch.cuda.synchronize();exact(buf,gold,rows)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,capture_error_mode='thread_local'):fn(x,xs,router,ex,buf,cfg)
                    for _ in range(3):graph.replay()
                    torch.cuda.synchronize();exact(buf,gold,rows);graphs[mode]=graph
                rec=dict(name=current['name'],layer=layer.index,rows=rows,exact=True,timings=[]);report['cases'].append(rec);save()
                for regime in ('warm','evict_128MiB'):
                    for rep in range(6):
                        for mode in (('serial','candidate') if (rep+layer.index)%2==0 else ('candidate','serial')):
                            if regime=='evict_128MiB':flush.add_(1)
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            calls=3 if regime=='warm' else 1
                            start.record()
                            for _ in range(calls):graphs[mode].replay()
                            end.record();end.synchronize();elapsed=start.elapsed_time(end)/calls;exact(buf,gold,rows)
                            rec['timings'].append(dict(regime=regime,mode=mode,repetition=rep,calls=calls,ms=elapsed))
                original(x,xs,router,ex,buf,cfg);torch.cuda.synchronize();exact(buf,gold,rows);save()
            observer=mod.provider_class(ops.TensorFoldPrefillProvider,observer=probe)
            control=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(w,torch,capacity=2048),chunk_rows=1024)
            observed=prefill.PrefillExecutor(observer(w,torch,capacity=2048),chunk_rows=1024);executors.extend((control,observed))
            owner=WideTarget(w,capacity=2048)
            texts=dict(english64='Explain how a cache differs from a queue. ',code512='Write a Python function that keeps unique items in order. Explain its invariant. ',
                code1024='Write a Python function that keeps unique items in order. Explain its invariant. ',mixed1024='A cache stores reusable results. 缓存保存重复计算结果。 Compare a cache with a queue. ')
            report['inputs']=[]
            for name,n in (('english64',64),('code512',512),('code1024',1024),('mixed1024',1024)):
                ids=session.tokenizer.encode(texts[name]*400,add_special_tokens=False)[:n];assert len(ids)==n
                import hashlib
                case=dict(name=name,length=n,token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())
                report['inputs'].append(case);current['name']=name;report['active']=name+':real_moe';save()
                a=owner.state();value=control.advance(a,ids);b=owner.state();got=observed.advance(b,ids)
                assert torch.equal(value.logits,got.logits) and torch.equal(value.features,got.features) and not owner.state_difference(a,b)
                report['diagnostic_prompts'].append(dict(name=name,exact=True,full_logits_exact=True,features_exact=True,state_exact=True))
                del a,b,value,got;gc.collect();save()
            report['summary']=account.summarize(report['cases']);report['candidate_calls']=candidate.calls;report['control_fallback_calls']=candidate.fallbacks
            report['status']='PASS_E43_COMPONENT';report.pop('active',None);save()
    except BaseException as exc:
        report.update(status='FAIL_E43_COMPONENT',error_type=type(exc).__name__,error=str(exc)[:2000]);save();raise
    finally:
        for executor in executors:executor.close()
        # Detach the observer/candidate closures before the stage package is removed.
        for executor in executors:
            executor.provider.ops=None
        executors.clear()
        gc.collect();torch.cuda.synchronize()
    return report
