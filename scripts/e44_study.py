"""Full-model and request validation after the fixed Mia tile cost gate."""
import gc
import importlib
import json
import time


def study(session,source,tag,args,load):
    torch=session.torch;helpers=load('scripts/e41_run.py');sha=load('scripts/e44_build.py').sha
    component_path=session.root/'jobs/done'/args['study_job'];job=json.loads(component_path.read_text())
    assert job['status']=='PASS' and job['result']['status']=='PASS_E44_COMPONENT' and job['load_count']==1
    component=job['result'];assert component['summary']['cost_gate_passed']
    assert component['target_object_id']==id(session.weights) and component['draft_object_id']==id(session.draft)
    for name in ('scripts/e44_candidate.py','scripts/e44_tile.cuh','scripts/e44_build.py','scripts/e44_bindings.cpp','scripts/e44_accounting.py'):
        assert sha(source/name)==component['source_sha256'][name],name
    binary=source.parent/('e44-prefill-20261024-'+args['build_version'])/'build/binary/spark_q_e44_pipeline_v1.so'
    assert sha(binary)==component['binary_sha256']
    mod=load('scripts/e44_candidate.py');candidate=mod.Candidate(mod.extension(binary))
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    graph=importlib.import_module(tag+'.kaifra_graph')
    from e3_wide_target import WideTarget
    executors=dict(serial=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=2048),chunk_rows=1024),
        candidate=prefill.PrefillExecutor(mod.provider_class(ops.TensorFoldPrefillProvider,candidate=candidate)(session.weights,torch,capacity=2048),chunk_rows=1024))
    report=dict(status='RUNNING_E44_STUDY',component_job_sha256=sha(component_path),binary_sha256=component['binary_sha256'],
        numeric=[],timings=[],requests=[],lifecycle=[],http=[],physical_capacity=2048,logical_capacity=1792,
        token_budget=1024,chunk_rows=1024,output_cap=64,default_promoted=False)
    def save():helpers.save(source,report)
    target=None
    try:
        with torch.inference_mode():
            numeric,requests=helpers.inputs(session)
            # Keep the exact numeric E41 cases; add controlled code1024 to requests.
            code=next(c for c in numeric if c['name']=='code512')
            code1024=list(code['ids'])*2
            requests=[r for r in requests if r['name'] in ('en','zh','mixed1536')]
            requests.insert(2,dict(name='code1024',length=1024,ids=code1024,token_sha256=helpers.token_sha(code1024)))
            report['inputs']=[{k:v for k,v in c.items() if k not in ('ids','messages')} for c in numeric+requests]
            owner=WideTarget(session.weights,capacity=2048)
            target=graph.GraphTarget(WideTarget(session.weights,capacity=2048));target.warmup();seen=set()
            for case in numeric:
                report['active']='numeric:'+case['name'];save();a=owner.state();b=target.state()
                try:
                    x=executors['serial'].advance(a,case['ids']);y=executors['candidate'].advance(b,case['ids'])
                    row=dict(name=case['name'],length=case['length'],chunks=list(y.chunks),
                        logits_exact=torch.equal(x.logits,y.logits),features_exact=torch.equal(x.features,y.features),
                        state_differences=owner.state_difference(a,b),windows=[])
                    report['numeric'].append(row);save();assert row['logits_exact'] and row['features_exact'] and not row['state_differences'],row
                    for width in range(1,9):
                        ids=case['ids'][:width];keep=width if width%2 else width//2;parity=b.cur[0];before=target.replays
                        lg,ft,buf=owner.run(a,ids);owner.commit(a,buf,width,keep)
                        g,f,buffer=target.run(b,ids);target.commit(b,buffer,width,keep)
                        exact=torch.equal(lg,g) and torch.equal(ft,f) and not owner.state_difference(a,b)
                        assert exact and target.replays==before+1;seen.add((width,parity))
                        row['windows'].append(dict(width=width,keep=keep,parity=parity,exact=True))
                    del x,y,lg,ft,buf,g,f,buffer
                finally:target.release_state(b)
                del a,b;gc.collect();save()
            assert seen=={(r,p) for r in range(1,9) for p in (0,1)}
            report['numeric_graph']=dict(target.capabilities,buckets_checked=[list(x) for x in sorted(seen)])
            target.close();target=None
            # Paired complete prompt timing; profile/initial workspace setup excluded.
            prompts=[next(c for c in numeric if c['name']=='english64'),requests[2],requests[3]]
            for case in prompts:
                report['active']='prompt:'+case['name'];save();reference=None
                for mode in executors:
                    state=owner.state();value=executors[mode].advance(state,case['ids'])
                    if reference is None:reference=(value.logits,value.features,state.clone())
                    else:assert torch.equal(value.logits,reference[0]) and torch.equal(value.features,reference[1]) and not owner.state_difference(state,reference[2])
                    del state,value
                for rep in range(4):
                    for mode in (('serial','candidate') if rep%2==0 else ('candidate','serial')):
                        torch.cuda.synchronize();start=time.perf_counter()
                        state=owner.state();value=executors[mode].advance(state,case['ids'])
                        elapsed=(time.perf_counter()-start)*1000
                        assert torch.equal(value.logits,reference[0]) and torch.equal(value.features,reference[1]) and not owner.state_difference(state,reference[2])
                        report['timings'].append(dict(name=case['name'],mode=mode,repetition=rep,wall_ms=elapsed,prompt_tokens=case['length'],exact=True))
                        del state,value
                    save()
                del reference;gc.collect()
            optimized=importlib.import_module(tag+'.kaifra_optimized');spec=importlib.import_module(tag+'.kaifra_speculative')
            native_mod=importlib.import_module(tag+'.native_scheduler')
            target=optimized.optimized_target(session);provider,metadata=optimized.optimized_provider(session)
            report['profile']=metadata;target.warmup()
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                def make(mode):
                    return spec.speculative_engine(native,target,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],
                        mode='adaptive',capacity=1792,max_requests=1,token_budget=1024,tokenizer=session.tokenizer,
                        prefill_executor=executors['candidate' if mode=='owned' else mode])
                natural={}
                for case in requests:
                    report['active']='requests:'+case['name'];save();reference=None
                    for rep in range(-1,4):
                        for mode in (('serial','candidate') if rep%2==0 else ('candidate','serial')):
                            rec,features,state,tokens=helpers.generate(session,target,make(mode),case['ids'],rep,mode)
                            rec.update(name=case['name'],prompt_tokens=case['length'],token_sha256=case['token_sha256'])
                            if reference is None:reference=(tokens,features,state.clone(),rec['trace'],rec['positions'])
                            gold,ft,st,trace,positions=reference
                            assert tokens==gold and torch.equal(features,ft) and not target.state_difference(st,state) and rec['trace']==trace and rec['positions']==positions
                            rec.update(outputs_exact=True,features_exact=True,state_exact=True,trace_exact=True)
                            if rep>=0:assert rec['captures_before']==rec['captures_after']==16 and rec['kernels_before']==rec['kernels_after'] and not rec['graph_fallbacks'] and rec['head_fallbacks']==0
                            report['requests'].append(rec);save()
                    natural[case['name']]=reference[0];del reference,features,state,ft,st
                helpers.lifecycle(session,source,report,target,make,requests[1])
                helpers.http_checks(session,source,report,target,make,requests[1],natural['zh'],tag)
            report['optimized_graph']=target.capabilities;report['candidate_calls']=candidate.calls
            report['status']='PASS_E44_STUDY';report.pop('active',None);save()
    except BaseException as exc:
        report.update(status='FAIL_E44_STUDY',error_type=type(exc).__name__,error=str(exc)[:2000]);save();raise
    finally:
        for executor in executors.values():
            executor.close()
            if hasattr(executor.provider,'ops'):executor.provider.ops=None
        if target is not None:
            if target.leased:target.release_state(target.workspace)
            target.close()
        del target,executors
        gc.collect();torch.cuda.synchronize()
    return report
