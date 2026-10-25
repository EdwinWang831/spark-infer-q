"""Exercise the final public provider hook and expert backend on resident weights."""
import gc
import importlib
import json


def integration(session,source,tag,args,load):
    torch=session.torch;helpers=load('scripts/e41_run.py');sha=load('scripts/e44_build.py').sha
    jobpath=session.root/'jobs/done'/args['study_job'];job=json.loads(jobpath.read_text())
    assert job['status']=='PASS' and job['result']['status']=='PASS_E44_STUDY' and job['load_count']==1
    measured=job['result'];assert measured['target_object_id']==id(session.weights) and measured['draft_object_id']==id(session.draft)
    binary=source.parent/('e44-prefill-20261024-'+args['build_version'])/'build/binary/spark_q_e44_pipeline_v1.so'
    assert sha(binary)==measured['binary_sha256']
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    tile=importlib.import_module(tag+'.prefill_experts');graph=importlib.import_module(tag+'.kaifra_graph')
    backend=tile.PrefillExpertTile.from_file(binary,expected_sha256=measured['binary_sha256'])
    executors={mode:prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=2048,
        expert_backend=backend if mode=='candidate' else None),chunk_rows=1024) for mode in ('serial','candidate')}
    report=dict(status='RUNNING_E44_INTEGRATION',study_job_sha256=sha(jobpath),binary_sha256=measured['binary_sha256'],
        backend_name=backend.name,numeric=[],requests=[],lifecycle=[],http=[],timing_claim=False)
    def save():helpers.save(source,report)
    target=None
    try:
        with torch.inference_mode():
            from e3_wide_target import WideTarget
            owner=WideTarget(session.weights,capacity=2048);target=graph.GraphTarget(WideTarget(session.weights,capacity=2048));target.warmup()
            _,requests=helpers.inputs(session);code=next(r for r in requests if r['name']=='code513')
            ids=(code['ids']*2)[:1024]
            cases=[dict(name='code1024',ids=ids),next(r for r in requests if r['name']=='mixed1536')]
            for case in cases:
                report['active']='numeric:'+case['name'];save();a=owner.state();b=target.state()
                try:
                    x=executors['serial'].advance(a,case['ids']);y=executors['candidate'].advance(b,case['ids'])
                    assert torch.equal(x.logits,y.logits) and torch.equal(x.features,y.features) and not owner.state_difference(a,b)
                    row=dict(name=case['name'],prompt_tokens=len(case['ids']),full_logits_exact=True,features_exact=True,state_exact=True,windows=[])
                    report['numeric'].append(row)
                    for width in range(1,9):
                        piece=case['ids'][:width];keep=width if width%2 else width//2;replays=target.replays
                        lg,ft,buf=owner.run(a,piece);owner.commit(a,buf,width,keep)
                        g,f,buffer=target.run(b,piece);target.commit(b,buffer,width,keep)
                        assert torch.equal(lg,g) and torch.equal(ft,f) and not owner.state_difference(a,b) and target.replays==replays+1
                        row['windows'].append(dict(width=width,keep=keep,exact=True))
                    del x,y,lg,ft,buf,g,f,buffer
                finally:target.release_state(b)
                del a,b;gc.collect();save()
            report['numeric_graph']=target.capabilities;target.close();target=None
            optimized=importlib.import_module(tag+'.kaifra_optimized');spec=importlib.import_module(tag+'.kaifra_speculative');native_mod=importlib.import_module(tag+'.native_scheduler')
            target=optimized.optimized_target(session);provider,metadata=optimized.optimized_provider(session);target.warmup()
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                def make(mode):
                    return spec.speculative_engine(native,target,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],mode='adaptive',
                        capacity=1792,max_requests=1,token_budget=1024,tokenizer=session.tokenizer,prefill_executor=executors['candidate' if mode=='owned' else mode])
                reference=None
                for mode in executors:
                    rec,features,state,tokens=helpers.generate(session,target,make(mode),ids,-1,mode)
                    if reference is None:reference=(tokens,features,state.clone(),rec['trace'],rec['positions'])
                    gold,ft,st,trace,positions=reference
                    assert tokens==gold and torch.equal(features,ft) and not target.state_difference(st,state) and rec['trace']==trace and rec['positions']==positions
                    report['requests'].append(dict(mode=mode,prompt_tokens=1024,output_tokens=len(tokens),exact=True))
                del reference,features,state,ft,st
                helpers.lifecycle(session,source,report,target,make,cases[0])
                chinese=next(r for r in requests if r['name']=='zh')
                _,features,state,gold=helpers.generate(session,target,make('candidate'),chinese['ids'],-1,'candidate')
                del features,state
                helpers.http_checks(session,source,report,target,make,chinese,gold,tag)
            report['candidate_calls']=backend.calls;assert backend.calls>0
            report['status']='PASS_E44_INTEGRATION';report.pop('active',None);save()
    except BaseException as exc:
        report.update(status='FAIL_E44_INTEGRATION',error_type=type(exc).__name__,error=str(exc)[:2000]);save();raise
    finally:
        for executor in executors.values():executor.close()
        if target is not None:
            if target.leased:target.release_state(target.workspace)
            target.close()
        del target,executors
        gc.collect();torch.cuda.synchronize()
    return report
