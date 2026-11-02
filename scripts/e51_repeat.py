"""Paired real-prefix repeats with identical KV/pooled pointers and eager scratch."""
import copy
import gc
import importlib
import json
import time
from contextlib import nullcontext
from functools import partial


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e51_run.py');common=load('scripts/e50_run.py')
    tasks=load('scripts/e49_study.py');old=load('scripts/e46_study.py');study=load('scripts/e51_study.py')
    optimized=importlib.import_module(tag+'.kaifra_optimized');bounded=importlib.import_module(tag+'.bounded_index')
    spec=importlib.import_module(tag+'.kaifra_speculative');native_mod=importlib.import_module(tag+'.native_scheduler')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    windows=importlib.import_module(tag+'.feature_window');bridge=importlib.import_module(tag+'.deepinfer_bridge')
    path=session.root/'jobs/done'/args['study_job'];job=json.loads(path.read_text());prior=job['result']
    assert job['status']=='PASS' and prior['status']=='PASS_E51_STUDY' and prior['kv_dtype']=='fp8'
    assert prior['target_object_id']==id(session.weights)
    for name,digest in prior['source_sha256'].items():
        if name.startswith('spark_infer_q/'):assert helper.sha(source/name)==digest,name
    ids,identity=tasks.inputs(session);gold=prior['runs'][0]
    report.update(status='RUNNING_E51_REPEAT',input=identity,study_job_sha256=helper.sha(path),checks=[],timings=[],profiles=[],
        scope='seeded real256K FP8 loop; identical KV/pool/scratch/head addresses; excludes native Scheduler/event overhead')
    def save():helper.save(source,report)
    targets={};executor=provider=None;seed=ringseed=features=None
    try:
        with torch.inference_mode():
            full=optimized.optimized_target(session,capacity=262160,kv_dtype='fp8');targets['full']=full;full.warmup()
            eager=copy.copy(full.eager);eager.forward=bounded.build_forward(2048);eager.index_ring_rows=2048
            eager.State=partial(bounded.make_state,ring_rows=2048)
            ring=optimized.ScreenedGraphTarget(eager,full.screen);targets['ring']=ring
            ring.workspace=eager.state()
            # Replace these before capture, and before either real prefix is
            # built: warmup writes row0. Every timed arm now uses identical KV.
            ring.workspace.kc=full.workspace.kc;ring.workspace.pooled=full.workspace.pooled
            gc.collect();torch.cuda.synchronize();ring.warmup()
            assert ring.eager.buffers is full.eager.buffers and ring.screen is full.screen
            assert all(a.k.data_ptr()==b.k.data_ptr() and a.v.data_ptr()==b.v.data_ptr() for a,b in zip(full.workspace.kc,ring.workspace.kc))
            report['same_kv_pool_scratch_head_addresses']=True
            for rows in range(1,9):
                x=torch.zeros((rows,full.screen.k),dtype=torch.bfloat16,device=full.screen.weight.device)
                for count in range(1,129):full.screen.subset(x,torch.arange(count,device=x.device))
            del x
            provider,_=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=262160,query_graphs=True)
            provider.worker.warmup()
            expert,digest=common.expert(session,tag);report['expert_binary_sha256']=digest
            executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,expert_backend=expert,kv_dtype='fp8'),chunk_rows=1024)
            report['active']='real256K_prefill';save()
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                with spec.speculative_engine(native,full,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],mode='adaptive',
                    capacity=262144,max_requests=1,token_budget=1024,prefill_executor=executor,output_cache=True) as engine:
                    handle=engine.submit_tokens('repeat-seed',ids,max_tokens=512);first=None
                    for step in range(260):
                        out=engine.step();mem=old.guard(torch)
                        if step%32==0:report['progress']=dict(prefill_step=step,**mem);save()
                        if out:
                            assert len(out)==1 and len(out[0].token_ids)==1
                            first=out[0].token_ids[0];st=engine.bridge.backend.states[handle]
                            seed=st.target.snapshot();features=st.context.clone();assert st.target.pos==len(ids)
                            break
                    assert first==gold['output_ids'][0] and len(ids)%4==0
                    engine.cancel(handle);list(engine.drain(max_steps=8));assert not engine.active_requests
            ringseed=dict(seed,bounded_index_tail=[a[:0].clone() for a in ring.workspace.ikc])
            prefix=study.live_state(torch,full.workspace)
            # Prefix fingerprint uses only its committed semantics; p is block
            # aligned, so no raw tail must be copied to the ring before replay.
            ring.workspace.restore(ringseed);assert prefix==study.live_state(torch,ring.workspace)
            def generate(mode,rep,round_limit=None,profile=False):
                target=targets[mode];backend=spec.SpeculativeBackend(target,provider,eos_ids=session.eos,depth=7,prior=session.calibration['conditional_prior'])
                handle=('e51-repeat',mode,rep);state=backend.create(handle);state.target.restore(seed if mode=='full' else ringseed)
                state.feature_window=windows.FeatureWindow(2040);state.feature_window.end=len(ids)-len(features)
                state.context=state.feature_window.append(features);state.context_start=state.feature_window.start;state.output_tokens=1
                output=[first];steps=[];predict,commit=target.run_predictions,target.commit
                def phase(name):return torch.profiler.record_function('e50.'+name) if profile else nullcontext()
                def run_prediction(*a,**kw):
                    with phase('verify'):
                        value=predict(*a,**kw)
                        if profile:torch.cuda.synchronize()
                        return value
                def run_commit(*a,**kw):
                    with phase('commit'):
                        value=commit(*a,**kw)
                        if profile:torch.cuda.synchronize()
                        return value
                target.run_predictions=run_prediction;target.commit=run_commit
                before=target.capabilities;hashes=dict(target.screen.kernel_sha256);draft=provider.worker.graph_capabilities
                try:
                    torch.cuda.synchronize();start=time.perf_counter()
                    while len(output)<512 and output[-1] not in session.eos and (round_limit is None or len(steps)<round_limit):
                        with phase('draft'):
                            proposals,_=provider.propose(handle,state,output[-1],512-len(output));torch.cuda.synchronize()
                        width=state.next_width;pos=state.target.pos
                        row=bridge.WorkRow(handle,pos,tuple([output[-1]]+list(proposals[:width-1])),False,False)
                        result=backend.advance(row,state,512);output.extend(result.outputs)
                        steps.append(dict(position=pos,width=width,keep=result.keep,accepted=result.accepted,observed=result.observed,
                            cumulative=list(state.cumulative),outputs=list(result.outputs)))
                    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                    caps=target.capabilities
                    assert caps['graph_captures']==before['graph_captures']==16 and not caps['graph_fallbacks'] and not caps['head_fallbacks']
                    assert hashes==target.screen.kernel_sha256 and draft['draft_graph_captures']==provider.worker.query_captures==10
                    value=dict(mode=mode,repetition=rep,profiled=profile,decode_ms=elapsed,output_tokens=len(output),output_ids=output,
                        output_sha256=tasks.token_sha(output),steps=steps,live_state=study.live_state(torch,state.target),
                        feature_sha256=study.tensor_sha(torch,state.context),graph_replays=caps['graph_replays']-before['graph_replays'],
                        timed_head_hashes_unchanged=True,query_captures=10)
                    return value
                finally:
                    target.run_predictions=predict;target.commit=commit;backend.release(handle,state)
            for mode in ('full','ring'):
                report['active']='fidelity:'+mode;save();r=generate(mode,-1)
                assert r['output_ids']==gold['output_ids'] and r['live_state']==gold['live_state'] and r['feature_sha256']==gold['feature_sha256']
                report['checks'].append(r);save()
            a,b=report['checks'];assert a['steps']==b['steps']
            for rep in range(4):
                for mode in (('full','ring') if rep%2==0 else ('ring','full')):
                    report['active']=f'timed:{rep}:{mode}';save();r=generate(mode,rep)
                    assert r['output_ids']==a['output_ids'] and r['steps']==a['steps'] and r['live_state']==a['live_state']
                    report['timings'].append(r);save()
            for mode in ('full','ring'):
                report['active']='profile:'+mode;save()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False,profile_memory=False,with_stack=False) as profiler:r=generate(mode,-2,16,True)
                p=source/(mode+'-verify-trace.json');profiler.export_chrome_trace(str(p))
                r['trace']=dict(file=p.name,sha256=helper.sha(p),bytes=p.stat().st_size);report['profiles'].append(r);save()
            assert report['profiles'][0]['steps']==report['profiles'][1]['steps'] and report['profiles'][0]['live_state']==report['profiles'][1]['live_state']
            total={mode:sum(x['decode_ms'] for x in report['timings'] if x['mode']==mode) for mode in ('full','ring')}
            gains=[]
            for rep in range(4):
                pair={x['mode']:x['decode_ms'] for x in report['timings'] if x['repetition']==rep};gains.append(pair['full']/pair['ring']-1)
            report.update(status='PASS_E51_REPEAT',seeded_tps={mode:4*511*1000/ms for mode,ms in total.items()},
                aggregate_change=total['full']/total['ring']-1,per_pair_change=gains,
                all_outputs_state_trace_exact=True,shared_kv_pool_prefix_immutable=all(
                    study.tensor_sha(torch,full.workspace.kc[i].k[:len(ids)])==prefix[str(i)+'k'] and
                    study.tensor_sha(torch,full.workspace.kc[i].v[:len(ids)])==prefix[str(i)+'v'] and
                    study.tensor_sha(torch,full.workspace.pooled[i][:len(ids)//4])==prefix[str(i)+'pool'] for i in range(12)))
            assert report['shared_kv_pool_prefix_immutable'];save();return report
    finally:
        seed=ringseed=features=None
        if executor is not None:executor.close()
        for target in targets.values():
            if target.leased:target.release_state(target.workspace)
            target.close()
        if provider is not None:provider.worker.close()
        targets.clear();executor=provider=None;gc.collect();torch.cuda.synchronize()
