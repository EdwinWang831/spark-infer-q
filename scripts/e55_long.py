"""Single256K INT4 research diagnostic, BF16 teacher and same-format greedy proof."""
import gc
import importlib
import json
import shutil
import time
import numpy as np


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e55_run.py');common=load('scripts/e50_run.py')
    tasks=load('scripts/e49_study.py');fingerprints=load('scripts/e51_study.py')
    measure=load('scripts/e53_capacity.py');metrics=load('scripts/e47_state.py')
    shared=importlib.import_module(tag+'.shared_graph');optimized=importlib.import_module(tag+'.kaifra_optimized')
    spec=importlib.import_module(tag+'.kaifra_speculative');native_mod=importlib.import_module(tag+'.native_scheduler')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    gate_path=session.root/'jobs/done'/args['study_job'];gate=json.loads(gate_path.read_text())
    assert gate['status']=='PASS' and gate['result']['status']=='PASS_E55_QUALITY_SCREEN_COMPLETED'
    for name,digest in gate['result']['source_sha256'].items():
        if name.startswith('spark_infer_q/'):assert helper.sha(source/name)==digest,name
    ids,identity=tasks.inputs(session);expert,digest=common.expert(session,tag)
    report.update(status='RUNNING_E55_LONG',input=identity,output_cap=512,runs=[],
        quality_job_sha256=helper.sha(gate_path),short_int4_quality_gate_passed=gate['result']['int4_quality_gate_passed'],
        expert_binary_sha256=digest,quality_promoted=False,scope='one request per format, same-format greedy reference; not eight-way',
        timing='guarded diagnostic wall with prefix checks and teacher pause; not serving or decode benchmark')
    def save():helper.save(source,report)
    def guard():
        m=measure.memory(torch)
        assert m['allocated_bytes']<=88*1024**3 and m['reserved_bytes']<=88*1024**3,m
        assert m['MemAvailable']>=8*1024**3,m
        return m
    reference=None;formats=('bf16','int4')
    if args.get('reuse_bf16',False):
        prior_job=session.root/'jobs/done'/args['reference_job']
        old=json.loads(prior_job.read_text())
        assert old['status']=='FAIL' and old['error']=="AttributeError: 'LongTarget' object has no attribute 'leased'"
        version=args['reference_job'].split('-e55-',1)[1][:-5]
        prior=session.root.parent/('e55-memory-20261027-'+version)
        old_report=json.loads((prior/'result.json').read_text())
        assert old_report['status']=='FAIL_E55_LONG' and len(old_report['runs'])==1
        assert old_report['target_object_id']==id(session.weights) and old_report['baseline_bytes']==report['baseline_bytes']
        for name,digest in old_report['source_sha256'].items():
            if name.startswith('spark_infer_q/'):assert helper.sha(source/name)==digest,name
        row=old_report['runs'][0]
        assert row['kv_dtype']=='bf16' and len(row['output_ids'])==512 and row['same_format_output_exact'] and row['same_format_state_exact']
        old_teacher=prior/'teacher-bf16.npz';assert helper.sha(old_teacher)==row['teacher_artifact_sha256']
        shutil.copyfile(old_teacher,source/'teacher-bf16.npz')
        with np.load(old_teacher,allow_pickle=False) as data:
            reference=dict(forced=data['forced'].tolist(),teacher=[torch.from_numpy(r.copy()) for r in data['logits']])
        report['runs'].append(row)
        report.update(reused_bf16_job_sha256=helper.sha(prior_job),reused_bf16_report_sha256=helper.sha(prior/'result.json'),
            reused_bf16_source_sha256=old_report['source_sha256'])
        formats=('int4',);save()
    target=executor=provider=seed=held=st=result=lg=buf=engine=backend=native=release_original=retain=None
    try:
        for fmt in formats:
            report['active']=fmt+':prepare';save();guard()
            provider,_=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=262160,
                query_graphs=True,request_projection_cache=True)
            provider.worker.warmup()
            target=shared.SharedGraphTarget.from_session(session,capacity=262160,kv_dtype=fmt,slots=1,index_ring_rows=2048)
            target.warmup()
            executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,
                expert_backend=expert,kv_dtype=fmt,index_ring_rows=2048),chunk_rows=1024)
            last={};done={};events=[];trace=[];output=[];teacher=[];forced=[];minimum_host=2**63
            original_run=executor.provider.run
            def capture_prefill(b,state,piece,*,capture):
                value=original_run(b,state,piece,capture=capture)
                if state.pos==len(ids):last['logits']=value[0][-1].float().cpu()
                return value
            executor.provider.run=capture_prefill
            try:
                with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                    with spec.speculative_engine(native,target,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],
                            mode='adaptive',capacity=262144,max_requests=1,token_budget=1024,tokenizer=session.tokenizer,
                            prefill_executor=executor,output_cache=True) as engine:
                        backend=engine.bridge.backend;release_original=backend.release
                        def retain(handle,state):
                            done['state']=fingerprints.live_state(torch,state.target)
                            done['features']=fingerprints.tensor_sha(torch,state.context)
                            done['projection_bytes']=state.projection_cache.nbytes
                            release_original(handle,state)
                        backend.release=retain
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();teacher_ms=0.
                        try:
                            engine.submit_tokens('e55-'+fmt,ids,max_tokens=512)
                            for step in range(1200):
                                if not engine.active_requests:break
                                produced=engine.step();trace.extend(engine.last_step_metrics)
                                for e in produced:
                                    output.extend(e.token_ids);events.append(dict(emitted=len(e.token_ids),elapsed=e.elapsed_seconds,
                                        ttft=e.ttft_seconds,finish=e.finish_reason))
                                if output and seed is None:
                                    assert len(output)==1 and last
                                    held=next(iter(backend.states.values()));st=held.target;seed=st.snapshot()
                                    before=fingerprints.live_state(torch,st);feature_sha=fingerprints.tensor_sha(torch,held.context)
                                    t=time.perf_counter();teacher=[last['logits']];forced=[]
                                    count=8 if reference is None else len(reference['forced'])
                                    for i in range(count):
                                        if i:
                                            lg,result=target.step(st,[forced[-1]],capture=False)
                                            teacher.append(lg[-1].float().cpu())
                                        forced.append(int(teacher[-1].argmax()) if reference is None else reference['forced'][i])
                                        if reference is None and forced[-1] in session.eos:break
                                    st.restore(seed);teacher_ms=(time.perf_counter()-t)*1000
                                    assert before==fingerprints.live_state(torch,st),'teacher probe changed committed prefix'
                                    assert feature_sha==fingerprints.tensor_sha(torch,held.context)
                                    assert st.pos==len(ids)
                                    last['prefix']=before
                                    if reference is None:reference=dict(forced=forced[:],teacher=teacher[:])
                                m=guard();minimum_host=min(minimum_host,m['MemAvailable'])
                                if step%32==0:
                                    report['progress']=dict(kv_dtype=fmt,step=step,output_tokens=len(output),
                                        positions=[s.target.pos for s in backend.states.values()],memory=m);save()
                            assert not engine.active_requests and len(output)==512 and done
                            torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
                            record=dict(kv_dtype=fmt,prompt_tokens=len(ids),output_ids=output,output_sha256=tasks.token_sha(output),
                                events=events,metrics=trace,request_ms=wall,teacher_pause_ms=teacher_ms,
                                teacher_forced_ids=forced,teacher_metrics=metrics.logit_metrics(torch,reference['teacher'],teacher),
                                prefix_probe_preserved=True,native_state=done['state'],feature_sha256=done['features'],
                                projection_bytes=done['projection_bytes'],capabilities=target.capabilities,
                                memory=measure.memory(torch),minimum_host_available_bytes=minimum_host,
                                retrieval_keys=[k for k in identity['retrieval_keys'] if k in session.tokenizer.decode(output,skip_special_tokens=True)])
                            path=source/('teacher-'+fmt+'.npz')
                            np.savez_compressed(path,logits=torch.stack(teacher).numpy(),forced=np.asarray(forced,dtype=np.int64))
                            record['teacher_artifact_sha256']=helper.sha(path)
                            report['runs'].append(record);save()
                        finally:backend.release=release_original
                # The existing workspace still owns this exact populated prefix.
                # Restore only recurrent/PLE/raw-tail seed; committed KV was never
                # overwritten by generation or teacher probes. Check before reuse.
                held=None;st=target.state();st.restore(seed)
                assert fingerprints.live_state(torch,st)==last['prefix'],'greedy seed prefix changed'
                tokens=[int(last['logits'].argmax())]
                for step in range(511):
                    if tokens[-1] in session.eos:break
                    lg,result=target.step(st,[tokens[-1]],capture=False);tokens.append(int(lg[-1].argmax()))
                    m=guard();minimum_host=min(minimum_host,m['MemAvailable'])
                    if step%64==0:report['progress']=dict(kv_dtype=fmt,phase='greedy_reference',output_tokens=len(tokens));save()
                state=fingerprints.live_state(torch,st)
                record.update(greedy_output_ids=tokens,greedy_output_sha256=tasks.token_sha(tokens),greedy_state=state,
                    same_format_output_exact=tokens==output,same_format_state_exact=state==done['state'],
                    minimum_host_available_bytes=minimum_host,memory=measure.memory(torch))
                save();assert record['same_format_output_exact'] and record['same_format_state_exact'],'INT4/BF16 speculation changed target result'
                target.release_state(st);st=None
            finally:
                executor.provider.run=original_run
                for owner in tuple(target.owners.values()):
                    target.release_state(owner.workspace)
                owner=None
                target.close();executor.close();provider.worker.close()
                original_run=None;target=executor=provider=seed=held=st=result=lg=buf=engine=backend=native=release_original=retain=None
                gc.collect();torch.cuda.synchronize()
            assert torch.cuda.memory_allocated()==report['baseline_bytes']
        report.update(status='PASS_E55_LONG_DIAGNOSTIC',int4_bf16_greedy_exact=report['runs'][0]['output_ids']==report['runs'][1]['output_ids'])
        save()
    finally:
        target=executor=provider=seed=held=st=result=lg=buf=engine=backend=native=release_original=retain=None;gc.collect();torch.cuda.synchronize()
    report['after_release_allocated_bytes']=torch.cuda.memory_allocated()
    assert report['after_release_allocated_bytes']==report['baseline_bytes']
    return report
