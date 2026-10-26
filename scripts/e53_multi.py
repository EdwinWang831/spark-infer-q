"""Isolated versus serialized adaptive requests with private full-history state."""
import gc
import hashlib
import importlib
import json
import time
from collections import Counter


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e53_run.py');common=load('scripts/e50_run.py')
    tasks=load('scripts/e49_study.py');fingerprints=load('scripts/e51_study.py');measure=load('scripts/e53_capacity.py')
    shared=importlib.import_module(tag+'.shared_graph');optimized=importlib.import_module(tag+'.kaifra_optimized')
    spec=importlib.import_module(tag+'.kaifra_speculative');native_mod=importlib.import_module(tag+'.native_scheduler')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    capacity=importlib.import_module(tag+'.long_capacity')
    path=session.root/'jobs/done'/args['study_job'];job=json.loads(path.read_text());gate=job['result']
    assert job['status']=='PASS' and gate['status']=='PASS_E53_CAPACITY_MEASURED'
    assert gate['target_object_id']==id(session.weights) and gate['draft_object_id']==id(session.draft)
    for name,digest in gate['source_sha256'].items():
        if name.startswith('spark_infer_q/'):assert helper.sha(source/name)==digest,name
    before=measure.memory(torch)
    budget=capacity.LongCapacityBudget(gate['measured_shared_bytes'],gate['conservative_slot_bytes'],
        gate['capacity_reserve_bytes'],gate['allocator_limit_bytes'],gate['host_reserve_bytes'])
    admission=budget.require(2,allocated_bytes=before['allocated_bytes'],reserved_bytes=before['reserved_bytes'],
        host_available_bytes=before['MemAvailable'])
    target=provider=executor=None
    report.update(status='RUNNING_E53_MULTI',capacity_job_sha256=helper.sha(path),admission=admission,
        runs=[],matches=[],lifecycle=[],scope='two private slots, serialized Graph scratch and per-request adaptive budgets; no GPU batch',
        timing='diagnostic guarded wall; state hashing at release included; not production throughput',
        actual_full256k_requests=0,max_populated_256k_live=0)
    def save():helper.save(source,report)
    def guard():
        m=measure.memory(torch)
        assert m['allocated_bytes']<=88*1024**3 and m['reserved_bytes']<=88*1024**3,m
        assert m['MemAvailable']>=8*1024**3,m
        return m
    try:
        with torch.inference_mode():
            provider,_=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=262160,query_graphs=True)
            provider.worker.warmup()
            target=shared.SharedGraphTarget.from_session(session,capacity=262160,kv_dtype='fp8',slots=2,index_ring_rows=2048)
            target.warmup()
            expert,digest=common.expert(session,tag);report['expert_binary_sha256']=digest
            executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,
                expert_backend=expert,kv_dtype='fp8',index_ring_rows=2048),chunk_rows=1024)
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                def run(name,requests,*,cancel=False):
                    report['active']=name;save()
                    outputs={key:[] for key,ids,cap in requests};states={};events=[];metrics=[];handles={}
                    max_live=max_long=0;max_positions=[];minimum_host=2**63;steps=0
                    rows0=target.capabilities['graph_replays'];draft0=provider.worker.query_captures
                    with spec.speculative_engine(native,target,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],
                            mode='adaptive',capacity=262144,max_requests=len(requests),token_budget=1024,tokenizer=session.tokenizer,
                            prefill_executor=executor,output_cache=True,serialized_postdraft=len(requests)>1) as engine:
                        backend=engine.bridge.backend;original=backend.release
                        def retain(handle,state):
                            states[handle.request_id]=dict(position=state.target.pos,
                                state=fingerprints.live_state(torch,state.target),
                                features=fingerprints.tensor_sha(torch,state.context) if state.context is not None else None,
                                context_start=state.context_start,output_tokens=state.output_tokens)
                            original(handle,state)
                        backend.release=retain
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                        try:
                            for key,ids,cap in requests:handles[key]=engine.submit_tokens(key,ids,max_tokens=cap)
                            for steps in range(1600):
                                if not engine.active_requests:break
                                if cancel and steps==1:engine.cancel(handles[requests[0][0]])
                                produced=engine.step();metrics.extend(engine.last_step_metrics)
                                for event in produced:
                                    key=event.handle.request_id;outputs[key].extend(event.token_ids)
                                    events.append(dict(request_id=key,emitted=len(event.token_ids),finish_reason=event.finish_reason,
                                        ttft_seconds=event.ttft_seconds,elapsed_seconds=event.elapsed_seconds))
                                live=list(backend.states.values());max_live=max(max_live,len(live))
                                positions=[s.target.pos for s in live]
                                populated=sum(p>=261120 for p in positions);max_long=max(max_long,populated)
                                if populated==2:max_positions=positions
                                m=guard();minimum_host=min(minimum_host,m['MemAvailable'])
                                if steps%32==0:
                                    report['progress']=dict(run=name,steps=steps,positions=positions,
                                        output_tokens={k:len(v) for k,v in outputs.items()},memory=m);save()
                            assert not engine.active_requests,'bounded drain exhausted'
                            torch.cuda.synchronize();wall=time.perf_counter()-start
                            assert len(states)==len(requests) and not backend.states and not target.owners
                        finally:backend.release=original
                    record=dict(name=name,inputs=[dict(request_id=k,prompt_tokens=len(ids),input_sha256=tasks.token_sha(ids),output_cap=cap) for k,ids,cap in requests],
                        outputs=outputs,states=states,events=events,metrics=metrics,wall_seconds=wall,
                        max_live_states=max_live,max_populated_256k_live=max_long,coincident_long_positions=max_positions,
                        graph_replays=target.capabilities['graph_replays']-rows0,capabilities=target.capabilities,
                        draft_captures_added=provider.worker.query_captures-draft0,
                        memory=measure.memory(torch),minimum_host_available_bytes=minimum_host,steps=steps)
                    report['runs'].append(record);save()
                    assert target.capabilities['graph_captures']==32 and not any(target.capabilities['graph_fallbacks'])
                    assert record['draft_captures_added']==0
                    return record
                def compare(control,combined,key):
                    def trace(record):
                        return [{k:m[k] for k in ('prompt','verified_rows','emitted','accepted','observed','context_rows','rates')}
                            for m in record['metrics'] if m['request_id']==key and not m['prompt']]
                    exact=dict(request_id=key,control=control['name'],combined=combined['name'],
                        output=control['outputs'][key]==combined['outputs'][key],
                        full_state_and_features=control['states'][key]==combined['states'][key],
                        budget_and_acceptance=trace(control)==trace(combined),budget_trace_scope='decode_only',
                        control_prompt_chunks=[m['verified_rows'] for m in control['metrics'] if m['request_id']==key and m['prompt']],
                        combined_prompt_chunks=[m['verified_rows'] for m in combined['metrics'] if m['request_id']==key and m['prompt']])
                    report['matches'].append(exact);save();assert all(exact[k] for k in ('output','full_state_and_features','budget_and_acceptance')),exact
                short=[]
                for key,text,n in (('english','Explain cache ownership and cancellation. ',3073),
                                   ('chinese','请解释缓存隔离和请求取消后的复用。',4097)):
                    ids=session.tokenizer.encode(text*2000,add_special_tokens=False)[:n];assert len(ids)==n
                    short.append((key,ids,64))
                controls=[run('isolated-'+item[0],[item]) for item in short]
                both=run('two-short',short)
                for item,control in zip(short,controls):compare(control,both,item[0])
                cancelled=run('cancel-short',short,cancel=True)
                assert any(e['request_id']=='english' and e['finish_reason']=='cancelled' for e in cancelled['events'])
                compare(controls[1],cancelled,'chinese')
                reused=run('reuse-short',short)
                for item,control in zip(short,controls):compare(control,reused,item[0])
                report['lifecycle'].append(dict(cancelled=True,reused=True,foreign_state_overlap=False))
                long_ids,identity=tasks.inputs(session)
                second_ids=list(long_ids);marker=session.tokenizer.encode('Independent request B owns a separate history. ',add_special_tokens=False)
                second_ids[1024:1024+len(marker)]=marker
                long=[('long-a',long_ids,512),('long-b',second_ids,512)]
                report['long_input']=identity;save()
                long_controls=[run('isolated-'+item[0],[item]) for item in long]
                together=run('two-256k',long)
                for item,control in zip(long,long_controls):compare(control,together,item[0])
                assert together['max_populated_256k_live']==2,'never had two independently populated256K states'
                assert all(len(together['outputs'][k])==512 for k,ids,cap in long)
                report.update(actual_full256k_requests=2,max_populated_256k_live=2,
                    actual_long_requests_including_isolated_controls=4,status='PASS_E53_TWO_LONG_ADAPTIVE')
                save()
    finally:
        if target is not None:target.close()
        if executor is not None:executor.close()
        if provider is not None:provider.worker.close()
        target=executor=provider=None;gc.collect();torch.cuda.synchronize()
    report['after_release']=measure.memory(torch)
    assert report['after_release']['allocated_bytes']==report['baseline_bytes']
    return report
