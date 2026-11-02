"""Isolated versus serialized adaptive requests with private full-history state."""
import gc
from types import SimpleNamespace
import importlib
import json
import time


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e54_run.py');common=load('scripts/e50_run.py')
    tasks=load('scripts/e49_study.py');fingerprints=load('scripts/e51_study.py');measure=load('scripts/e53_capacity.py')
    shared=importlib.import_module(tag+'.shared_graph');optimized=importlib.import_module(tag+'.kaifra_optimized')
    spec=importlib.import_module(tag+'.kaifra_speculative');native_mod=importlib.import_module(tag+'.native_scheduler')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    capacity=importlib.import_module(tag+'.long_capacity')
    path=session.root/'jobs/done'/args['study_job'];job=json.loads(path.read_text());gate=job['result']
    assert job['status']=='PASS' and gate['status']=='PASS_E53_CAPACITY_MEASURED'
    assert gate['target_object_id']==id(session.weights) and gate['draft_object_id']==id(session.draft)
    for name,digest in gate['source_sha256'].items():
        if name.startswith('spark_infer_q/') and name not in ('spark_infer_q/kaifra_speculative.py','spark_infer_q/kaifra_optimized.py'):
            assert helper.sha(source/name)==digest,name
    reference_path=session.root/'jobs/done'/args['reference_job']
    reference_version=args['reference_job'].split('-e53-',1)[1][:-5]
    reference_root=session.root.parent/('e53-long-20261026-'+reference_version)
    reference=json.loads((reference_root/'result.json').read_text())
    assert reference['target_object_id']==id(session.weights) and reference['draft_object_id']==id(session.draft)
    reference_runs={r['name']:r for r in reference['runs']}
    projection_bytes=50*1024**2
    before=measure.memory(torch)
    budget=capacity.LongCapacityBudget(gate['measured_shared_bytes'],gate['conservative_slot_bytes']+projection_bytes,
        gate['capacity_reserve_bytes'],gate['allocator_limit_bytes'],gate['host_reserve_bytes'])
    admission=budget.require(2,allocated_bytes=before['allocated_bytes'],reserved_bytes=before['reserved_bytes'],
        host_available_bytes=before['MemAvailable'])
    target=provider=executor=None
    report.update(status='RUNNING_E54_MULTI',capacity_job_sha256=helper.sha(path),admission=admission,
        runs=[],matches=[],lifecycle=[],probes=[],baseline_matches=[],per_request_projection_bytes=projection_bytes,
        reference_job_sha256=helper.sha(reference_path),reference_report_sha256=helper.sha(reference_root/'result.json'),scope='two private slots, serialized Graph scratch and per-request adaptive budgets; no GPU batch',
        timing='diagnostic guarded wall; state hashing at release included; not production throughput',
        actual_full256k_requests=0,max_populated_256k_live=0)
    def save():helper.save(source,report)
    def guard():
        m=measure.memory(torch)
        assert m['allocated_bytes']<=88*1024**3 and m['reserved_bytes']<=88*1024**3,m
        assert m['MemAvailable']>=8*1024**3,m
        return m
    def cache_signature(cache,n):
        return {key:fingerprints.tensor_sha(torch,t[:n] if key=='_context_rows' else t[:,:n])
            for key,t in cache.tensors.items()}

    def probe_rebuild(handle,state,anchor,remaining,normal,window):
        cache=state.projection_cache;n=len(state.context)
        signature=cache_signature(cache,n);counters=dict(cache.counters)
        feature_hash=fingerprints.tensor_sha(torch,state.context);position=state.target.pos
        probe_state=SimpleNamespace(context=state.context,context_start=state.context_start,
            output_tokens=state.output_tokens,projection_cache=None,next_width=1,cumulative=())
        diagnostic=spec.DFlashProvider(provider.worker,model=provider.model,budget=provider.budget,
            feature_group=provider.feature_group,history_window_rows=2040,max_request_context=262160,
            request_projection_cache=True)
        original=tuple(state.cumulative);width=state.next_width
        try:
            other=diagnostic.propose(('e54-rebuild-probe',window),probe_state,anchor,remaining)
            torch.cuda.synchronize();differences={}
            for key,a in cache.tensors.items():
                b=probe_state.projection_cache.tensors[key]
                a=a[:n] if key=='_context_rows' else a[:,:n]
                b=b[:n] if key=='_context_rows' else b[:,:n]
                differences[key]=dict(exact=torch.equal(a,b),maximum_abs=float((a.float()-b.float()).abs().max()))
            restored=(signature==cache_signature(cache,n) and counters==cache.counters and
                state.target.pos==position and fingerprint_same(feature_hash,state.context))
            record=dict(window=window,absolute_position=position,context_start=state.context_start,anchor=anchor,
                suffix_rows=n,suffix_sha256=feature_hash,incremental_rates=list(original),
                rebuild_rates=list(probe_state.cumulative),incremental_width=width,rebuild_width=probe_state.next_width,
                proposals_exact=normal[0]==other[0],rates_exact=original==probe_state.cumulative,
                maximum_rate_delta=max(abs(a-b) for a,b in zip(original,probe_state.cumulative)),
                projection_differences=differences,request_cache_preserved=restored,
                incrementally_appended_rows=cache.counters['appended_rows'],
                rebuilt_rows=probe_state.projection_cache.counters['appended_rows'])
            report['probes'].append(record);save();assert restored,'diagnostic changed the real request cache'
        finally:
            if probe_state.projection_cache is not None:probe_state.projection_cache.close()
            probe_state.projection_cache=None

    def fingerprint_same(digest,features):return digest==fingerprints.tensor_sha(torch,features)
    try:
        with torch.inference_mode():
            provider,_=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=262160,query_graphs=True,request_projection_cache=True)
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
                    max_live=max_long=max_cache=0;max_positions=[];minimum_host=2**63;steps=0
                    rows0=target.capabilities['graph_replays'];draft0=provider.worker.query_captures
                    propose_original=provider.propose;proposal_counts={}
                    def propose(handle,state,anchor,remaining):
                        result=propose_original(handle,state,anchor,remaining)
                        proposal_counts[handle]=proposal_counts.get(handle,0)+1
                        if name=='isolated-long-a' and proposal_counts[handle] in (30,31,32,36):
                            probe_rebuild(handle,state,anchor,remaining,result,proposal_counts[handle])
                        return result
                    provider.propose=propose
                    with spec.speculative_engine(native,target,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],
                            mode='adaptive',capacity=262144,max_requests=len(requests),token_budget=1024,tokenizer=session.tokenizer,
                            prefill_executor=executor,output_cache=True,serialized_postdraft=len(requests)>1) as engine:
                        backend=engine.bridge.backend;original=backend.release
                        def retain(handle,state):
                            states[handle.request_id]=dict(position=state.target.pos,
                                state=fingerprints.live_state(torch,state.target),
                                features=fingerprints.tensor_sha(torch,state.context) if state.context is not None else None,
                                context_start=state.context_start,output_tokens=state.output_tokens,
                                projection_bytes=0 if state.projection_cache is None else state.projection_cache.nbytes,
                                projection_counters={} if state.projection_cache is None else dict(state.projection_cache.counters))
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
                                max_cache=max(max_cache,sum(s.projection_cache.nbytes for s in live if s.projection_cache is not None))
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
                        finally:
                            backend.release=original
                            provider.propose=propose_original
                    record=dict(name=name,inputs=[dict(request_id=k,prompt_tokens=len(ids),input_sha256=tasks.token_sha(ids),output_cap=cap) for k,ids,cap in requests],
                        outputs=outputs,states=states,events=events,metrics=metrics,wall_seconds=wall,
                        max_live_states=max_live,max_populated_256k_live=max_long,max_live_projection_bytes=max_cache,coincident_long_positions=max_positions,
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
                    report['matches'].append(exact);save();return exact
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
                long_controls=[]
                for item in long:
                    control=run('isolated-'+item[0],[item])
                    long_controls.append(control)
                    prior=reference_runs['isolated-'+item[0]]
                    key=item[0]
                    fields=('position','state','features','context_start','output_tokens')
                    trace=lambda record:[{k:m[k] for k in ('verified_rows','emitted','accepted','observed','context_rows','rates')} for m in record['metrics'] if not m['prompt']]
                    report['baseline_matches'].append(dict(request_id=key,output=control['outputs']==prior['outputs'],
                        full_state_and_features=all(control['states'][key][k]==prior['states'][key][k] for k in fields),
                        budget_and_acceptance=trace(control)==trace(prior)))
                    save()
                together=run('two-256k',long)
                for item,control in zip(long,long_controls):compare(control,together,item[0])
                assert together['max_populated_256k_live']==2,'never had two independently populated256K states'
                assert all(len(together['outputs'][k])==512 for k,ids,cap in long)
                gate_passed=all(all(m[k] for k in ('output','full_state_and_features','budget_and_acceptance')) for m in report['matches']+report['baseline_matches'])
                assert len(report['probes'])==4 and all(p['request_cache_preserved'] for p in report['probes'])
                report['projection_rebuild_changes_observed']=any(not p['rates_exact'] for p in report['probes'])
                report.update(actual_full256k_requests=2,max_populated_256k_live=2,strict_gate_passed=gate_passed,
                    actual_long_requests_including_isolated_controls=4,status='PASS_E54_REQUEST_CACHE' if gate_passed else 'REJECT_E54_STRICT_TRACE')
                save()
    finally:
        if target is not None:target.close()
        if executor is not None:executor.close()
        if provider is not None:provider.worker.close()
        target=executor=provider=None;gc.collect();torch.cuda.synchronize()
    report['after_release']=measure.memory(torch)
    assert report['after_release']['allocated_bytes']==report['baseline_bytes']
    return report
