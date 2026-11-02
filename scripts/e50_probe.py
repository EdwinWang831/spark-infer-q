"""One real prefix, exact seeded decode pairs and an annotated verify trace."""
import gc
import importlib
import json
import time
from contextlib import nullcontext
from pathlib import Path


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e50_run.py');old=load('scripts/e46_study.py')
    tasks=load('scripts/e49_study.py');fp=load('scripts/e48_study.py');checks=load('scripts/e47_state.py')
    shared=importlib.import_module(tag+'.shared_graph');optimized=importlib.import_module(tag+'.kaifra_optimized')
    spec=importlib.import_module(tag+'.kaifra_speculative');native_mod=importlib.import_module(tag+'.native_scheduler')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    windows=importlib.import_module(tag+'.feature_window');bridge=importlib.import_module(tag+'.deepinfer_bridge')
    cap_path=session.root/'jobs/done/520-e49-capacity-v1.json';capjob=json.loads(cap_path.read_text());c=capjob['result']
    assert capjob['status']=='PASS' and capjob['worker_pid']==__import__('os').getpid() and capjob['load_count']==1
    assert c['target_object_id']==id(session.weights) and c['baseline_bytes']==report['baseline_bytes']
    mem=old.memory(torch);total=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemTotal:')))*1024
    forecast=c['after_common_prefill']['allocated_bytes']+8*c['one_state_allocation_delta_bytes']
    cuda_free,cuda_total=torch.cuda.mem_get_info()
    report.update(status='RUNNING_E50_PROBE',eight_capacity=dict(forecast_allocated_bytes=forecast,current_limit_bytes=c['allocation_limit_bytes'],
        host_total_bytes=total,host_before=mem,cuda_driver_free_bytes=cuda_free,cuda_driver_total_bytes=cuda_total,
        source_capacity_job_sha256=helper.sha(cap_path),excluded=['eight decode Graph pools','screened head auxiliaries','eight draft caches/features'],
        actual_eight_execution=False,scope='measured_layout_forecast_not_runtime_capacity_validation'),checks=[],timings=[],profiles=[])
    def save():helper.save(source,report)
    save();ids,identity=tasks.inputs(session);report['input']=identity
    goldpath=session.root/'jobs/done/528-e49-study-v2.json';goldjob=json.loads(goldpath.read_text());gold=goldjob['result']['formats'][0]['runs'][0]
    assert goldjob['status']=='PASS' and gold['input_sha256']==identity['input_sha256']
    report['e49_reference_job_sha256']=helper.sha(goldpath)
    target=executor=None;providers={};seed=features=None
    try:
        with torch.inference_mode():
            target=shared.SharedGraphTarget.from_session(session,kv_dtype='bf16',slots=1);target.warmup()
            for rows in range(1,9):
                x=torch.zeros((rows,target.screen.k),dtype=torch.bfloat16,device=target.screen.weight.device)
                for count in range(1,129):target.screen.subset(x,torch.arange(count,device=x.device))
            del x
            for mode in ('baseline','query_graph'):
                providers[mode],_=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=262160,query_graphs=mode=='query_graph')
            providers['query_graph'].worker.warmup()
            expert,digest=helper.expert(session,tag);report['expert_binary_sha256']=digest
            executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,expert_backend=expert),chunk_rows=1024)
            report['active']='real256K_prefill';save()
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                with spec.speculative_engine(native,target,providers['baseline'],eos_ids=session.eos,prior=session.calibration['conditional_prior'],mode='adaptive',
                    capacity=262144,max_requests=1,token_budget=1024,prefill_executor=executor) as engine:
                    handle=engine.submit_tokens('e50-seed',ids,max_tokens=512);first=None
                    for step in range(260):
                        events=engine.step();m=old.guard(torch)
                        if step%32==0:report['progress']=dict(prefill_step=step,**m);save()
                        if events:
                            assert len(events)==1 and len(events[0].token_ids)==1
                            first=events[0].token_ids[0];st=engine.bridge.backend.states[handle]
                            assert st.target.pos==len(ids) and len(st.context)==2040
                            seed=st.target.snapshot();features=st.context.clone();break
                    assert first==gold['output_ids'][0]
                    prefix_sha=checks.digest(torch,st.target,len(ids));engine.cancel(handle)
                    list(engine.drain(max_steps=8));assert not engine.active_requests
            report['prefix_sha256']=prefix_sha;report['prefix_rows']=len(ids);save()
            # Exact positioned draft hidden/proposal/signal test before timings.
            a,b=(providers[m].worker for m in ('baseline','query_graph'))
            for w in (a,b):w.begin(('positioned-fidelity',));w.set_window(len(ids)-len(features))
            ha,_=a.forward_hidden(features,first);hb,_=b.forward_hidden(features,first)
            assert torch.equal(ha,hb),'positioned draft hidden mismatch'
            pa,ia=a.propose(features,first);pb,ib=b.propose(features,first)
            assert pa==pb and ia==ib,'positioned proposals/signals mismatch'
            report['positioned_hidden_proposals_signals_exact']=True;save()
            del ha,hb
            def generate(mode,rep,*,round_limit=None,profile=False):
                backend=spec.SpeculativeBackend(target,providers[mode],eos_ids=session.eos,depth=7,prior=session.calibration['conditional_prior'])
                handle=('e50',mode,rep,profile);state=backend.create(handle);state.target.restore(seed)
                state.feature_window=windows.FeatureWindow(2040);state.feature_window.end=len(ids)-len(features)
                state.context=state.feature_window.append(features);state.context_start=state.feature_window.start;state.output_tokens=1
                tokens=[first];steps=[];provider=providers[mode]
                old_run,old_commit=target.run_predictions,target.commit
                def phase(name):return torch.profiler.record_function('e50.'+name) if profile else nullcontext()
                def predict(*a,**kw):
                    with phase('verify'):
                        value=old_run(*a,**kw)
                        if profile:torch.cuda.synchronize()
                        return value
                def commit(*a,**kw):
                    with phase('commit'):
                        value=old_commit(*a,**kw)
                        if profile:torch.cuda.synchronize()
                        return value
                target.run_predictions=predict;target.commit=commit
                hashes=dict(target.screen.kernel_sha256);caps=target.capabilities
                gcaps=provider.worker.query_captures if mode=='query_graph' else 0
                try:
                    torch.cuda.synchronize();start=time.perf_counter()
                    while len(tokens)<512 and tokens[-1] not in session.eos and (round_limit is None or len(steps)<round_limit):
                        with phase('draft'):
                            proposals,_=provider.propose(handle,state,tokens[-1],512-len(tokens));torch.cuda.synchronize()
                        width=state.next_width;position=state.target.pos
                        row=bridge.WorkRow(handle,position,tuple([tokens[-1]]+list(proposals[:width-1])),False,False)
                        result=backend.advance(row,state,512);tokens.extend(result.outputs)
                        steps.append(dict(position=position,width=width,keep=result.keep,accepted=result.accepted,observed=result.observed,
                            cumulative=list(state.cumulative),outputs=list(result.outputs)))
                    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                    assert target.screen.kernel_sha256==hashes,'timed target head JIT'
                    assert not any(target.capabilities['graph_fallbacks']) and not target.capabilities['head_fallbacks']
                    if mode=='query_graph':assert provider.worker.query_captures==gcaps==10,'timed query capture'
                    value=dict(mode=mode,repetition=rep,profiled=profile,decode_ms=elapsed,output_tokens=len(tokens),output_sha256=tasks.token_sha(tokens),
                        output_ids=tokens,steps=steps,state_sha256=fp.state_sha(torch,checks,state.target),final_position=state.target.pos,
                        graph_replays=target.capabilities['graph_replays']-caps['graph_replays'],head_kernel_hashes_unchanged=True,
                        query_graph_captures=gcaps,query_graph=provider.worker.graph_capabilities if mode=='query_graph' else None)
                    return value
                finally:
                    target.run_predictions=old_run;target.commit=old_commit;backend.release(handle,state)
            for mode in ('baseline','query_graph'):
                report['active']='full512_fidelity:'+mode;save();r=generate(mode,-1)
                report['checks'].append(r);save();assert r['output_ids']==gold['output_ids'],'E49 greedy continuation mismatch'
            base,candidate=report['checks']
            assert base['steps']==candidate['steps'] and base['state_sha256']==candidate['state_sha256'],'candidate policy/state mismatch'
            for label in ('normal_before','profile','normal_after'):
                report['active']='target_profile:'+label;save()
                if label=='profile':
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=False,profile_memory=False,with_stack=False) as prof:
                        r=generate('baseline',-2,round_limit=16,profile=True)
                    path=source/'verify-trace.json';prof.export_chrome_trace(str(path));report['trace']=dict(file=path.name,sha256=helper.sha(path),bytes=path.stat().st_size)
                else:r=generate('baseline',-2,round_limit=16)
                r['label']=label;report['profiles'].append(r);save()
            assert len({r['output_sha256'] for r in report['profiles']})==len({r['state_sha256'] for r in report['profiles']})==1
            for rep in range(4):
                for mode in (('baseline','query_graph') if rep%2==0 else ('query_graph','baseline')):
                    report['active']=f'timed:{rep}:{mode}';save();r=generate(mode,rep)
                    assert r['steps']==base['steps'] and r['state_sha256']==base['state_sha256'] and r['output_ids']==base['output_ids']
                    report['timings'].append(r);save()
            totals={m:sum(r['decode_ms'] for r in report['timings'] if r['mode']==m) for m in ('baseline','query_graph')}
            per=[]
            for rep in range(4):
                pair={r['mode']:r['decode_ms'] for r in report['timings'] if r['repetition']==rep};per.append(pair['baseline']/pair['query_graph']-1)
            gain=totals['baseline']/totals['query_graph']-1
            report.update(seed_decode_gain=gain,per_repetition_gain=per,seeded_tps={m:4*511*1000/t for m,t in totals.items()},
                candidate_gate_passed=gain>=.01 and all(g>0 for g in per),prefix_immutable=checks.digest(torch,target.targets[0].workspace,len(ids))==prefix_sha)
            assert report['prefix_immutable'];report['status']='PASS_E50_PROBE_GAIN' if report['candidate_gate_passed'] else 'STOP_E50_NO_GAIN'
            save()
    finally:
        seed=features=None
        for p in providers.values():
            close=getattr(p.worker,'close',None)
            if close:close()
        if executor is not None:executor.close()
        if target is not None:
            for owner in list(target.owners):target.release_state(target.owners[owner].workspace)
            target.close()
        target=executor=None;providers.clear();gc.collect();torch.cuda.synchronize()
    return report
