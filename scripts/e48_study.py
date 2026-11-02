"""Contemporary native/interleave latency pair, with checks outside timers."""
import gc
import hashlib
import importlib
import json
import time


def token_sha(ids):return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()


def state_sha(torch,checks,st):
    h=hashlib.sha256(checks.digest(torch,st,st.pos).encode());snap=st.snapshot()
    h.update(str((snap['pos'],snap['mtp_len'])).encode())
    for name in ('rec','conv','ple_tail'):
        h.update(name.encode());h.update(snap[name].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    if snap['ple_history'] is not None:h.update(snap['ple_history'].tobytes())
    return h.hexdigest()


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e48_run.py');old=load('scripts/e46_study.py');checks=load('scripts/e47_state.py');tasks=load('scripts/e47_pilot.py')
    p=session.root/'jobs/done'/args['study_job'];job=json.loads(p.read_text());pilot=job['result']
    assert job['status']=='PASS' and pilot['status']=='PASS_E48_PILOT'
    assert pilot['target_object_id']==id(session.weights) and pilot['draft_object_id']==id(session.draft)
    for name,digest in pilot['source_sha256'].items():
        if name.startswith('spark_infer_q/'):assert helper.sha(source/name)==digest,name
    shared=importlib.import_module(tag+'.shared_graph');native_mod=importlib.import_module(tag+'.native_scheduler')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    report.update(status='RUNNING_E48_STUDY',pilot_job_sha256=helper.sha(p),short=[],long=[],lifecycle=[],
        kv_dtype='int8',quality_gate_promoted=False,scope='single_rank_two_target_only_slots; common_Graphs; no_HTTP_or_DFlash',
        timing='uninterrupted_native_token_API; model/Graph/group_preparation_and_postrun_checks_outside')
    ids_a,identity_a=old.inputs(session);keys_a=old.KEYS
    old.KEYS=('BIRCH-6284','CREEK-1579','SPRUCE-8042');ids_b,identity_b=old.inputs(session)
    assert identity_a['token_sha256']=='994e53ff07618e1b5d48081837c94cb58910064278f5d3f6e2ed060a608bc4d7'
    long_cases=dict(a=dict(ids=ids_a,identity=identity_a,keys=keys_a),b=dict(ids=ids_b,identity=identity_b,keys=old.KEYS))
    shorts=tasks.tasks(session);short_cases={name:dict(ids=next(c['ids'] for c in shorts if c['name']==task),keys=()) for name,task in (('a','long_en'),('b','long_zh'))}
    expert,digest=helper.expert(session,tag);report['expert_binary_sha256']=digest
    target=shared.SharedGraphTarget.from_session(session,kv_dtype='int8');executor=None
    def save():helper.save(source,report)
    try:
        with torch.inference_mode():
            old.guard(torch);target.warmup();assert target.capabilities['graph_captures']==32
            # prepare(rows) captures the bound screen, while the exact subset
            # specializes its candidate-count stride separately. Pay that setup
            # before request timing, including the first warmup request.
            x=torch.zeros((1,target.screen.k),dtype=torch.bfloat16,device=target.screen.weight.device)
            for count in range(1,129):target.screen.subset(x,torch.arange(count,device=x.device))
            torch.cuda.synchronize();del x
            report['head_subset_prewarm_counts']=list(range(1,129));save()
            executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,expert_backend=expert,kv_dtype='int8'),chunk_rows=1024)
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=0) as native:
                def make(policy):return shared.shared_graph_engine(native,target,eos_ids=session.eos,capacity=262144,token_budget=1024,
                    tokenizer=session.tokenizer,prefill_executor=executor,prefill_interleave=policy)
                def run(cases,policy,rep,kind,*,late=False,cap=64):
                    outputs={n:[] for n in cases};events={n:[] for n in cases};done={};plans=[];progress=[]
                    with make(policy) as engine:
                        old_release=engine.bridge.backend.release;old_step=engine.bridge.step
                        def release(h,st):
                            done[h.request_id]=st;old_release(h,st)
                        def observe():
                            bridge=engine.bridge
                            ready=any(e['seq'].status==native.core.SequenceStatus.RUNNING for e in bridge.requests.values())
                            waiting=any(e['seq'].status==native.core.SequenceStatus.WAITING for e in bridge.requests.values())
                            plan=old_step()
                            if plan.rows:plans.append(dict(ready_and_waiting=ready and waiting,phase='prompt' if plan.rows[0].prompt else 'decode',
                                rows=[dict(name=r.handle.request_id,position=r.position,rows=len(r.token_ids),prompt=r.prompt) for r in plan.rows]))
                            return plan
                        engine.bridge.backend.release=release;engine.bridge.step=observe
                        before=target.capabilities;kernel_before=dict(target.screen.kernel_sha256)
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();submitted={}
                        try:
                            for name,c in cases.items():
                                if late and name=='b':continue
                                submitted[name]=time.perf_counter()-start;engine.submit_tokens(name,c['ids'],max_tokens=cap)
                            for step in range(1200):
                                if not engine.active_requests:break
                                out=engine.step()
                                for e in out:
                                    name=e.handle.request_id;outputs[name].extend(e.token_ids)
                                    events[name].append(dict(emitted=len(e.token_ids),finish=e.finish_reason,position=e.processed_position,
                                        ttft_seconds=e.ttft_seconds,elapsed_seconds=e.elapsed_seconds))
                                if late and 'b' not in submitted and outputs['a']:
                                    submitted['b']=time.perf_counter()-start;engine.submit_tokens('b',cases['b']['ids'],max_tokens=cap)
                                mem=old.guard(torch)
                                if kind=='long':
                                    progress.append(dict(step=step,**mem))
                                    if step%16==0 or out:
                                        report['progress']=dict(kind=kind,interleave=policy,step=step,
                                            positions={n:st['state'].pos for n,st in engine.bridge.requests.items()},**mem);save()
                            torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000;peak=torch.cuda.max_memory_allocated()
                            assert not engine.active_requests and set(done)==set(cases)
                        finally:
                            engine.bridge.backend.release=old_release;engine.bridge.step=old_step
                        # All expensive cache/GDN/PLE checks are after the timer.
                        rows=[]
                        for name,c in cases.items():
                            ev=events[name];nonempty=[e for e in ev if e['emitted']]
                            assert nonempty and sum(e['emitted'] for e in ev)==len(outputs[name])
                            gaps=[b['elapsed_seconds']-a['elapsed_seconds'] for a,b in zip(nonempty,nonempty[1:])]
                            text=session.tokenizer.decode(outputs[name],skip_special_tokens=True)
                            rows.append(dict(name=name,prompt_tokens=len(c['ids']),input_sha256=token_sha(c['ids']),output_tokens=len(outputs[name]),
                                output_sha256=token_sha(outputs[name]),state_sha256=state_sha(torch,checks,done[name]),
                                final_position=done[name].pos,generated_text=text,retrieval_keys=list(c['keys']),retrieval_hits=[k in text for k in c['keys']],
                                ttft_ms=nonempty[0]['ttft_seconds']*1000,completion_ms=ev[-1]['elapsed_seconds']*1000,
                                maximum_output_gap_ms=max(gaps,default=0)*1000,finish_reason=ev[-1]['finish'],events=ev,submitted_ms=submitted[name]*1000))
                        after=target.capabilities
                        assert before['graph_captures']==after['graph_captures']==32 and not any(after['graph_fallbacks']) and after['head_fallbacks']==0
                        kernels_unchanged=kernel_before==target.screen.kernel_sha256
                        if rep>=0:assert kernels_unchanged,'timed head JIT changed'
                        record=dict(interleave=policy,repetition=rep,warmup=rep<0,late_admission=late,output_cap=cap,request_ms=wall,requests=rows,
                            plans=plans,graph_replays=after['graph_replays']-before['graph_replays'],graph_captures=32,
                            head_fallbacks=0,graph_fallbacks=after['graph_fallbacks'],timed_kernel_sources_unchanged=kernels_unchanged,
                            interleave_hints=dict(engine.bridge.interleave_hints),peak_allocated_bytes=peak,
                            minimum_host_available_bytes=min((r['MemAvailable'] for r in progress),default=None))
                    assert not target.owners
                    return record
                # Same real weights and same physical Graph slots for both arms.
                canonical=None
                for rep in range(-1,2):
                    for policy in ((False,True) if rep<0 or rep%2==0 else (True,False)):
                        report['active']='short:'+str((rep,policy));save()
                        r=run(short_cases,policy,rep,'short')
                        signature={x['name']:(x['output_sha256'],x['output_tokens'],x['final_position'],x['state_sha256']) for x in r['requests']}
                        if canonical is None:canonical=signature
                        assert signature==canonical,'short policy output/state differs'
                        r['outputs_and_state_exact']=True;report['short'].append(r);save()
                for policy in (False,True):
                    report['active']='late_admission:'+str(policy);save();r=run(short_cases,policy,0,'late',late=True)
                    assert {x['name']:(x['output_sha256'],x['output_tokens'],x['final_position'],x['state_sha256']) for x in r['requests']}==canonical
                    report.setdefault('late',[]).append(dict(r,outputs_and_state_exact=True));save()
                # Valid complete chat, explicit live check before preparing decode.
                short=next(c['ids'] for c in shorts if c['name']=='explain')
                for phase in ('waiting','prepared_prompt','prepared_decode'):
                    with make(True) as engine:
                        h=engine.submit_tokens('reuse',short,max_tokens=4)
                        if phase=='prepared_decode':engine.step();assert engine.active_requests==1
                        p=None
                        if phase!='waiting':p=engine.bridge.prepare();assert p.rows;engine.bridge.execute(p)
                        try:engine.cancel(h)
                        finally:
                            if p is not None:engine.bridge.complete(p)
                        cancelled=list(engine.drain(max_steps=16));assert cancelled[-1].finish_reason=='cancelled'
                        h2=engine.submit_tokens('reuse',short,max_tokens=1);reused=list(engine.drain(max_steps=16))
                        assert h2.generation>h.generation and sum(len(e.token_ids) for e in reused)==1
                    report['lifecycle'].append(dict(phase=phase,exact=True));save()
                canonical=None
                for policy in (False,True):
                    report['active']='long:'+str(policy);report.pop('progress',None);save()
                    r=run(long_cases,policy,0,'long')
                    signature={x['name']:(x['output_sha256'],x['output_tokens'],x['final_position'],x['state_sha256']) for x in r['requests']}
                    if canonical is None:canonical=signature
                    assert signature==canonical,'256K policy output/state differs'
                    assert all(x['retrieval_hits']==[True]*3 for x in r['requests'])
                    r['outputs_and_state_exact']=True;report['long'].append(r);save()
            control,candidate=report['long'];a={x['name']:x for x in control['requests']};b={x['name']:x for x in candidate['requests']}
            passed=(b['a']['maximum_output_gap_ms']<=.1*a['a']['maximum_output_gap_ms'] and b['a']['completion_ms']<a['a']['completion_ms'] and
                candidate['request_ms']<=1.05*control['request_ms'] and b['b']['ttft_ms']<=1.05*a['b']['ttft_ms'])
            report['latency_gate_passed']=passed;report['status']='PASS_E48_STUDY' if passed else 'STOP_E48_LATENCY_GATE'
            report['graph']=target.capabilities;report.pop('active',None);report.pop('progress',None);save()
    finally:
        if executor is not None:executor.close()
        for t in target.targets:
            if t.leased:target.release_state(t.workspace)
        target.close();gc.collect();torch.cuda.synchronize()
    return report
