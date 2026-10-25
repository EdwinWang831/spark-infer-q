"""Two native256K INT8 states, shared serial Graph scratch; BF16 control."""
import gc
import hashlib
import importlib
import json
import time

PROMPT=262080
KEYS_B=('BIRCH-6284','CREEK-1579','SPRUCE-8042')


def graph_gates(torch,target,state,ids,checks):
    prefix=state.snapshot();start=state.pos
    for parity in (0,1):
        for width in range(1,9):
            state.restore(prefix);keep=width if width%2 else width//2
            a,f,b=target.eager.run(state,ids[:width]);target.eager.commit(state,b,width,keep)
            expected=state.snapshot();cache=checks.tail(state,start)
            state.restore(prefix)
            if parity:state.rec[1].copy_(state.rec[0]);state.cur=[1]*len(state.cur)
            g,h,b=target.run(state,ids[:width]);target.commit(state,b,width,keep)
            assert torch.equal(a,g) and torch.equal(f,h),'shared Graph full logits/features differ'
            checks.exact(torch,state,expected,cache,start)
    state.restore(prefix)
    return dict(width_parity_buckets=16,full_logit_rows=72,features_exact=True,state_and_scales_exact=True)


def oracle(torch,target,state,first,eos,checks):
    prefix=state.snapshot();start=state.pos;gold=[first];teacher=[]
    # Initial row is evaluated by prefill and supplied separately by the caller.
    for _ in range(63):
        if gold[-1] in eos:break
        lg,_=target.eager.step(state,[gold[-1]],capture=False);teacher.append(lg[-1].float().cpu());gold.append(int(lg[-1].argmax()))
    result=dict(tokens=gold,teacher=teacher,snapshot=state.snapshot(),tail=checks.tail(state,start),start=start)
    state.restore(prefix);return result


def dual(session,source,tag,load,report,cases,fmt):
    torch=session.torch;helper=load('scripts/e47_run.py');checks=load('scripts/e47_state.py');old=load('scripts/e46_study.py')
    shared=importlib.import_module(tag+'.shared_graph');native_mod=importlib.import_module(tag+'.native_scheduler')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    target=shared.SharedGraphTarget.from_session(session,kv_dtype=fmt);executor=None
    teacher_result=None
    try:
        torch.cuda.reset_peak_memory_stats();target.warmup();old.guard(torch)
        expert,digest=helper.expert(session,tag);report['expert_binary_sha256']=digest
        executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,expert_backend=expert,kv_dtype=fmt),chunk_rows=1024)
        report.update(chunks=[],requests=[],lifecycle=[],long_graph=[],selector=[],dual_kv_dtype=fmt,capacity_only=True,
            quality_promoted=False,original_quality_gate_passed=False,shared_eager_buffers=all(t.eager is target.eager for t in target.targets),
            shared_screened_head=all(t.screen is target.screen for t in target.targets))
        def save():helper.save(source,report)
        with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=0) as native:
            def make():return shared.shared_graph_engine(native,target,eos_ids=session.eos,capacity=262144,token_budget=1024,tokenizer=session.tokenizer,prefill_executor=executor)
            with make() as engine:
                paused=[0.];start=time.perf_counter();engine.clock=lambda:time.perf_counter()-paused[0]
                handles={name:engine.submit_tokens(name,c['ids'],max_tokens=64) for name,c in cases.items()}
                states={name:engine.bridge.backend.states[h] for name,h in handles.items()}
                assert engine.active_requests==2 and len(target.owners)==2
                pointers=[{a.data_ptr() for _,a in checks.arrays(st,0,st.capacity)}|{st.rec.data_ptr(),st.conv.data_ptr()} for st in states.values()]
                assert pointers[0].isdisjoint(pointers[1]),'request cache/scale storage aliases'
                report['private_state_storage_disjoint']=True
                report['cache_bytes_per_request']={name:st.cache_bytes() for name,st in states.items()};report['admitted_memory']=old.guard(torch);save()
                oracles={};events={name:[] for name in cases};event_times={name:[] for name in cases};done={}
                original=engine.bridge.backend.release
                original_run=executor.provider.run;selector_seen=set()
                def observe(b,st,ids,*,capture):
                    result=original_run(b,st,ids,capture=capture)
                    if st.pos==PROMPT and id(st) in {id(s) for s in states.values()}:
                        t=time.perf_counter();name=next(n for n,s in states.items() if s is st)
                        rows=checks.selector(torch,b,st.pos)
                        report['selector'].append(dict(name=name,rows=rows));selector_seen.add(name)
                        paused[0]+=time.perf_counter()-t
                    return result
                executor.provider.run=observe
                def release(handle,st):
                    t=time.perf_counter();name=handle.request_id;gold=oracles[name]
                    checks.exact(torch,st,gold['snapshot'],gold['tail'],gold['start'])
                    after=checks.digest(torch,st,PROMPT);assert after==gold['prefix_sha256'],'long request prefix or scales changed'
                    done[name]=dict(state_exact=True,prefix_and_scales_unchanged=True,position=st.pos,prefix_sha256=after)
                    original(handle,st);paused[0]+=time.perf_counter()-t
                engine.bridge.backend.release=release
                try:
                    for step in range(1200):
                        if not engine.active_requests:break
                        before={name:st.pos for name,st in states.items()};t=time.perf_counter();out=engine.step();elapsed=(time.perf_counter()-t)*1000
                        mem=old.guard(torch)
                        advances=[dict(name=name,start=before[name],end=st.pos,rows=st.pos-before[name]) for name,st in states.items() if st.pos!=before[name]]
                        report['chunks'].append(dict(step=step,advances=advances,wall_ms=elapsed,**mem))
                        for event in out:
                            name=event.handle.request_id;events[name].extend(event.token_ids);event_times[name].append(time.perf_counter()-start-paused[0])
                            if name not in oracles:
                                assert states[name].pos==PROMPT and len(event.token_ids)==1
                                report['active']='diagnostic:'+name;save();t=time.perf_counter();st=states[name]
                                assert name in selector_seen,'missing per-request prefill observer'
                                gates=graph_gates(torch,target,st,[event.token_ids[0]]+cases[name]['ids'][-7:],checks)
                                report['long_graph'].append(dict(name=name,**gates))
                                # Re-evaluate final prefill logits was unnecessary: the
                                # token is known; teacher forcing below starts after it.
                                gold=oracle(torch,target,st,event.token_ids[0],session.eos,checks)
                                gold['prefix_sha256']=checks.digest(torch,st,PROMPT);oracles[name]=gold
                                paused[0]+=time.perf_counter()-t
                                del st
                        report['active']='dual256K:'+str(step);save()
                    assert not engine.active_requests and len(done)==2
                    for name,c in cases.items():
                        assert events[name]==oracles[name]['tokens'],'shared Graph output differs from private oracle'
                        text=session.tokenizer.decode(events[name],skip_special_tokens=True);times=event_times[name]
                        report['requests'].append(dict(name=name,prompt_tokens=PROMPT,output_tokens=len(events[name]),generated_text=text,
                            retrieval_keys=list(c['keys']),retrieval_hits=[k in text for k in c['keys']],
                            other_keys_absent=not any(k in text for k in cases['b' if name=='a' else 'a']['keys']),
                            oracle_tokens_exact=True,**done[name],first_event_active_seconds=times[0],last_event_active_seconds=times[-1],
                            maximum_output_gap_seconds=max((b-a for a,b in zip(times,times[1:])),default=0.),
                            output_sha256=hashlib.sha256(json.dumps(events[name],separators=(',',':')).encode()).hexdigest()))
                    teacher_result=dict(tokens=oracles['a']['tokens'],teacher=oracles['a']['teacher'])
                    report.update(dual_graph=target.capabilities,dual_active_seconds=time.perf_counter()-start-paused[0],
                        dual_raw_seconds=time.perf_counter()-start,dual_diagnostic_seconds=paused[0],
                        dual_peak_allocated_bytes=torch.cuda.max_memory_allocated(),dual_final_memory=old.memory(torch))
                    save()
                finally:
                    engine.bridge.backend.release=original;executor.provider.run=original_run
            # Reuse existing captured addresses after cancel, with short work.
            short=next(c['ids'] for c in load('scripts/e47_pilot.py').tasks(session) if c['name']=='explain')
            for phase in ('waiting','prepared_prompt','prepared_decode'):
                with make() as engine:
                    h=engine.submit_tokens('reuse',short,max_tokens=4)
                    if phase=='prepared_decode':
                        engine.step();assert engine.active_requests==1,'cancel fixture ended before decode'
                    if phase!='waiting':
                        p=engine.bridge.prepare();engine.bridge.execute(p)
                    engine.cancel(h)
                    if phase!='waiting':engine.bridge.complete(p)
                    list(engine.drain(max_steps=16))
                    h2=engine.submit_tokens('reuse',short,max_tokens=1)
                    result=list(engine.drain(max_steps=16))
                    assert h2.generation>h.generation and sum(len(e.token_ids) for e in result)==1
                assert not target.owners
                report['lifecycle'].append(dict(phase=phase,cancel_and_reuse=True));save()
        return teacher_result
    finally:
        if executor is not None:executor.close()
        if target.owners:
            for t in target.targets:
                if t.leased:target.release_state(t.workspace)
        target.close();gc.collect();torch.cuda.synchronize()


def bf16_control(session,source,tag,load,report,case,candidate):
    torch=session.torch;helper=load('scripts/e47_run.py');old=load('scripts/e46_study.py');checks=load('scripts/e47_state.py')
    long=importlib.import_module(tag+'.long_target');prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    target=long.LongTarget(session.weights,capacity=262160,kv_dtype='bf16');expert,_=helper.expert(session,tag)
    executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,expert_backend=expert),chunk_rows=1024)
    try:
        st=target.state();teacher=[];report['bf16_chunks']=[];start=time.perf_counter()
        for at in range(0,PROMPT,1024):
            report['active']='bf16_control:'+str(at//1024+1)+'/256'
            got=executor.advance(st,case['ids'][at:at+1024],capture=False)
            report['bf16_chunks'].append(dict(position=st.pos,**old.guard(torch)));helper.save(source,report)
        prefix=st.snapshot();anchor=int(got.logits[-1].argmax());prefill_ms=(time.perf_counter()-start)*1000
        for token in candidate['tokens'][:-1]:
            lg,_=target.step(st,[token],capture=False);teacher.append(lg[-1].float().cpu())
        comparison=checks.logit_metrics(torch,teacher,candidate['teacher'])
        st.restore(prefix);tokens=[anchor]
        for _ in range(63):
            if tokens[-1] in session.eos:break
            lg,_=target.step(st,[tokens[-1]],capture=False);tokens.append(int(lg[-1].argmax()))
        text=session.tokenizer.decode(tokens,skip_special_tokens=True)
        report['bf16_control']=dict(prompt_tokens=PROMPT,output_tokens=len(tokens),generated_text=text,
            retrieval_hits=[k in text for k in case['keys']],greedy_tokens_exact=tokens==candidate['tokens'],
            first_token_exact=anchor==candidate['tokens'][0],teacher_forced_logit_comparison=comparison,
            prefill_ms=prefill_ms,cache_bytes=st.cache_bytes())
        helper.save(source,report)
    finally:executor.close();del executor,target;gc.collect();torch.cuda.synchronize()


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e47_run.py');old=load('scripts/e46_study.py')
    p=session.root/'jobs/done'/args['study_job'];job=json.loads(p.read_text());pilot=job['result']
    assert job['status']=='PASS' and pilot['status']=='STOP_E47_QUALITY' and pilot['selected_kv_dtype'] is None
    assert pilot['target_object_id']==id(session.weights) and pilot['draft_object_id']==id(session.draft)
    for name,digest in pilot['source_sha256'].items():
        if name.startswith('spark_infer_q/'):assert helper.sha(source/name)==digest,name
    selected=pilot['summaries']['int8'];assert selected['greedy_exact_cases']==8 and selected['teacher_top1']==1 and not selected['lost_objectives'] and not selected['quality_gate_passed']
    report.update(status='RUNNING_E47_STUDY',pilot_job_sha256=helper.sha(p),pilot_outcome=pilot['status'])
    ids,identity=old.inputs(session);assert identity['token_sha256']=='994e53ff07618e1b5d48081837c94cb58910064278f5d3f6e2ed060a608bc4d7'
    a=dict(ids=ids,identity=identity,keys=old.KEYS);old.KEYS=KEYS_B;other,identity_b=old.inputs(session)
    cases=dict(a=a,b=dict(ids=other,identity=identity_b,keys=KEYS_B));report['inputs']={k:v['identity'] for k,v in cases.items()};helper.save(source,report)
    with torch.inference_mode():
        old.guard(torch);candidate=dual(session,source,tag,load,report,cases,'int8')
        gc.collect();torch.cuda.synchronize();old.guard(torch)
        bf16_control(session,source,tag,load,report,a,candidate)
    report['status']='PASS_E47_CAPACITY';report.pop('active',None);helper.save(source,report);return report
