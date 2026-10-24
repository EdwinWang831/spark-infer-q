"""Matched E17 independent vs measured-singleton vs joint-cost decisions."""
import hashlib,importlib,json,sys,time,http.client,statistics,ctypes
from pathlib import Path
from types import ModuleType
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier,Thread

COST_ROOT='/workspace/e1/k4batch-20261023-joint-cost-v1'
COST_SHA='494337bd1dcfa027041e65d3d2669687d8b01fb26d4e36f458143670f842390a'


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_k4_joint_validate_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    torch=session.torch;target=worker=server=server_thread=None
    modes=('original','measured_singleton','joint')
    report=dict(status='RUNNING',source_sha256=manifest,model_reloaded=False,cost_sha256=COST_SHA,
        correctness=[],timings=[],service_timings=[],contract=dict(modes=list(modes),repetitions=3,
            policy='same frozen E17/W8/E33; original E29 per-request vs measured singleton vs joint expected tokens / batch wall',
            direct='after both first tokens through terminal/native retirement; first draft included, prefill excluded',
            service='same default CUDA stream and SSE handler; cohort after both first outputs, full request separately',
            comparison='widths and acceptance traces may change; final greedy tokens and eager logits/features/state must match',
            costs='frozen before evaluation; no fitting on evaluation outcomes; frozen paid W8 draft cost per actual drafted request'))
    def save(active=None):
        if active is not None:report['active']=active
        p=root/'joint-result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    try:
        opt=importlib.import_module(tag+'.kaifra_optimized');batch=importlib.import_module(tag+'.kaifra_batch')
        nm=importlib.import_module(tag+'.native_scheduler');tm=importlib.import_module(tag+'.kaifra_joint_target')
        hm=importlib.import_module(tag+'.screened_head_fused');jb=importlib.import_module(tag+'.joint_budget')
        wm=importlib.import_module(tag+'.kaifra_worker');sm=importlib.import_module(tag+'.kaifra_service')
        from e3_wide_target import WideTarget
        probe=arguments.get('service_probe',False)
        getcpu=ctypes.CDLL(None).sched_getcpu;getcpu.restype=ctypes.c_int
        provider,metadata=opt.optimized_provider(session);report['profile_metadata']=metadata
        policy=jb.JointBudget.load(Path(COST_ROOT)/'joint-costs.json',sha256=COST_SHA)
        class SingletonControl:
            sha256=policy.sha256
            validate_runtime=policy.validate_runtime
            def choose(self,requests):
                return dict(widths=tuple(policy.choose([r])['widths'][0] for r in requests),
                    independent_widths=tuple(r.independent_width for r in requests),scope='measured-singleton control')
        target=tm.JointTarget(WideTarget(session.weights,capacity=2048),hm.FusedScreenedHead(session.weights.head));save('graphs');target.warmup()
        eager=WideTarget(session.weights,capacity=2048)
        texts=['Write a Python function to remove duplicates while preserving order. Return only code.','请用三点解释缓存与队列的区别。',
            'Read this context: '+('A cache stores recent values; a queue orders pending work. '*25)+' Explain the difference in one sentence.',
            'What is 17 + 25? Answer with only the integer.']
        groups=[dict(name='original_short',indices=[0,1],caps=[32,32]),dict(name='original_unequal',indices=[2,3],caps=[24,16]),
            dict(name='longer_decode',indices=[0,1],caps=[128,128])]
        ids=[session.tokenizer.apply_chat_template([dict(role='user',content=x)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False) for x in texts]
        gold={};probabilities={};counters=dict(logits_rows=0,feature_rows=0,state_checks=0,shared_position_rate_checks=0)
        with nm.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
            def make(mode,observer=None):
                engine=batch.batch_engine(native,target,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],
                    capacity=768,tokenizer=session.tokenizer,observer=observer,
                    joint_budget=None if mode=='original' else (policy if mode=='joint' else SingletonControl()))
                engine.validation_metrics=[];engine.budget_decisions=[];collect=engine._collect_events
                def traced(plan):
                    result=collect(plan);engine.validation_metrics.extend(engine.last_step_metrics)
                    if hasattr(plan,'joint_budget'):engine.budget_decisions.append(plan.joint_budget)
                    return result
                engine._collect_events=traced;engine.probe_stages=[]
                if probe:
                    def wrap(obj,name):
                        original=getattr(obj,name)
                        def call(*args,**kwargs):
                            cpu=getcpu();started=time.perf_counter();result=original(*args,**kwargs)
                            engine.probe_stages.append(dict(api=name,ms=(time.perf_counter()-started)*1000,cpu=cpu,cpu_end=getcpu()))
                            return result
                        setattr(obj,name,call)
                    for name in ('prepare','poll','wait_for_device','_postdraft_widths'):wrap(engine.bridge,name)
                    wrap(engine.bridge.backend.provider,'propose')
                return engine
            def direct(mode,group,diagnostic=False,reference=False):
                outputs={};shadows={};positions={};before=dict(target.counters);kernels=set(target.screen.kernel_sha256)
                def observe(handle,pos,tokens,logits,keep,state):
                    ref=shadows.setdefault(handle,eager.state());assert ref.pos==pos
                    lg,ft,buf=eager.run(ref,list(tokens),capture=True);assert torch.equal(logits,lg),'joint logits differ'
                    eager.commit(ref,buf,len(tokens),keep);assert not eager.state_difference(state,ref),'joint state differs'
                    context=engine.bridge.backend.states[handle].context
                    assert torch.equal(context[-keep:],ft[:keep]),'accepted features differ'
                    counters['logits_rows']+=len(tokens);counters['feature_rows']+=keep;counters['state_checks']+=1
                with make(mode,observe if diagnostic else None) as engine:
                    for slot,index in enumerate(group['indices']):
                        h=engine.submit_tokens(str(slot),ids[index],max_tokens=group['caps'][slot]);outputs[h]=[]
                    cohort=None
                    while engine.active_requests:
                        for ev in engine.step():outputs[ev.handle].extend(ev.token_ids)
                        if cohort is None and all(outputs.values()):
                            assert all(len(v)==1 for v in outputs.values());cohort=time.perf_counter()
                    torch.cuda.synchronize();end=time.perf_counter()
                    for h,tokens in outputs.items():
                        key=(group['name'],int(h.request_id))
                        if reference:gold[key]=tokens
                        else:assert tokens==gold[key],('greedy output changed',mode,key)
                    for m in engine.validation_metrics:
                        if m['prompt']:continue
                        key=(group['name'],int(m['request_id']),m['context_rows'])
                        if key in probabilities:
                            assert m['rates']==probabilities[key],('same-position E17 changed',key)
                            counters['shared_position_rate_checks']+=1
                        else:probabilities[key]=m['rates']
                    assert not engine.bridge.backend.states and not engine.bridge.backend.provider.providers
                    assert not native.core is None and not engine.bridge.native.has_unfinished_seqs()
                    assert engine.bridge.blocks.get_num_free_blocks()==engine.bridge.block_count-1
                    return dict(mode=mode,group=group['name'],decode_ms=(end-cohort)*1000,
                        decode_tokens=sum(len(v)-1 for v in outputs.values()),completion_tokens=sum(map(len,outputs.values())),
                        new_jit_shapes=sorted(set(target.screen.kernel_sha256)-kernels),outputs_exact=True,
                        widths=[m['verified_rows'] for m in engine.validation_metrics if not m['prompt']],
                        rounds=sum(not m['prompt'] for m in engine.validation_metrics),
                        accepted=sum(m.get('accepted',0) for m in engine.validation_metrics),
                        observed=sum(m.get('observed',0) for m in engine.validation_metrics),
                        decision_count=len(engine.budget_decisions),changed_decisions=sum(tuple(d['widths'])!=tuple(d['independent_widths']) for d in engine.budget_decisions),
                        counters={k:target.counters[k]-before.get(k,0) for k in target.counters})
            for group in groups:
                for mode in modes:
                    if probe and mode!='original':continue
                    save(dict(phase='correctness',group=group['name'],mode=mode))
                    report['correctness'].append(direct(mode,group,diagnostic=not probe,reference=mode=='original'));save()
                    # Remove diagnostic full-head/observer effects from warmup.
                    for attempt in range(3):
                        row=direct(mode,group)
                        if not row['new_jit_shapes']:break
                    else:raise AssertionError('direct JIT did not stabilize')
            for rep in range(0 if probe else 3):
                for mode in modes[rep:]+modes[:rep]:
                    for group in groups:
                        save(dict(phase='direct',mode=mode,rep=rep,group=group['name']))
                        row=direct(mode,group);assert not row['new_jit_shapes'];row['repetition']=rep;report['timings'].append(row);save()
            stream=torch.cuda.current_stream()
            @contextmanager
            def context():
                with torch.inference_mode(),torch.cuda.stream(stream):yield
            def start(mode):
                nonlocal worker,server,server_thread
                worker=wm.AsyncWorker(lambda:make(mode),context_factory=context).start();native.bind_worker(worker.thread)
                server=sm.make_server(worker,('127.0.0.1',0));server_thread=Thread(target=server.serve_forever);server_thread.start()
            def stop():
                nonlocal worker,server,server_thread
                worker.close(60);server.shutdown();server.server_close();server_thread.join(5);worker=server=server_thread=None
            def client(group,slot,barrier):
                barrier.wait();begin=time.perf_counter();conn=http.client.HTTPConnection(*server.server_address,timeout=60)
                conn.request('POST','/v1/chat/completions',json.dumps(dict(messages=[dict(role='user',content=texts[group['indices'][slot]])],max_tokens=group['caps'][slot])),{'Content-Type':'application/json'})
                response=conn.getresponse();assert response.status==200,response.read();chunks=[];terminal=None
                try:
                    for line in response:
                        if not line.startswith(b'data: '):continue
                        data=line[6:].strip()
                        if data==b'[DONE]':break
                        ev=json.loads(data);assert ev['event']!='error',ev
                        if ev['event']=='output':
                            if ev['token_ids']:chunks.append((time.perf_counter(),ev['token_ids']))
                            if ev['finish_reason']:terminal=ev['finish_reason']
                    assert terminal in ('length','stop')
                    assert [t for _,tokens in chunks for t in tokens]==gold[(group['name'],slot)]
                    return dict(begin=begin,end=time.perf_counter(),chunks=chunks)
                finally:response.close();conn.close()
            def pair(mode,group):
                barrier=Barrier(2);kernels=set(target.screen.kernel_sha256);worker.engine.budget_decisions.clear();worker.engine.probe_stages.clear()
                with ThreadPoolExecutor(2) as pool:rows=list(pool.map(lambda slot:client(group,slot,barrier),range(2)))
                worker.wait_idle(60);cohort=max(r['chunks'][0][0] for r in rows);end=max(r['end'] for r in rows)
                tpot=[(b[0]-a[0])*1000/len(b[1]) for r in rows for a,b in zip(r['chunks'],r['chunks'][1:]) for _ in b[1]]
                decisions=worker.engine.budget_decisions
                return dict(mode=mode,group=group['name'],decode_ms=(end-cohort)*1000,full_ms=(end-min(r['begin'] for r in rows))*1000,
                    decode_tokens=sum(len(v) for r in rows for t,v in r['chunks'] if t>cohort),completion_tokens=sum(len(v) for r in rows for _,v in r['chunks']),
                    ttft_ms=[(r['chunks'][0][0]-r['begin'])*1000 for r in rows],tpot_amortized_ms=tpot,
                    decision_count=len(decisions),changed_decisions=sum(tuple(d['widths'])!=tuple(d['independent_widths']) for d in decisions),
                    probe_stages=list(worker.engine.probe_stages),
                    new_jit_shapes=sorted(set(target.screen.kernel_sha256)-kernels),outputs_exact=True)
            for rep in range(4 if probe else 3):
                order=modes[rep%3:]+modes[:rep%3] if probe else (('original','joint') if rep%2==0 else ('joint','original'))
                for mode in order:
                    start(mode)
                    try:
                        for group in groups:
                            save(dict(phase='SSE',mode=mode,rep=rep,group=group['name']))
                            for attempt in range(3):
                                row=pair(mode,group)
                                if not row['new_jit_shapes']:break
                                report.setdefault('service_shape_warmups',[]).append(row)
                            else:raise AssertionError('SSE JIT did not stabilize')
                            row['repetition']=rep;report['service_timings'].append(row);save()
                    finally:stop()
        def summarize(rows):
            return {g['name']:{mode:sum(r['decode_tokens'] for r in rows if r['group']==g['name'] and r['mode']==mode)*1000/
                sum(r['decode_ms'] for r in rows if r['group']==g['name'] and r['mode']==mode) for mode in sorted({r['mode'] for r in rows})} for g in groups}
        assert not any(target.leased) and target.inflight is None and not target.fallbacks
        report.update(status='PASS_K4_JOINT_BUDGET',summary=summarize(report['timings']),service_summary=summarize(report['service_timings']),
            correctness_counts=counters,capabilities=target.capabilities,gloo_released=not torch.distributed.is_initialized())
        report.pop('active',None);save();return report
    finally:
        if worker is not None:worker.close(60)
        if server is not None:server.shutdown();server.server_close();server_thread.join(5)
        if target is not None:
            torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for st,leased in zip(target.states,target.leased):
                if leased:target.release_state(st)
            target.close()
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.'):sys.modules.pop(n,None)
