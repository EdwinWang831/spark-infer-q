"""Matched native prefill fairness / E40 integration, plus draft wall probes."""
import hashlib
import importlib
import json
import http.client
import ctypes
import os
from pathlib import Path
import sys
import time
from types import ModuleType
from threading import Barrier, Event, Thread
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_k4_progress_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    torch=session.torch;target=executor=worker=server=server_thread=None
    affinity_service=bool(arguments.get('affinity_service'))
    modes=('fast_prefill','fast_cpu') if affinity_service else ('baseline','interleave','fast_prefill')
    probe_only=bool(arguments.get('draft_probe_only'))
    report=dict(status='RUNNING',source_sha256=manifest,model_reloaded=False,correctness=[],timings=[],draft_probe=[],
        contract=dict(modes=modes,repetitions=0 if probe_only else 5,target='owned compact',draft='same owned batch DFlash/Q4/W8/E17/E29',
            baseline='rows8 prompt, native token budget32, prompt polling each step',
            interleave='same rows8/budget32, one ready decode step between prompt plans',
            fast_prefill='same interleaving, E40 owned executor and native prompt budget128',
            warm='no new Target capture or exact-head JIT inside formal timing',
            timing='earliest first output to cohort retirement, subtract exactly one token per request; full request separately',
            staggered='long request begins only after short request first output, no fixed delay'))
    def save(active=None):
        if active is not None:report['active']=active
        p=root/'progress-result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    try:
        opt=importlib.import_module(tag+'.kaifra_optimized');batch=importlib.import_module(tag+'.kaifra_batch')
        nm=importlib.import_module(tag+'.native_scheduler');cm=importlib.import_module(tag+'.compact_target')
        hm=importlib.import_module(tag+'.screened_head_fused');loader=importlib.import_module(tag+'.qwen_state_kernel')
        dl=importlib.import_module(tag+'.draft_batch_w8');pm=importlib.import_module(tag+'.prefill');po=importlib.import_module(tag+'.prefill_tensorfold')
        wm=importlib.import_module(tag+'.kaifra_worker');sm=importlib.import_module(tag+'.kaifra_service')
        cp=importlib.import_module(tag+'.worker_cpu') if affinity_service else None
        from e3_wide_target import WideTarget
        component=json.loads(Path('/workspace/e1/k4batch-20261023-target-component-v6/target-component.json').read_text())
        draft=json.loads(Path('/workspace/e1/k4batch-20261023-batch-draft-component-v5/batch-draft-result.json').read_text())
        ext=loader.load_state_kernel(component['kernel']['library'],component['kernel']['binary_sha256'])
        matmul,identity=dl.load_library(draft['kernel']['library'],draft['kernel']['binary_sha256'])
        base,profile=opt.optimized_provider(session);report['profile_metadata']=profile
        target=cm.CompactTarget(WideTarget(session.weights,capacity=2048),hm.FusedScreenedHead(session.weights.head),ext)
        target.warmup();executor=pm.PrefillExecutor(po.TensorFoldPrefillProvider(session.weights,torch,capacity=2048),chunk_rows=128)
        oracle=WideTarget(session.weights,capacity=2048)
        texts=['Write a Python function to remove duplicates while preserving order. Return only code.',
            '请用三点解释缓存与队列的区别。',
            'Read this context: '+('A cache stores recent values; a queue orders pending work. '*25)+' Explain the difference in one sentence.']
        groups=[dict(name='short',indices=[0,1],caps=[32,32]),dict(name='cap128',indices=[0,1],caps=[128,128]),
            dict(name='staggered_prefill',indices=[1,2],caps=[32,24],staggered=True)]
        if probe_only:groups=groups[:1]
        ids=[session.tokenizer.apply_chat_template([dict(role='user',content=x)],tokenize=True,add_generation_prompt=True,
            enable_thinking=False,return_dict=False) for x in texts]
        report['prompt_lengths']=list(map(len,ids));gold={}
        stream=torch.cuda.current_stream()
        @contextmanager
        def context(affinity=None,performance=False):
            previous=os.sched_getaffinity(0) if affinity is not None else None
            try:
                if affinity is not None:os.sched_setaffinity(0,affinity)
                if performance:
                    with cp.worker_cpu('performance'),torch.inference_mode(),torch.cuda.stream(stream):yield
                else:
                    with torch.inference_mode(),torch.cuda.stream(stream):yield
            finally:
                if previous is not None:os.sched_setaffinity(0,previous)
        def make(mode):
            provider=batch.BatchedRequestDraftProvider(base,matmul,kernel_identity=identity)
            engine=batch.batch_engine(native,target,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],
                capacity=768,tokenizer=session.tokenizer,interleave_prefill=mode!='baseline',
                prompt_token_budget=128 if mode.startswith('fast') else 32,prefill_executor=executor if mode.startswith('fast') else None)
            engine.capabilities['worker_cpu_policy']='performance' if mode=='fast_cpu' else 'default'
            engine.metrics=[];engine.plan_trace=[];collect=engine._collect_events
            def traced(plan):
                result=collect(plan);engine.metrics.extend(engine.last_step_metrics)
                engine.plan_trace.append([dict(request=row.handle.request_id,prompt=row.prompt,rows=len(row.token_ids)) for row in plan.rows])
                return result
            engine._collect_events=traced
            return engine
        counts=dict(logits_rows=0,feature_rows=0,state_checks=0,prefill_checks=0)
        def diagnostic(mode,group):
            shadows={};saved={};captured=[]
            launch,commit,retire=target.launch,target.commit,target.retire
            advance=executor.advance
            def capture(*args,**kwargs):
                result=advance(*args,**kwargs);captured.append(result);return result
            def exact_launch(requests,**kwargs):return launch(requests,diagnostic=True)
            def exact_commit(ticket,keeps):
                for win in ticket.windows:
                    ref=shadows[id(win.state)];lg,ft,b=oracle.run(ref,list(win.ids));a=win.at;n=len(win.ids)
                    assert torch.equal(lg,ticket.full_logits[a:a+n]),'decode logits'
                    assert torch.equal(ft,ticket.features[a:a+n]),'decode features'
                    if keeps.get(id(win.state),0):oracle.commit(ref,b,n,keeps[id(win.state)])
                    counts['logits_rows']+=n;counts['feature_rows']+=n
                return commit(ticket,keeps)
            def exact_retire(ticket):
                for win in ticket.windows:
                    assert not oracle.state_difference(win.state,shadows[id(win.state)]),'decode state'
                    counts['state_checks']+=1
                return retire(ticket)
            target.launch,target.commit,target.retire=exact_launch,exact_commit,exact_retire;executor.advance=capture
            try:
                with make(mode) as engine:
                    original=engine.bridge.backend.advance
                    def prompt(row,state,cap):
                        result=original(row,state,cap)
                        if row.prompt:
                            ref=shadows.setdefault(id(state.target),oracle.state());features=[]
                            for at in range(0,len(row.token_ids),8):
                                lg,ft=oracle.step(ref,list(row.token_ids[at:at+8]));features.append(ft)
                            ft=torch.cat(features)
                            assert torch.equal(ft,state.context[-len(ft):]),'prompt features'
                            assert not oracle.state_difference(state.target,ref),'prompt state'
                            if mode.startswith('fast'):assert torch.equal(captured.pop().logits,lg[-1:]),'prompt final logits'
                            counts['prefill_checks']+=1
                        return result
                    engine.bridge.backend.advance=prompt
                    hs=[engine.submit_tokens(str(i),ids[index],max_tokens=group['caps'][i]) for i,index in enumerate(group['indices'])]
                    out={h:[] for h in hs}
                    for ev in engine.drain():out[ev.handle].extend(ev.token_ids)
                    for i,h in enumerate(hs):
                        key=(group['name'],i)
                        if mode==modes[0]:gold[key]=out[h]
                        else:assert out[h]==gold[key],('output',mode,key)
                    assert not any(target.leased) and target.inflight is None
                    return dict(mode=mode,group=group['name'],outputs_exact=True,plan_trace=engine.plan_trace,
                        widths=[m['verified_rows'] for m in engine.metrics if not m['prompt']])
            finally:target.launch,target.commit,target.retire=launch,commit,retire;executor.advance=advance
        def start(mode,probe=False,affinity=None):
            nonlocal worker,server,server_thread
            worker=wm.AsyncWorker(lambda:make(mode),context_factory=lambda:context(affinity,mode=='fast_cpu')).start();native.bind_worker(worker.thread)
            if probe:instrument(worker.engine)
            server=sm.make_server(worker,('127.0.0.1',0));server_thread=Thread(target=server.serve_forever);server_thread.start()
        def stop():
            nonlocal worker,server,server_thread
            worker.close(60);server.shutdown();server.server_close();server_thread.join(5);worker=server=server_thread=None
        def client(group,slot,barrier,first):
            if group.get('staggered'):
                if slot==1:assert first.wait(60),'short never emitted'
            else:barrier.wait()
            begin=time.perf_counter();conn=http.client.HTTPConnection(*server.server_address,timeout=60)
            conn.request('POST','/v1/chat/completions',json.dumps(dict(messages=[dict(role='user',content=texts[group['indices'][slot]])],
                max_tokens=group['caps'][slot])),{'Content-Type':'application/json'})
            response=conn.getresponse();assert response.status==200,response.read();chunks=[];terminal=None
            try:
                for line in response:
                    if not line.startswith(b'data: '):continue
                    data=line[6:].strip()
                    if data==b'[DONE]':break
                    ev=json.loads(data);assert ev['event']!='error',ev
                    if ev['event']=='output':
                        if ev['token_ids']:
                            chunks.append((time.perf_counter(),ev['token_ids']))
                            if slot==0:first.set()
                        if ev['finish_reason']:terminal=ev['finish_reason']
                assert terminal in ('stop','length')
                assert [t for _,ts in chunks for t in ts]==gold[(group['name'],slot)]
                return dict(begin=begin,end=time.perf_counter(),chunks=chunks)
            finally:response.close();conn.close()
        def pair(mode,group):
            before=dict(target.counters);jit=set(target.screen.kernel_sha256);worker.engine.metrics.clear();worker.engine.plan_trace.clear()
            barrier=Barrier(2);first=Event()
            with ThreadPoolExecutor(2) as pool:rows=list(pool.map(lambda i:client(group,i,barrier,first),range(2)))
            worker.wait_idle(60);end=max(r['end'] for r in rows);cohort=min(r['chunks'][0][0] for r in rows)
            return dict(mode=mode,group=group['name'],full_ms=(end-min(r['begin'] for r in rows))*1000,
                decode_ms=(end-cohort)*1000,completion_tokens=sum(len(ts) for r in rows for _,ts in r['chunks']),
                decode_tokens=sum(len(ts) for r in rows for _,ts in r['chunks'])-2,
                ttft_ms=[(r['chunks'][0][0]-r['begin'])*1000 for r in rows],
                request_ms=[(r['end']-r['begin'])*1000 for r in rows],
                request_tpot_ms=[(r['chunks'][-1][0]-r['chunks'][0][0])*1000/max(1,sum(len(ts) for _,ts in r['chunks'])-1) for r in rows],
                max_chunk_gap_ms=[max(((b[0]-a[0])*1000 for a,b in zip(r['chunks'],r['chunks'][1:])),default=0.) for r in rows],
                chunk_gap_ms=[(b[0]-a[0])*1000 for r in rows for a,b in zip(r['chunks'],r['chunks'][1:])],
                stage_ms={k:sum(m[k] for m in worker.engine.metrics if not m['prompt']) for k in ('draft_ms','verify_ms','commit_ms')},
                prompt_ms=sum(m['verify_ms'] for m in worker.engine.metrics if m['prompt']),
                plan_trace=list(worker.engine.plan_trace),outputs_exact=True,
                captures=target.counters['captures']-before.get('captures',0),new_jit_shapes=sorted(set(target.screen.kernel_sha256)-jit),
                counters={k:target.counters[k]-before.get(k,0) for k in target.counters})
        def instrument(engine):
            engine.draft_probe=[];backend=engine.bridge.backend.provider.backend
            getcpu=ctypes.CDLL(None).sched_getcpu;getcpu.restype=ctypes.c_int
            for name in ('forward_hidden','select'):
                original=getattr(backend,name)
                def call(*args,_name=name,_original=original,**kwargs):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    signature=([dict(context_rows=len(r.features),anchor=r.anchor) for r in args[0]]
                        if _name=='forward_hidden' else list(args[1]))
                    cpu_begin=getcpu();wall=time.perf_counter();cpu=time.thread_time();start.record();result=_original(*args,**kwargs);end.record()
                    engine.draft_probe.append(dict(api=_name,wall_ms=(time.perf_counter()-wall)*1000,
                        thread_cpu_ms=(time.thread_time()-cpu)*1000,signature=signature,cpu_begin=cpu_begin,cpu_end=getcpu(),start=start,end=end))
                    return result
                setattr(backend,name,call)
        with nm.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
            for group in groups:
                for mode in modes:
                    save(dict(phase='correctness',mode=mode,group=group['name']))
                    report['correctness'].append(diagnostic(mode,group));save()
            for mode in modes:
                start(mode)
                try:
                    for group in groups:
                        for attempt in range(5):
                            save(dict(phase='warmup',mode=mode,group=group['name']))
                            row=pair(mode,group)
                            if not row['captures'] and not row['new_jit_shapes']:break
                        else:raise AssertionError('serving warmup did not stabilize')
                finally:stop()
            for rep in range(0 if probe_only else 5):
                order=modes[rep%len(modes):]+modes[:rep%len(modes)]
                for mode in order:
                    start(mode)
                    try:
                        for group in groups:
                            save(dict(phase='SSE',mode=mode,group=group['name'],repetition=rep))
                            row=pair(mode,group)
                            if row['captures'] or row['new_jit_shapes']:
                                report.setdefault('late_shape_warmups',[]).append(row)
                                row=pair(mode,group)
                            assert not row['captures'] and not row['new_jit_shapes']
                            row['repetition']=rep;report['timings'].append(row);save()
                    finally:stop()
            # Isolated eager draft spans on the same workload. GPU spans include
            # launch gaps; they are not a sum of individual kernel durations.
            scopes=() if affinity_service else ('direct','SSE')
            affinity_groups={}
            if arguments.get('affinity_probe'):
                frequencies={cpu:int((Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'cpufreq/cpuinfo_max_freq').read_text())
                    for cpu in os.sched_getaffinity(0)}
                assert min(frequencies.values())<max(frequencies.values())
                affinity_groups={name:{cpu for cpu,freq in frequencies.items() if freq==value}
                    for name,value in (('SSE_fast',max(frequencies.values())),('SSE_slow',min(frequencies.values())))}
                report['cpu_max_khz']=frequencies;report['affinity_groups']={k:sorted(v) for k,v in affinity_groups.items()}
                scopes=('direct','SSE','SSE_fast','SSE_slow')
            for rep in range(3 if scopes else 0):
                for scope in scopes[rep%len(scopes):]+scopes[:rep%len(scopes)]:
                    group=groups[0];mode='fast_prefill';save(dict(phase='draft_probe',scope=scope,repetition=rep))
                    if scope.startswith('SSE'):
                        start(mode,True,affinity_groups.get(scope))
                        try:pair(mode,group);records=worker.engine.draft_probe
                        finally:stop()
                    else:
                        with make(mode) as engine:
                            instrument(engine)
                            for i,index in enumerate(group['indices']):engine.submit_tokens(str(i),ids[index],max_tokens=group['caps'][i])
                            list(engine.drain());records=engine.draft_probe
                    torch.cuda.synchronize()
                    report['draft_probe'].append(dict(scope=scope,repetition=rep,records=[dict(api=r['api'],wall_ms=r['wall_ms'],
                        thread_cpu_ms=r['thread_cpu_ms'],signature=r['signature'],cpu_begin=r['cpu_begin'],cpu_end=r['cpu_end'],
                        gpu_span_ms=r['start'].elapsed_time(r['end'])) for r in records]));save()
        assert target.inflight is None and not any(target.leased) and not torch.distributed.is_initialized()
        if affinity_service:report['contract'].update(experiment='same fast prefill, default vs performance-core owner thread',repetitions=5)
        report.update(status='PASS_K4_PROGRESS_CPU_SERVICE' if affinity_service else ('PASS_K4_PROGRESS_DRAFT' if probe_only else 'PASS_K4_PROGRESS_SERVICE'),correctness_counts=counts,capabilities=target.capabilities,
            gloo_restored=True,environment_restored=True);report.pop('active',None);save();return report
    finally:
        if worker is not None:worker.close(60)
        if server is not None:server.shutdown();server.server_close();server_thread.join(5)
        if executor is not None:executor.close()
        if target is not None:
            torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for st,leased in zip(target.states,target.leased):
                if leased:target.release_state(st)
            target.close()
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.'):sys.modules.pop(n,None)
