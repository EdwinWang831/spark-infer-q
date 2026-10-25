"""Resident full-model adaptive batch, asynchronous service and matched-pair gate."""
import hashlib,importlib,json,sys,time,statistics,http.client,socket
from pathlib import Path
from types import ModuleType
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier,Thread


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    def check():
        for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    check();tag='_k4_gpu_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    target=None;worker=None;server=None;server_thread=None
    report=dict(status='RUNNING',correctness=[],timings=[],lifecycle=[],source_sha256=manifest,model_reloaded=False,
        contract=dict(policy='E17 NumPy / E29 W8 current post-draft per-request budget; not global optimization',
            control='same native plans, Target buckets/head, draft/predictor, worker and HTTP; split vs joint decode',
            timing='concurrent loopback SSE: admission to terminal, and post-both-first-output cohort; capture/diagnostics excluded',
            tpot='amortized observed SSE chunk interval per token; speculative tokens arrive in bursts'))
    def save(active=None):
        if active is not None:report['active']=active
        path=root/'gpu-result.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    save('initializing')
    try:
        batch=importlib.import_module(tag+'.kaifra_batch');nm=importlib.import_module(tag+'.native_scheduler')
        tm=importlib.import_module(tag+'.kaifra_batch_target');hm=importlib.import_module(tag+'.screened_head_fused')
        opt=importlib.import_module(tag+'.kaifra_optimized');wm=importlib.import_module(tag+'.kaifra_worker');sm=importlib.import_module(tag+'.kaifra_service')
        from e3_wide_target import WideTarget
        torch=session.torch
        provider,metadata=opt.optimized_provider(session);report['profile']=metadata
        target=tm.BatchTarget(WideTarget(session.weights,capacity=2048),hm.FusedScreenedHead(session.weights.head),graphs=True)
        save('warming 80 graphs');target.warmup();assert len(target.graphs)==80
        report['warm_capabilities']=target.capabilities
        # Independent eager buffers, immediate per-window commits: no shared reference alias.
        eager=WideTarget(session.weights,capacity=2048)
        texts=['Write a Python function to remove duplicates while preserving order. Return only code.',
            '请用三点解释缓存与队列的区别。',
            'Read this context: '+('A cache stores recent values; a queue orders pending work. '*25)+' Explain the difference in one sentence.',
            'What is 17 + 25? Answer with only the integer.']
        limits=[32,32,24,16]
        ids=[session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False) for text in texts]
        gold={};policy={};diagnostic_rows=state_checks=feature_rows=0
        before_env={k:__import__('os').environ.get(k) for k in ('DEEPINFER_USE_NEXTN_DECODE','DEEPINFER_ACCELERATOR')}
        with nm.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
            def make(joint=True,observer=None):
                engine=batch.batch_engine(native,target,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],
                    capacity=768,tokenizer=session.tokenizer,observer=observer,joint_decode=joint)
                engine.validation_metrics=[];collect=engine._collect_events
                def traced(plan):
                    events=collect(plan);engine.validation_metrics.extend(engine.last_step_metrics);return events
                engine._collect_events=traced
                return engine
            for pair in range(2):
                for joint in (False,True):
                    save(dict(phase='correctness',pair=pair,joint=joint));shadows={};out={};trace={};references={}
                    def observe(handle,pos,tokens,logits,keep,state):
                        nonlocal diagnostic_rows,state_checks,feature_rows
                        ref=shadows.setdefault(handle,eager.state());assert ref.pos==pos
                        lg,ft,buf=eager.run(ref,list(tokens),capture=True)
                        assert torch.equal(logits,lg),'committed/verification logits differ'
                        eager.commit(ref,buf,len(tokens),keep)
                        assert not eager.state_difference(state,ref),'live state differs'
                        current=engine.bridge.backend.states[handle]
                        assert torch.equal(current.context[-keep:],ft[:keep]),'accepted DFlash features differ'
                        diagnostic_rows+=len(tokens);feature_rows+=keep;state_checks+=1
                    with make(joint,observe) as engine:
                        for i in range(2):
                            index=pair*2+i;h=engine.submit_tokens(str(index),ids[index],max_tokens=limits[index]);out[h]=[];trace[h]=[]
                        while engine.active_requests:
                            for ev in engine.step():out[ev.handle].extend(ev.token_ids)
                            for m in engine.last_step_metrics:
                                if not m['prompt']:
                                    h=next(h for h in out if h.request_id==m['request_id'])
                                    trace[h].append({k:m[k] for k in ('context_rows','verified_rows','emitted','accepted','observed','rates')})
                        for h,tokens in out.items():
                            index=int(h.request_id)
                            if not joint:gold[index]=tokens;policy[index]=trace[h]
                            else:assert tokens==gold[index] and trace[h]==policy[index],('joint policy/output differs',index)
                        assert not any(target.leased) and not engine.bridge.backend.provider.providers
                        assert not engine.bridge.native.has_unfinished_seqs()
                        assert engine.bridge.blocks.get_num_free_blocks()==engine.bridge.block_count-1
                    report['correctness'].append(dict(pair=pair,joint=joint,completion_tokens={h.request_id:len(v) for h,v in out.items()},
                        policy={h.request_id:t for h,t in trace.items()},tokens={h.request_id:v for h,v in out.items()},logits_features_state_exact=True));save()
            # A one-token cap must preserve the pending-token invariant.
            with make() as engine:
                hs=[engine.submit_tokens('cap'+str(i),ids[i],max_tokens=1) for i in range(2)]
                events=list(engine.drain());assert [ev.token_ids for ev in events if ev.token_ids]==[(gold[0][0],),(gold[1][0],)]
                assert all(ev.processed_position==len(ids[i]) for i,h in enumerate(hs) for ev in events if ev.handle==h)
            report['lifecycle'].append(dict(case='one_token_cap',pending_position_exact=True))
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            @contextmanager
            def context():
                with torch.inference_mode(),torch.cuda.stream(stream):yield
            def start(joint=True,capacity=64):
                nonlocal worker,server,server_thread
                worker=wm.AsyncWorker(lambda:make(joint),context_factory=context,event_capacity=capacity).start()
                native.bind_worker(worker.thread)
                server=sm.make_server(worker,('127.0.0.1',0));server_thread=Thread(target=server.serve_forever);server_thread.start()
            def stop():
                nonlocal worker,server,server_thread
                if worker is not None:worker.close(60)
                if server is not None:server.shutdown();server.server_close();server_thread.join(5)
                worker=server=server_thread=None
            def http_request(path,body=None):
                conn=http.client.HTTPConnection(*server.server_address,timeout=60)
                conn.request('GET' if body is None else 'POST',path,body=None if body is None else json.dumps(body),headers={'Content-Type':'application/json'})
                return conn,conn.getresponse()
            def client(index,barrier):
                barrier.wait();started=time.perf_counter()
                conn,response=http_request('/v1/chat/completions',dict(messages=[dict(role='user',content=texts[index])],max_tokens=limits[index]))
                assert response.status==200,response.read();chunks=[];terminal=None
                try:
                    for line in response:
                        if not line.startswith(b'data: '):continue
                        data=line[6:].strip()
                        if data==b'[DONE]':break
                        ev=json.loads(data);assert ev['event']!='error',ev
                        if ev['event']=='output':
                            now=time.perf_counter()
                            if ev['token_ids']:chunks.append(dict(time=now,tokens=ev['token_ids']))
                            if ev['finish_reason']:terminal=ev['finish_reason']
                    assert terminal in ('length','stop'),terminal
                    tokens=[t for c in chunks for t in c['tokens']];assert tokens==gold[index],('SSE output differs',index)
                    return dict(index=index,request_id=response.getheader('X-Request-Id'),started=started,finished=time.perf_counter(),chunks=chunks,reason=terminal)
                finally:response.close();conn.close()
            # Alternate paired repetitions; the same client implementation measures both modes.
            for rep in range(3):
                for joint in ((False,True) if rep%2==0 else (True,False)):
                    start(joint)
                    try:
                        for pair in range(2):
                            save(dict(phase='http_timing',rep=rep,joint=joint,pair=pair))
                            for attempt in range(4):
                                before=dict(target.counters);barrier=Barrier(2);before_kernels=set(target.screen.kernel_sha256)
                                worker.engine.validation_metrics.clear()
                                with ThreadPoolExecutor(2) as pool:rows=list(pool.map(lambda i:client(i,barrier),range(pair*2,pair*2+2)))
                                worker.wait_idle(60)
                                for r in rows:
                                    metrics=[m for m in worker.engine.validation_metrics if m['request_id']==r['request_id'] and not m['prompt']]
                                    actual=[{k:m[k] for k in ('context_rows','verified_rows','emitted','accepted','observed','rates')} for m in metrics]
                                    assert actual==policy[r['index']],('SSE policy differs',r['index'])
                                new_kernels=set(target.screen.kernel_sha256)-before_kernels
                                if not new_kernels:break
                                report.setdefault('cold_shape_warmups',[]).append(dict(rep=rep,joint=joint,pair=pair,new_shapes=sorted(new_kernels)))
                            else:raise AssertionError('subset shapes did not stabilize')
                            end=max(r['finished'] for r in rows);begin=min(r['started'] for r in rows);cohort=max(r['chunks'][0]['time'] for r in rows)
                            decode_tokens=sum(len(c['tokens']) for r in rows for c in r['chunks'] if c['time']>cohort)
                            ttft=[(r['chunks'][0]['time']-r['started'])*1000 for r in rows];tpot=[]
                            for r in rows:
                                for prev,c in zip(r['chunks'],r['chunks'][1:]):tpot.extend([(c['time']-prev['time'])*1000/len(c['tokens'])]*len(c['tokens']))
                            delta={k:target.counters[k]-before.get(k,0) for k in target.counters}
                            if joint:assert delta['joint_forwards']>0,delta
                            else:assert delta.get('joint_forwards',0)==0,delta
                            report['timings'].append(dict(repetition=rep,joint=joint,pair=pair,full_ms=(end-begin)*1000,
                                completion_tokens=sum(len(c['tokens']) for r in rows for c in r['chunks']),decode_ms=(end-cohort)*1000,
                                decode_tokens=decode_tokens,ttft_ms=ttft,tpot_amortized_ms=tpot,counters=delta,reasons=[r['reason'] for r in rows],
                                policy_exact=True,new_head_jit_shapes=0,head_kernel_count=len(before_kernels)))
                            save();print(json.dumps(dict(rep=rep,joint=joint,pair=pair,status='PASS_SSE')),flush=True)
                    finally:stop()
            # Real in-flight cancel and stale-generation reuse; delay polling only,
            # leaving the actual GPU verification and commit events unchanged.
            start()
            try:
                from threading import Event
                launched=Event();allow=Event();bridge=worker.engine.bridge;poll=bridge.poll
                def gated(plan):
                    if hasattr(plan,'execution') and not launched.is_set():launched.set()
                    if launched.is_set() and not allow.is_set():return False
                    return poll(plan)
                bridge.poll=gated
                a=worker.submit_tokens('reuse',ids[0],max_tokens=32)
                assert launched.wait(60);prefix=[]
                while not a.events.empty():prefix.extend(a.receive(1).token_ids)
                assert a.cancel();assert target.inflight is not None
                a.close()
                with ThreadPoolExecutor(1) as pool:
                    future=pool.submit(worker.submit_tokens,'reuse',ids[1],max_tokens=8)
                    time.sleep(.02);assert not future.done(),'reused before GPU fence'
                    allow.set();b=future.result(60)
                assert b.handle.generation>a.handle.generation and not worker.cancel(a.handle)
                assert [t for ev in b for t in ev.token_ids]==gold[1][:8]
                worker.wait_idle(60);assert not any(target.leased)
                report['lifecycle'].append(dict(case='inflight_cancel_same_id_reuse',prefix_tokens=len(prefix),fence_before_reuse=True,stale_rejected=True))
                # HTTP disconnect while a real ticket is outstanding.
                launched.clear();allow.clear();conn,response=http_request('/v1/chat/completions',dict(messages=[dict(role='user',content=texts[0])],max_tokens=32))
                assert response.status==200;assert launched.wait(60)
                response.fp.raw._sock.shutdown(socket.SHUT_RDWR);response.close();conn.close()
                deadline=time.monotonic()+5
                while worker.snapshot()['open_streams']:
                    assert time.monotonic()<deadline;time.sleep(.02)
                assert target.inflight is not None;allow.set();worker.wait_idle(60)
                report['lifecycle'].append(dict(case='http_disconnect',cancel_before_fence=True,resources_released=True))
            finally:allow.set();stop()
            start(capacity=2)
            try:
                slow=worker.submit_tokens('slow',ids[0],max_tokens=32);worker.wait_idle(60)
                try:slow.receive(1)
                except wm.WorkerFailed as exc:assert str(exc)=='slow_consumer',str(exc)
                else:raise AssertionError('slow stream silently succeeded')
                assert not any(target.leased);assert worker.snapshot()['status']=='ready'
                report['lifecycle'].append(dict(case='slow_consumer',explicit_failure=True,worker_healthy=True))
            finally:stop()
        assert before_env=={k:__import__('os').environ.get(k) for k in before_env}
        assert not torch.distributed.is_initialized()
        assert len(target.graphs)==80 and not target.fallbacks,target.capabilities
        def percentile(values,p):
            values=sorted(values);return values[int(p*(len(values)-1))]
        summary={}
        for joint in (False,True):
            rows=[r for r in report['timings'] if r['joint']==joint];full=sum(r['full_ms'] for r in rows);decode=sum(r['decode_ms'] for r in rows)
            ttft=[x for r in rows for x in r['ttft_ms']];tpot=[x for r in rows for x in r['tpot_amortized_ms']]
            summary['joint' if joint else 'split']=dict(full_tps=sum(r['completion_tokens'] for r in rows)*1000/full,
                decode_tps=sum(r['decode_tokens'] for r in rows)*1000/decode,ttft_p50_ms=statistics.median(ttft),ttft_p95_ms=percentile(ttft,.95),
                tpot_amortized_p50_ms=statistics.median(tpot),tpot_amortized_p95_ms=percentile(tpot,.95))
        check();report.update(status='PASS_K4_BATCH_GPU_SERVICE',summary=summary,diagnostic_rows=diagnostic_rows,feature_rows=feature_rows,
            state_checks=state_checks,final_capabilities=target.capabilities,head_kernel_sha256=target.screen.kernel_sha256,
            gloo_restored=True,environment_restored=True)
        report.pop('active',None);save();return report
    finally:
        if worker is not None:worker.close(60)
        if server is not None:server.shutdown();server.server_close();server_thread.join(5)
        if target is not None:
            session.torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for st,leased in zip(target.states,target.leased):
                if leased:target.release_state(st)
            target.close()
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.'):sys.modules.pop(n,None)
