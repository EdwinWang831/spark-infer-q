"""Matched old/fixed asynchronous SSE throughput with unchanged GPU fences."""
import hashlib,importlib,json,sys,time,http.client
from pathlib import Path
from types import ModuleType
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import Thread,Barrier


def run(session,arguments):
    root=Path(arguments['source_root']);before=Path('/workspace/e1/k4batch-20261023-wait-v1')
    tags=[]
    def load(source,label):
        seal=json.loads((source/'manifest.json').read_text())
        for n,h in seal.items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h,n
        tag='_k4_wait_service_'+label+'_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
        pkg=ModuleType(tag);pkg.__path__=[str(source/'spark_infer_q')];sys.modules[tag]=pkg;tags.append(tag)
        return tag,seal
    tag,manifest=load(root,'fixed');oldtag,oldseal=load(before,'old')
    assert oldseal['spark_infer_q/kaifra_batch.py']=='80cce3008dc4ddcb7c8bdb1efa9a81c93d599f47d1f45f94b1993c0ef04611b9'
    target=worker=server=server_thread=None
    report=dict(status='RUNNING',timings=[],source_sha256=manifest,baseline_source_sha256=oldseal,model_reloaded=False,
        contract=dict(modes=['old','fixed'],pairs=3,repetitions=2,
            change='old sealed sleep polling vs current per-plan event wait, same Target, provider and SSE implementation',
            timing='client-observed loopback SSE after both first outputs to both terminal streams; same per-request E17 traces',
            stream='same resident default CUDA stream, exclusively owned by this serialized job, selected explicitly on owner thread'))
    def save(active=None):
        if active is not None:report['active']=active
        p=root/'wait-service-result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    try:
        torch=session.torch;batch=importlib.import_module(tag+'.kaifra_batch');oldbatch=importlib.import_module(oldtag+'.kaifra_batch')
        wm=importlib.import_module(tag+'.kaifra_worker');oldwm=importlib.import_module(oldtag+'.kaifra_worker')
        sm=importlib.import_module(tag+'.kaifra_service');nm=importlib.import_module(tag+'.native_scheduler')
        opt=importlib.import_module(tag+'.kaifra_optimized');tm=importlib.import_module(tag+'.kaifra_joint_target');hm=importlib.import_module(tag+'.screened_head_fused')
        from e3_wide_target import WideTarget
        save('warmup');provider,metadata=opt.optimized_provider(session);report['profile_metadata']=metadata
        target=tm.JointTarget(WideTarget(session.weights,capacity=2048),hm.FusedScreenedHead(session.weights.head));target.warmup()
        texts=['Write a Python function to remove duplicates while preserving order. Return only code.','请用三点解释缓存与队列的区别。',
            'Read this context: '+('A cache stores recent values; a queue orders pending work. '*25)+' Explain the difference in one sentence.',
            'What is 17 + 25? Answer with only the integer.']
        groups=[dict(name='original_short',indices=[0,1],caps=[32,32]),dict(name='original_unequal',indices=[2,3],caps=[24,16]),dict(name='longer_decode',indices=[0,1],caps=[128,128])]
        ids=[session.tokenizer.apply_chat_template([dict(role='user',content=x)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False) for x in texts]
        prior=session.calibration['conditional_prior'];gold={};policies={}
        with nm.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
            def make(mode):
                impl=oldbatch if mode=='old' else batch
                engine=impl.batch_engine(native,target,provider,eos_ids=session.eos,prior=prior,capacity=768,tokenizer=session.tokenizer)
                metrics=[];engine.validation_metrics=metrics;collect=engine._collect_events
                def traced(plan):
                    events=collect(plan);metrics.extend(engine.last_step_metrics);return events
                engine._collect_events=traced;return engine
            for group in groups:
                for slot,index in enumerate(group['indices']):
                    save(dict(phase='reference',group=group['name'],slot=slot))
                    with make('fixed') as engine:
                        h=engine.submit_tokens('reference',ids[index],max_tokens=group['caps'][slot]);events=list(engine.drain())
                        key=(group['name'],slot);gold[key]=[t for ev in events for t in ev.token_ids]
                        policies[key]=[{k:m[k] for k in ('context_rows','verified_rows','emitted','accepted','observed','rates')}
                            for m in engine.validation_metrics if not m['prompt']]
            stream=torch.cuda.current_stream()
            @contextmanager
            def context():
                with torch.inference_mode(),torch.cuda.stream(stream):yield
            def start(mode):
                nonlocal worker,server,server_thread
                owner=oldwm.AsyncWorker if mode=='old' else wm.AsyncWorker
                worker=owner(lambda:make(mode),context_factory=context).start();native.bind_worker(worker.thread)
                server=sm.make_server(worker,('127.0.0.1',0));server_thread=Thread(target=server.serve_forever);server_thread.start()
            def stop():
                nonlocal worker,server,server_thread
                worker.close(60);server.shutdown();server.server_close();server_thread.join(5);worker=server=server_thread=None
            def client(group,slot,barrier):
                barrier.wait();start=time.perf_counter();conn=http.client.HTTPConnection(*server.server_address,timeout=60)
                conn.request('POST','/v1/chat/completions',body=json.dumps(dict(messages=[dict(role='user',content=texts[group['indices'][slot]])],max_tokens=group['caps'][slot])),headers={'Content-Type':'application/json'})
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
                    return dict(start=start,end=time.perf_counter(),chunks=chunks,request_id=response.getheader('X-Request-Id'),slot=slot)
                finally:response.close();conn.close()
            def pair(mode,group):
                barrier=Barrier(2);before_counts=dict(target.counters);kernels=set(target.screen.kernel_sha256);worker.engine.validation_metrics.clear()
                with ThreadPoolExecutor(2) as pool:rows=list(pool.map(lambda slot:client(group,slot,barrier),range(2)))
                worker.wait_idle(60)
                for r in rows:
                    actual=[{k:m[k] for k in ('context_rows','verified_rows','emitted','accepted','observed','rates')}
                        for m in worker.engine.validation_metrics if not m['prompt'] and m['request_id']==r['request_id']]
                    assert actual==policies[(group['name'],r['slot'])]
                cohort=max(r['chunks'][0][0] for r in rows);end=max(r['end'] for r in rows);begin=min(r['start'] for r in rows)
                return dict(mode=mode,group=group['name'],decode_tokens=sum(len(tokens) for r in rows for t,tokens in r['chunks'] if t>cohort),
                    decode_ms=(end-cohort)*1000,full_ms=(end-begin)*1000,completion_tokens=sum(len(tokens) for r in rows for _,tokens in r['chunks']),
                    ttft_ms=[(r['chunks'][0][0]-r['start'])*1000 for r in rows],counters={k:target.counters[k]-before_counts.get(k,0) for k in target.counters},
                    new_jit_shapes=sorted(set(target.screen.kernel_sha256)-kernels),outputs_policy_exact=True)
            for rep in range(2):
                for mode in (('old','fixed') if rep==0 else ('fixed','old')):
                    start(mode)
                    try:
                        for group in groups:
                            save(dict(phase='SSE',mode=mode,rep=rep,group=group['name']))
                            row=pair(mode,group)
                            if row['new_jit_shapes']:
                                report.setdefault('untimed_shape_warmups',[]).append(row);row=pair(mode,group)
                            assert not row['new_jit_shapes'];row['repetition']=rep;report['timings'].append(row);save()
                            print(json.dumps(dict(mode=mode,group=group['name'],tps=row['decode_tokens']*1000/row['decode_ms'])),flush=True)
                    finally:stop()
        summary={}
        for group in groups:
            summary[group['name']]={mode:sum(r['decode_tokens'] for r in report['timings'] if r['group']==group['name'] and r['mode']==mode)*1000/
                sum(r['decode_ms'] for r in report['timings'] if r['group']==group['name'] and r['mode']==mode) for mode in ('old','fixed')}
        assert not any(target.leased) and target.inflight is None and len(target.graphs)==80 and not target.fallbacks
        report.update(status='PASS_K4_WAIT_FIX_SERVICE',summary=summary,capabilities=target.capabilities,gloo_released=not torch.distributed.is_initialized())
        report.pop('active',None);save();return report
    finally:
        if worker is not None:worker.close(60)
        if server is not None:server.shutdown();server.server_close();server_thread.join(5)
        if target is not None:
            session.torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for state,leased in zip(target.states,target.leased):
                if leased:target.release_state(state)
            target.close()
        for name in list(sys.modules):
            if any(name==tag or name.startswith(tag+'.') for tag in tags):sys.modules.pop(name,None)
