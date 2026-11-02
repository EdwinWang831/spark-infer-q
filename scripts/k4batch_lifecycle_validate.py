"""Real GPU NEXTN=0/E17, survivor cancellation, bounded shutdown and fault fences."""
import hashlib,importlib,json,sys,time,os
from pathlib import Path
from types import ModuleType
from contextlib import contextmanager
from threading import Event
from concurrent.futures import ThreadPoolExecutor


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_k4_lifecycle_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    target=worker=None;cases=[];owned_providers=[]
    def save(active):
        (root/'lifecycle-progress.json').write_text(json.dumps(dict(active=active,cases=cases),indent=2)+'\n')
    try:
        batch=importlib.import_module(tag+'.kaifra_batch');nm=importlib.import_module(tag+'.native_scheduler')
        tm=importlib.import_module(tag+'.kaifra_batch_target');hm=importlib.import_module(tag+'.screened_head_fused')
        opt=importlib.import_module(tag+'.kaifra_optimized');wm=importlib.import_module(tag+'.kaifra_worker')
        from e3_wide_target import WideTarget
        torch=session.torch;env={k:os.environ.get(k) for k in ('DEEPINFER_USE_NEXTN_DECODE','DEEPINFER_ACCELERATOR')}
        save('warmup')
        if arguments.get('target_state_library'):
            cm=importlib.import_module(tag+'.compact_target');loader=importlib.import_module(tag+'.qwen_state_kernel')
            ext=loader.load_state_kernel(arguments['target_state_library'],arguments['target_state_sha256'])
            target=cm.CompactTarget(WideTarget(session.weights,capacity=2048),hm.FusedScreenedHead(session.weights.head),ext)
        else:target=tm.BatchTarget(WideTarget(session.weights,capacity=2048),hm.FusedScreenedHead(session.weights.head))
        target.warmup()
        texts=['What is 17 + 25? Answer with only the integer.','请用三点解释缓存与队列的区别。']
        ids=[session.tokenizer.apply_chat_template([dict(role='user',content=x)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False) for x in texts]
        prior=session.calibration['conditional_prior'];gold={}
        joint_budget=None
        if arguments.get('joint_budget'):
            controller=importlib.import_module(tag+'.joint_budget')
            joint_budget=controller.JointBudget.load('/workspace/e1/k4batch-20261023-joint-cost-v1/joint-costs.json',
                sha256='494337bd1dcfa027041e65d3d2669687d8b01fb26d4e36f458143670f842390a')
        for depth in (0,7):
            provider=opt.optimized_provider(session)[0] if depth else None
            if depth and arguments.get('batch_draft_library'):
                loader=importlib.import_module(tag+'.draft_batch_w8')
                matmul,identity=loader.load_library(arguments['batch_draft_library'],arguments['batch_draft_sha256'])
                provider=batch.BatchedRequestDraftProvider(provider,matmul,kernel_identity=identity,
                    buffers=bool(arguments.get('draft_buffers')),graphs=bool(arguments.get('draft_graphs')))
                owned_providers.append(provider)
            with nm.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=depth) as native:
                def make():
                    executor=None
                    if arguments.get('owned_prefill'):
                        pm=importlib.import_module(tag+'.prefill');po=importlib.import_module(tag+'.prefill_tensorfold')
                        executor=pm.PrefillExecutor(po.TensorFoldPrefillProvider(session.weights,torch,capacity=2048),chunk_rows=128)
                    engine=batch.batch_engine(native,target,provider,eos_ids=session.eos,prior=prior,capacity=256,
                        target_only=not depth,joint_budget=joint_budget if depth else None,
                        prefill_executor=executor,prompt_token_budget=128 if executor else 32,
                        interleave_prefill=bool(arguments.get('interleave_prefill')))
                    if executor is not None:
                        close=engine.close
                        def close_with_prefill():
                            close();executor.close()
                        engine.close=close_with_prefill
                    return engine
                def audit(engine):
                    assert not any(target.leased) and target.inflight is None
                    assert not engine.bridge.backend.states and not engine.bridge.native.has_unfinished_seqs()
                    assert engine.bridge.blocks.get_num_free_blocks()==engine.bridge.block_count-1
                save('reference:'+str(depth))
                with make() as engine:
                    hs=[engine.submit_tokens(str(i),ids[i],max_tokens=16) for i in range(2)];out={h:[] for h in hs};terminal={}
                    for ev in engine.drain():
                        out[ev.handle].extend(ev.token_ids)
                        if ev.finish_reason:terminal[ev.handle]=ev.finish_reason
                    for i,h in enumerate(hs):
                        if not depth:gold[i]=out[h]
                        else:assert out[h]==gold[i]
                    assert terminal[hs[0]]=='stop' and gold[0][-1] in session.eos
                    audit(engine);cases.append(dict(case='dual_eos_and_cap',depth=depth,lengths=[len(out[h]) for h in hs],reasons=list(terminal.values()),exact=True))
                with make() as engine:
                    hs=[engine.submit_tokens(str(i),ids[i],max_tokens=1) for i in range(2)];events=list(engine.drain())
                    for i,h in enumerate(hs):
                        e=next(e for e in events if e.handle==h);assert e.token_ids==(gold[i][0],) and e.processed_position==len(ids[i])
                    audit(engine);cases.append(dict(case='dual_one_token_cap',depth=depth,pending_position_exact=True))
                for phase in ('waiting','prepared_prompt','prepared_decode'):
                    save(dict(depth=depth,cancel=phase))
                    with make() as engine:
                        a=engine.submit_tokens('reuse',ids[0],max_tokens=16);b=engine.submit_tokens('survivor',ids[1],max_tokens=16);out={a:[],b:[]};plan=None
                        if phase=='prepared_decode':
                            while not all(out.values()):
                                for ev in engine.step():out[ev.handle].extend(ev.token_ids)
                        prefix=list(out[a])
                        if phase!='waiting':plan=engine.bridge.prepare()
                        engine.cancel(a)
                        if plan is not None:
                            engine.bridge.execute_async(plan)
                            while not engine.bridge.poll(plan):time.sleep(.001)
                            for ev in engine.finish_plan(plan):out[ev.handle].extend(ev.token_ids)
                        for ev in engine.drain():out[ev.handle].extend(ev.token_ids)
                        assert out[a]==prefix and out[b]==gold[1];audit(engine)
                        c=engine.submit_tokens('reuse',ids[0],max_tokens=4);assert c.generation>a.generation
                        assert [t for ev in engine.drain() for t in ev.token_ids]==gold[0][:4];audit(engine)
                        cases.append(dict(case='cancel_survivor_reuse',depth=depth,phase=phase,exact=True))
                if not depth:continue
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                affinity_before=os.sched_getaffinity(0) if arguments.get('performance_cpu') else None
                @contextmanager
                def context():
                    if arguments.get('performance_cpu'):
                        cpu=importlib.import_module(tag+'.worker_cpu')
                        with cpu.worker_cpu('performance'),torch.inference_mode(),torch.cuda.stream(stream):yield
                    else:
                        with torch.inference_mode(),torch.cuda.stream(stream):yield
                def start():
                    nonlocal worker
                    worker=wm.AsyncWorker(make,context_factory=context).start();native.bind_worker(worker.thread);return worker.engine.bridge
                def stop(expected_failure=False):
                    nonlocal worker
                    try:worker.close(60)
                    except wm.WorkerFailed:
                        if not expected_failure:raise
                    worker=None
                save('async in-flight cancel with survivor');bridge=start();launched=Event();allow=Event();poll=bridge.poll
                def gated(plan):
                    if hasattr(plan,'execution') and len(plan.rows)==2:launched.set()
                    if launched.is_set() and not allow.is_set():return False
                    return poll(plan)
                bridge.poll=gated
                # Queue the survivor during the first request's prompt work.
                a=worker.submit_tokens('reuse',ids[1],max_tokens=16)
                b=worker.submit_tokens('survivor',ids[1],max_tokens=16)
                try:
                    assert launched.wait(60),'no two-request asynchronous ticket'
                    assert target.inflight is not None and len(target.inflight.windows)==2
                    assert a.cancel();a.close()
                    with ThreadPoolExecutor(1) as pool:
                        pending=pool.submit(worker.submit_tokens,'reuse',ids[0],max_tokens=4)
                        time.sleep(.02);assert not pending.done()
                        allow.set();c=pending.result(60)
                    assert [t for ev in b for t in ev.token_ids]==gold[1]
                    assert [t for ev in c for t in ev.token_ids]==gold[0][:4]
                    assert c.handle.generation>a.handle.generation and not worker.cancel(a.handle)
                    worker.wait_idle(60);audit(worker.engine)
                    cases.append(dict(case='async_inflight_cancel_survivor',two_gpu_windows=True,survivor_exact=True,reuse_exact=True))
                finally:allow.set();stop()
                save('bounded close and native owner guard');bridge=start();launched=Event();allow=Event();poll=bridge.poll
                def gated_close(plan):
                    if hasattr(plan,'execution'):launched.set()
                    if launched.is_set() and not allow.is_set():return False
                    return poll(plan)
                bridge.poll=gated_close;a=worker.submit_tokens('closing',ids[1],max_tokens=16)
                try:
                    assert launched.wait(60)
                    try:worker.close(.01)
                    except TimeoutError:pass
                    else:raise AssertionError('unfinished GPU ownership was closed')
                    try:native.close()
                    except RuntimeError as exc:assert 'live worker' in str(exc)
                    else:raise AssertionError('native group destroyed under GPU owner')
                    assert any(target.leased) and target.inflight is not None
                    allow.set();worker.close(60);assert list(a)[-1].finish_reason=='cancelled'
                    audit(worker.engine);cases.append(dict(case='bounded_close',state_retained=True,native_guard=True,final_fence=True))
                finally:allow.set();stop()
                save('injected worker failure fences real GPU work');bridge=start();poll=bridge.poll
                def failing(plan):
                    if hasattr(plan,'execution'):raise RuntimeError('injected K4 owner failure')
                    return poll(plan)
                bridge.poll=failing;a=worker.submit_tokens('fault',ids[1],max_tokens=16)
                try:
                    while True:
                        try:a.receive(60)
                        except wm.WorkerFailed as exc:assert 'injected K4 owner failure' in str(exc);break
                    deadline=time.monotonic()+60
                    while worker.thread.is_alive():
                        assert time.monotonic()<deadline;time.sleep(.01)
                    assert worker.snapshot()['status']=='failed' and worker.engine.closed
                    assert not any(target.leased) and target.inflight is None
                    cases.append(dict(case='owner_failure',quarantined=True,real_gpu_fenced=True,leases_released=True))
                finally:stop(True)
                bridge=start()
                try:
                    a=worker.submit_tokens('after_fault',ids[0],max_tokens=4)
                    assert [t for ev in a for t in ev.token_ids]==gold[0][:4];worker.wait_idle(60);audit(worker.engine)
                    cases.append(dict(case='new_owner_after_fault',exact=True))
                finally:stop()
                if affinity_before is not None:assert os.sched_getaffinity(0)==affinity_before
        assert env=={k:os.environ.get(k) for k in env} and not torch.distributed.is_initialized()
        assert (len(target.graphs)<=target.max_graphs if arguments.get('target_state_library') else len(target.graphs)==80) and not target.fallbacks
        save('PASS')
        return dict(status='PASS_K4_BATCH_GPU_LIFECYCLE',cases=cases,capabilities=target.capabilities,joint_budget_enabled=joint_budget is not None,
            source_sha256=manifest,gloo_restored=True,environment_restored=True,model_reloaded=False)
    finally:
        if worker is not None:
            try:worker.close(60)
            except wm.WorkerFailed:pass
        for provider in owned_providers:provider.close()
        if target is not None:
            session.torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for st,leased in zip(target.states,target.leased):
                if leased:target.release_state(st)
            target.close()
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.'):sys.modules.pop(n,None)
