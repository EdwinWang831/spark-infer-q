"""Separate CPU dispatch and actual CUDA kernel work across Target-to-draft."""
import gc,hashlib,importlib,json,os,sys,time,gzip
from pathlib import Path
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_k4_dispatch_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    solo=target=None
    report=dict(status='RUNNING',source_sha256=manifest,model_reloaded=False,profiles=[],contract=dict(
        scope='attribution only, no throughput claim; actual CUDA kernels from Kineto trace partitioned by DFLASH CPU range',
        cpu='thread/process CPU and wall; placement/frequency belong to the separate core probe',
        modes='old=optimized single; new=batch with explicit regression-control 0.5 ms sleeps',
        fences='unchanged; every Target commit event retained'))
    def save(active=None):
        if active is not None:report['active']=active
        p=root/'dispatch-result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    try:
        torch=session.torch
        opt=importlib.import_module(tag+'.kaifra_optimized');batch=importlib.import_module(tag+'.kaifra_batch')
        spec=importlib.import_module(tag+'.kaifra_speculative');nm=importlib.import_module(tag+'.native_scheduler')
        tm=importlib.import_module(tag+'.kaifra_joint_target');hm=importlib.import_module(tag+'.screened_head_fused')
        from e3_wide_target import WideTarget
        provider,metadata=opt.optimized_provider(session);solo=opt.optimized_target(session);solo.warmup()
        target=tm.JointTarget(WideTarget(session.weights,capacity=2048),hm.FusedScreenedHead(session.weights.head));target.warmup()
        ids=session.tokenizer.apply_chat_template([dict(role='user',content='请用三点解释缓存与队列的区别。')],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        prior=session.calibration['conditional_prior'];gold=policy=None
        with nm.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
            def make(mode):
                if mode=='old':return spec.speculative_engine(native,solo,provider,eos_ids=session.eos,prior=prior,capacity=256,mode='adaptive')
                engine=batch.batch_engine(native,target,provider,eos_ids=session.eos,prior=prior,capacity=256)
                bridge=engine.bridge
                # Preserve the historical regression control after production
                # execute() has switched to current-event waits.
                def sleep_execute(plan):
                    bridge.execute_async(plan)
                    while not bridge.poll(plan):time.sleep(.0005)
                bridge.execute=sleep_execute
                return engine
            def generate(mode,instrument=False):
                nonlocal gold,policy
                rows=[];tokens=[];steps=[]
                with make(mode) as engine:
                    handle=engine.submit_tokens('probe',ids,max_tokens=32)
                    while not tokens:
                        for ev in engine.step():tokens.extend(ev.token_ids)
                    assert len(tokens)==1
                    source=engine.bridge.backend.provider;original=source.propose
                    if instrument:
                        def wrapped(handle,state,anchor,remaining):
                            w=time.perf_counter();c=time.thread_time();p=time.process_time()
                            with torch.profiler.record_function('DFLASH'):
                                result=original(handle,state,anchor,remaining)
                            rows.append(dict(context_rows=len(state.context),remaining=remaining,wall_ms=(time.perf_counter()-w)*1000,
                                thread_cpu_ms=(time.thread_time()-c)*1000,process_cpu_ms=(time.process_time()-p)*1000))
                            return result
                        source.propose=wrapped
                    try:
                        while engine.active_requests:
                            for ev in engine.step():tokens.extend(ev.token_ids)
                            steps.extend({k:m[k] for k in ('context_rows','verified_rows','emitted','accepted','observed','rates')}
                                for m in engine.last_step_metrics if not m['prompt'])
                    finally:
                        if instrument:source.__dict__.pop('propose',None)
                    assert not engine.bridge.backend.states and not engine.bridge.native.has_unfinished_seqs()
                if gold is None:gold=tokens;policy=steps
                else:assert tokens==gold and steps==policy
                return rows
            save('warmup correctness')
            generate('old');generate('new')
            activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
            for mode in ('old','new'):
                save('profile:'+mode)
                with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False) as prof:
                    rows=generate(mode,True)
                path=root/('dispatch-'+mode+'.json');prof.export_chrome_trace(str(path))
                trace=json.loads(path.read_text());events=trace['traceEvents']
                ranges=[e for e in events if e.get('name')=='DFLASH' and e.get('ph')=='X' and e.get('cat')=='user_annotation']
                kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
                assert len(ranges)==len(rows) and kernels,(len(ranges),len(rows),len(kernels))
                detail=[]
                for row,event in zip(rows,sorted(ranges,key=lambda e:e['ts'])):
                    a,b=event['ts'],event['ts']+event['dur']
                    included=[e for e in kernels if a<=e['ts'] and e['ts']+e['dur']<=b]
                    detail.append(dict(row,kernel_count=len(included),kernel_ms=sum(e['dur'] for e in included)/1000,
                        trace_range_ms=event['dur']/1000))
                # Full traces remain local raw evidence; public report has scalar
                # events only and never private paths, stacks or model values.
                with gzip.open(str(path)+'.gz','wb') as output:output.write(path.read_bytes())
                report['profiles'].append(dict(mode=mode,drafts=detail,trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    cuda_kernel_count=len(kernels),kernel_names=sorted({e['name'] for e in kernels if 'draft' in e['name'].lower()})))
                save()
        report.update(status='PASS_K4_DISPATCH_PROFILE',outputs_policy_exact=True,gloo_released=not torch.distributed.is_initialized())
        report.pop('active',None);save();return report
    finally:
        if target is not None:
            session.torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for st,leased in zip(target.states,target.leased):
                if leased:target.release_state(st)
            target.close()
        if solo is not None:
            if solo.leased:solo.release_state(solo.workspace)
            solo.close()
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.'):sys.modules.pop(n,None)
