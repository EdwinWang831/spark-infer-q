"""Matched4K native requests, bounded suffix reference and ownership checks."""
import gc
import importlib
import json
import time


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e45_run.py');smoke=load('scripts/e41_run.py')
    pilot_path=session.root/'jobs/done'/args['study_job'];pilot_job=json.loads(pilot_path.read_text())
    assert pilot_job['status']=='PASS' and pilot_job['result']['status']=='PASS_E45_PILOT'
    pilot=pilot_job['result'];assert pilot['target_object_id']==id(session.weights) and pilot['draft_object_id']==id(session.draft)
    for name,digest in pilot['source_sha256'].items():
        if name.startswith('spark_infer_q/') and name!='spark_infer_q/deepinfer_bridge.py':
            assert helper.sha(source/name)==digest,name
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    long=importlib.import_module(tag+'.long_target');optimized=importlib.import_module(tag+'.kaifra_optimized')
    native_mod=importlib.import_module(tag+'.native_scheduler');runtime=importlib.import_module(tag+'.kaifra_runtime')
    spec=importlib.import_module(tag+'.kaifra_speculative')
    expert,digest=helper.expert(session,tag)
    executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=4608,expert_backend=expert),chunk_rows=1024)
    report.update(status='RUNNING_E45_STUDY',pilot_job_sha256=helper.sha(pilot_path),expert_binary_sha256=digest,
        requests=[],oracles=[],lifecycle=[],http=[],dual=[],token_budget=1024,output_cap=64)
    def save():helper.save(source,report)
    cases=smoke.inputs(session)[1][:2]+[c for c in helper.make_inputs(session) if c['name'] in ('mixed3073','code4096')]
    target=None
    def capture_generate(engine,ids,mode,rep,target):
        captured=[];release=engine.bridge.backend.release
        def record(handle,state):
            ctx=state.context
            captured.append(dict(context=ctx,target=state.target,start=state.context_start,
                rows=len(ctx),end=state.target.pos,storage_bytes=ctx.untyped_storage().nbytes()))
            release(handle,state)
        with engine:
            engine.bridge.backend.release=record
            counts=(target.captures,target.replays,target.head_fallbacks);kernels=dict(target.screen.kernel_sha256)
            try:
                torch.cuda.synchronize();start=time.perf_counter();engine.submit_tokens('e45',ids,max_tokens=64)
                events=[];trace=[]
                while engine.active_requests:
                    events.extend(engine.step());trace.extend({k:m[k] for k in ('prompt','context_rows','verified_rows','emitted','accepted','observed','rates','head_fallback','head_union_count')} for m in engine.last_step_metrics)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            finally:engine.bridge.backend.release=release
            assert len(captured)==1 and not target.leased and not engine.bridge.backend.states
            tokens=[t for e in events for t in e.token_ids];ttft=events[0].ttft_seconds*1000
            record=dict(mode=mode,repetition=rep,warmup=rep<0,request_ms=elapsed,ttft_ms=ttft,decode_ms=elapsed-ttft,
                completion_tokens=len(tokens),output_sha256=smoke.token_sha(tokens),positions=[e.processed_position for e in events],trace=trace,
                captures_before=counts[0],captures_after=target.captures,graph_replays=target.replays-counts[1],
                head_fallbacks=target.head_fallbacks-counts[2],graph_fallbacks=dict(target.fallbacks),kernels_before=kernels,kernels_after=dict(target.screen.kernel_sha256),
                context_start=captured[0]['start'],retained_rows=captured[0]['rows'],context_end=captured[0]['end'],feature_storage_bytes=captured[0]['storage_bytes'])
            return record,captured[0],tokens
    try:
        with torch.inference_mode():
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=0) as native:
                report['scheduler']=load('scripts/e45_scheduler.py').check(native,importlib.import_module(tag+'.deepinfer_bridge').SchedulerBridge)
                report['pilot_bridge_sha256']=pilot['source_sha256']['spark_infer_q/deepinfer_bridge.py'];save()
            eager=long.LongTarget(session.weights,capacity=4608);oracle={}
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=0) as native:
                for case in cases:
                    report['active']='oracle:'+case['name'];save();captured=[]
                    with runtime.KaifraEngine.from_native(native,eager,eos_ids=session.eos,capacity=4352,max_requests=1,token_budget=1024,prefill_executor=executor) as engine:
                        old=engine.bridge.backend.release
                        def release(handle,state):captured.append(state);old(handle,state)
                        engine.bridge.backend.release=release
                        try:
                            engine.submit_tokens('oracle',case['ids'],max_tokens=64)
                            events=list(engine.drain(max_steps=(len(case['ids'])+1023)//1024+64+4))
                        finally:engine.bridge.backend.release=old
                    tokens=[t for e in events for t in e.token_ids];assert len(captured)==1
                    oracle[case['name']]=(tokens,captured[0]);report['oracles'].append(dict(name=case['name'],prompt_tokens=case['length'],output_tokens=len(tokens),output_sha256=smoke.token_sha(tokens),position=captured[0].pos));del captured
                    save()
            target=optimized.optimized_target(session,capacity=4608);target.warmup()
            bounded,metadata=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=4608)
            class FullHistorySuffix(spec.DFlashProvider):
                def __init__(self,base):
                    super().__init__(base.worker,model=base.model,feature_group=base.feature_group,budget=base.budget,
                                     history_window_rows=2040,max_request_context=4608)
                    self.history_window_rows=None;self.deferred_draft=True
                def propose(self,handle,state,anchor,remaining):
                    full,start=state.context,state.context_start
                    state.context=full[-2040:];state.context_start=start+len(full)-len(state.context)
                    try:return super().propose(handle,state,anchor,remaining)
                    finally:state.context=full;state.context_start=start
            reference=FullHistorySuffix(bounded);providers=dict(full=reference,bounded=bounded)
            report['profile']=metadata
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                def make(mode):
                    return spec.speculative_engine(native,target,providers['bounded' if mode=='owned' else mode],eos_ids=session.eos,
                        prior=session.calibration['conditional_prior'],mode='adaptive',capacity=4352,max_requests=1,token_budget=1024,
                        tokenizer=session.tokenizer,prefill_executor=executor)
                for case in cases:
                    canonical=None;gold,oracle_state=oracle[case['name']]
                    report['active']='requests:'+case['name'];save()
                    for rep in range(-1,4):
                        for mode in (('full','bounded') if rep<0 or rep%2==0 else ('bounded','full')):
                            rec,data,tokens=capture_generate(make(mode),case['ids'],mode,rep,target)
                            assert tokens==gold and not target.state_difference(oracle_state,data['target']),'target-only oracle differs'
                            if canonical is None:
                                assert mode=='full';canonical=(data['context'],rec['positions'],rec['trace'])
                            context,positions,trace=canonical
                            assert torch.equal(data['context'],context[-data['rows']:]) and rec['positions']==positions and rec['trace']==trace
                            assert data['start']+data['rows']==data['end']==oracle_state.pos
                            if mode=='bounded':assert data['rows']<=2040 and data['storage_bytes']==104448000
                            else:assert data['start']==0 and data['rows']==data['end']
                            rec.update(name=case['name'],prompt_tokens=case['length'],token_sha256=case['token_sha256'],
                                outputs_exact=True,features_suffix_exact=True,state_exact=True,trace_exact=True,oracle_exact=True)
                            if rep>=0:assert rec['captures_before']==rec['captures_after']==16 and rec['kernels_before']==rec['kernels_after'] and not rec['graph_fallbacks'] and rec['head_fallbacks']==0
                            report['requests'].append(rec);save()
                    del canonical,context,data;gc.collect()
                smoke.lifecycle(session,source,report,target,make,next(c for c in cases if c['name']=='code4096'))
                # Actual long HTTP message, tokenized identically for its token API oracle.
                messages=[dict(role='user',content=('缓存保存重复计算结果，队列保存等待处理的任务。 '*180)+'请用两句话总结两者的区别。')]
                ids=session.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
                assert 2048<len(ids) and len(ids)+64<=4352,len(ids)
                _,data,gold=capture_generate(make('bounded'),ids,'bounded',-1,target);del data
                smoke.http_checks(session,source,report,target,make,dict(messages=messages),gold,tag)
                report['http_prompt_tokens']=len(ids)
            report['optimized_graph']=target.capabilities;target.close();target=None
            # Two admitted target-only states, serialized device execution (not GPU batching).
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=0) as native:
                long_cases=[c for c in cases if c['length']>2048];captured={}
                with runtime.KaifraEngine.from_native(native,eager,eos_ids=session.eos,capacity=4352,max_requests=2,token_budget=1024,prefill_executor=executor) as engine:
                    old=engine.bridge.backend.release
                    def release(handle,state):captured[handle.request_id]=state;old(handle,state)
                    engine.bridge.backend.release=release
                    try:
                        for case in long_cases:engine.submit_tokens(case['name'],case['ids'],max_tokens=64)
                        assert engine.active_requests==2
                        events=list(engine.drain(max_steps=sum((len(c['ids'])+1023)//1024+64+4 for c in long_cases)))
                    finally:engine.bridge.backend.release=old
                for case in long_cases:
                    tokens=[t for e in events if e.handle.request_id==case['name'] for t in e.token_ids]
                    gold,state=oracle[case['name']]
                    assert tokens==gold and not eager.state_difference(state,captured[case['name']])
                    report['dual'].append(dict(name=case['name'],prompt_tokens=case['length'],output_tokens=len(tokens),exact=True))
                report['dual_scope']='two_admitted_target_only; serialized_device_execution; not_batched_DFlash_or_throughput'
            report['status']='PASS_E45_STUDY';report.pop('active',None);save()
    finally:
        executor.close()
        if target is not None:
            if target.leased:target.release_state(target.workspace)
            target.close()
        del target,executor
        gc.collect();torch.cuda.synchronize()
    return report
