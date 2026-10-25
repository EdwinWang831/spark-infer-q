"""Contemporary uninterrupted dynamic requests: original vs query/cache options."""
import gc
import importlib
import json
import time


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e50_run.py');tasks=load('scripts/e49_study.py');old=load('scripts/e46_study.py')
    fp=load('scripts/e48_study.py');checks=load('scripts/e47_state.py')
    shared=importlib.import_module(tag+'.shared_graph');optimized=importlib.import_module(tag+'.kaifra_optimized')
    spec=importlib.import_module(tag+'.kaifra_speculative');native_mod=importlib.import_module(tag+'.native_scheduler')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    pp=session.root/'jobs/done'/args['study_job'];pj=json.loads(pp.read_text());probe=pj['result']
    pilot_path=session.root/'jobs/done'/args['build_job'];pilot=json.loads(pilot_path.read_text())
    assert pj['status']=='PASS' and probe['status']=='PASS_E50_PROBE_GAIN' and probe['candidate_gate_passed']
    assert pilot['status']=='PASS' and pilot['result']['status']=='PASS_E50_PILOT' and len(pilot['result']['cases'])==12
    assert probe['target_object_id']==id(session.weights) and probe['draft_object_id']==id(session.draft)
    ids,identity=tasks.inputs(session)
    report.update(status='RUNNING_E50_STUDY',probe_job_sha256=helper.sha(pp),pilot_job_sha256=helper.sha(pilot_path),input=identity,
        runs=[],warmups=[],lifecycle=[],scope='single_request_BF16_native_token_API; baseline vs query-islands/native-output-cache; no fixed drafts or cross-engine run')
    def save():helper.save(source,report)
    save();target=executor=None;providers={}
    try:
        with torch.inference_mode():
            target=shared.SharedGraphTarget.from_session(session,kv_dtype='bf16',slots=1);target.warmup()
            for rows in range(1,9):
                x=torch.zeros((rows,target.screen.k),dtype=torch.bfloat16,device=target.screen.weight.device)
                for count in range(1,129):target.screen.subset(x,torch.arange(count,device=x.device))
            del x
            for mode in ('baseline','optimized'):
                providers[mode],_=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=262160,query_graphs=mode=='optimized')
            providers['optimized'].worker.warmup()
            expert,digest=helper.expert(session,tag);report['expert_binary_sha256']=digest
            executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,expert_backend=expert),chunk_rows=1024)
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                def make(mode):
                    return spec.speculative_engine(native,target,providers[mode],eos_ids=session.eos,prior=session.calibration['conditional_prior'],mode='adaptive',
                        capacity=262144,max_requests=1,token_budget=1024,tokenizer=session.tokenizer,prefill_executor=executor,output_cache=mode=='optimized')
                def run(mode,prompt,cap,warmup=False):
                    outputs=[];events=[];metrics=[];done={};progress=[];reads=dict(full=0,cached=0)
                    with make(mode) as engine:
                        backend=engine.bridge.backend;release=backend.release;read=engine.bridge.output_tokens
                        def retain(h,st):done[h.request_id]=st.target;release(h,st)
                        def observe(h):
                            reads['cached' if engine.bridge.output_cache else 'full']+=1
                            return read(h)
                        backend.release=retain;engine.bridge.output_tokens=observe
                        before=target.capabilities;hashes=dict(target.screen.kernel_sha256)
                        graph_before=providers['optimized'].worker.graph_capabilities
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                        try:
                            engine.submit_tokens('e50',prompt,max_tokens=cap)
                            for step in range(1200):
                                if not engine.active_requests:break
                                out=engine.step();metrics.extend(engine.last_step_metrics)
                                for e in out:
                                    outputs.extend(e.token_ids);events.append(dict(emitted=len(e.token_ids),position=e.processed_position,
                                        finish=e.finish_reason,ttft_seconds=e.ttft_seconds,elapsed_seconds=e.elapsed_seconds))
                                mem=old.guard(torch)
                                if not warmup:
                                    progress.append(mem)
                                    if step%32==0:report['progress']=dict(mode=mode,step=step,output_tokens=len(outputs),**mem);save()
                            torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
                            assert not engine.active_requests and done and outputs
                        finally:backend.release=release;engine.bridge.output_tokens=read
                        nonempty=[e for e in events if e['emitted']];decode=[m for m in metrics if not m['prompt']]
                        graph_after=providers['optimized'].worker.graph_capabilities;caps=target.capabilities
                        record=dict(mode=mode,warmup=warmup,prompt_tokens=len(prompt),output_cap=cap,output_tokens=len(outputs),output_ids=outputs,
                            input_sha256=tasks.token_sha(prompt),output_sha256=tasks.token_sha(outputs),generated_text=session.tokenizer.decode(outputs,skip_special_tokens=True),
                            request_ms=wall,ttft_ms=nonempty[0]['ttft_seconds']*1000,completion_ms=nonempty[-1]['elapsed_seconds']*1000,
                            decode_ms=(nonempty[-1]['elapsed_seconds']-nonempty[0]['elapsed_seconds'])*1000,decode_tokens=len(outputs)-nonempty[0]['emitted'],
                            finish_reason=events[-1]['finish'],events=events,metrics=metrics,verify_rounds=len(decode),
                            accepted=sum(m['accepted'] for m in decode),observed=sum(m['observed'] for m in decode),
                            phase_ms={k:sum(m[k] for m in decode) for k in ('draft_ms','verify_ms','commit_ms')},
                            kernel_hashes_unchanged=hashes==target.screen.kernel_sha256,graph_captures=caps['graph_captures'],
                            graph_replays=caps['graph_replays']-before['graph_replays'],graph_fallbacks=caps['graph_fallbacks'],head_fallbacks=caps['head_fallbacks'],
                            draft_graph_before=graph_before,draft_graph_after=graph_after,native_output_reads=reads,
                            native_terminal_history_checked=mode=='optimized',peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            minimum_host_available_bytes=min((m['MemAvailable'] for m in progress),default=None))
                        record['decode_tok_s']=record['decode_tokens']*1000/record['decode_ms']
                        st=done['e50'];record['state_sha256']=fp.state_sha(torch,checks,st);record['final_position']=st.pos
                        record['retrieval_hits']=[k in record['generated_text'] for k in identity['retrieval_keys']]
                        report['last_run']=record;save()
                        if not warmup:assert record['kernel_hashes_unchanged'],'formal head JIT changed'
                        assert caps['graph_captures']==16 and not any(caps['graph_fallbacks']) and not caps['head_fallbacks']
                        assert graph_before['draft_graph_captures']==graph_after['draft_graph_captures']==10
                        if mode=='optimized':assert reads['full']==0 and reads['cached']>0
                        return record
                short=session.tokenizer.apply_chat_template([dict(role='user',content='Explain why a cache needs a clear owner and how cancellation affects it.')],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False,return_dict=False)
                for mode in ('baseline','optimized'):
                    report['active']='warmup:'+mode;save();report['warmups'].append(run(mode,list(short),32,True));save()
                    for phase in ('waiting','prepared_prompt','prepared_decode'):
                        with make(mode) as engine:
                            h=engine.submit_tokens('reuse',short,max_tokens=4)
                            if phase=='prepared_decode':engine.step();assert engine.active_requests==1
                            p=None
                            if phase!='waiting':p=engine.bridge.prepare();assert p.rows;engine.bridge.execute(p)
                            try:engine.cancel(h)
                            finally:
                                if p is not None:engine.bridge.complete(p)
                            end=list(engine.drain(max_steps=16));assert end[-1].finish_reason=='cancelled'
                            h2=engine.submit_tokens('reuse',short,max_tokens=1);out=list(engine.drain(max_steps=16))
                            assert h2.generation>h.generation and sum(len(e.token_ids) for e in out)==1
                        report['lifecycle'].append(dict(mode=mode,phase=phase,exact=True));save()
                for mode in ('baseline','optimized'):
                    report['active']='long:'+mode;save();report['runs'].append(run(mode,ids,512));save()
            a,b=report['runs']
            trace=lambda r:[{k:m[k] for k in ('prompt','verified_rows','emitted','accepted','observed','context_rows','rates','head_fallback','head_union_count')} for m in r['metrics']]
            assert a['output_ids']==b['output_ids']==probe['checks'][0]['output_ids'] and a['state_sha256']==b['state_sha256'],'full request output/state mismatch'
            assert trace(a)==trace(b),'full request budget/acceptance trace mismatch'
            report.update(outputs_state_trace_exact=True,decode_gain=b['decode_tok_s']/a['decode_tok_s']-1,request_wall_change=b['request_ms']/a['request_ms']-1,
                full_request_gate_passed=b['decode_tok_s']>=1.01*a['decode_tok_s'] and b['request_ms']<=1.05*a['request_ms'])
            report['status']='PASS_E50_STUDY_GAIN' if report['full_request_gate_passed'] else 'STOP_E50_STUDY_NO_GAIN';save()
    finally:
        for p in providers.values():
            close=getattr(p.worker,'close',None)
            if close:close()
        if executor is not None:executor.close()
        if target is not None:
            for owner in list(target.owners):target.release_state(target.owners[owner].workspace)
            target.close()
        providers.clear();target=executor=None;gc.collect();torch.cuda.synchronize()
    return report
