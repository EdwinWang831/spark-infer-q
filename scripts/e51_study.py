"""Same-format full256K baseline/ring requests; no fixed drafts or cross-engine run."""
import gc
import hashlib
import importlib
import json
import time


def tensor_sha(torch,tensor):
    h=hashlib.sha256()
    for at in range(0,len(tensor),8192):
        h.update(tensor[at:at+8192].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def live_state(torch,state):
    """Compare complete committed KV/pools and only semantically live raw tail."""
    value={};end=state.pos;start=end//state.ratio*state.ratio
    for i,cache in enumerate(state.kc):
        for name in ('k','v'):
            value[str(i)+name]=tensor_sha(torch,getattr(cache,name)[:end])
        if cache.quantized:
            for name in ('ks','vs'):value[str(i)+name]=tensor_sha(torch,getattr(cache,name)[:end])
        value[str(i)+'pool']=tensor_sha(torch,state.pooled[i][:end//state.ratio])
        raw=state.ikc[i];offset=start%len(raw) if hasattr(state,'index_ring_rows') else start
        value[str(i)+'live_raw_tail']=tensor_sha(torch,raw[offset:offset+end-start])
    snap=state.snapshot()
    value.update(pos=end,mtp_len=snap['mtp_len'])
    for name in ('rec','conv','ple_tail'):value[name]=tensor_sha(torch,snap[name])
    value['ple_history']=hashlib.sha256(snap['ple_history'].tobytes()).hexdigest()
    return value


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e51_run.py');common=load('scripts/e50_run.py')
    tasks=load('scripts/e49_study.py');old=load('scripts/e46_study.py');layout=load('scripts/e50_layout.py')
    shared=importlib.import_module(tag+'.shared_graph');optimized=importlib.import_module(tag+'.kaifra_optimized')
    spec=importlib.import_module(tag+'.kaifra_speculative');native_mod=importlib.import_module(tag+'.native_scheduler')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    gate_path=session.root/'jobs/done'/args['study_job'];gate=json.loads(gate_path.read_text())
    assert gate['status']=='PASS' and gate['result']['status']=='PASS_E51_KERNEL'
    assert gate['result']['target_object_id']==id(session.weights)
    # All execution-library files must match the sealed GPU gate.
    for name,digest in gate['result']['source_sha256'].items():
        if name.startswith('spark_infer_q/'):
            assert helper.sha(source/name)==digest,name
    fmt=args['kv_dtype'];ids,identity=tasks.inputs(session)
    report.update(status='RUNNING_E51_STUDY',kv_dtype=fmt,input=identity,kernel_gate_job_sha256=helper.sha(gate_path),
        runs=[],warmups=[],lifecycle=[],scope='same-format baseline/ring single256K native adaptive; full KV/pools preserved')
    def save():helper.save(source,report)
    save();expert,digest=common.expert(session,tag);report['expert_binary_sha256']=digest
    target=executor=provider=None
    try:
        with torch.inference_mode():
            # All requests use the E50 query Graph and output cache, so only
            # the target raw-index layout/dispatch changes between arms.
            provider,_=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=262160,query_graphs=True)
            provider.worker.warmup()
            for mode in ('full','ring'):
                report['active']='prepare:'+mode;save();old.guard(torch)
                ring=2048 if mode=='ring' else None
                target=shared.SharedGraphTarget.from_session(session,kv_dtype=fmt,slots=1,index_ring_rows=ring)
                target.warmup()
                for rows in range(1,9):
                    x=torch.zeros((rows,target.screen.k),dtype=torch.bfloat16,device=target.screen.weight.device)
                    for count in range(1,129):target.screen.subset(x,torch.arange(count,device=x.device))
                del x
                executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,
                    expert_backend=expert,kv_dtype=fmt,index_ring_rows=ring),chunk_rows=1024)
                with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                    def make():return spec.speculative_engine(native,target,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],
                        mode='adaptive',capacity=262144,max_requests=1,token_budget=1024,tokenizer=session.tokenizer,
                        prefill_executor=executor,output_cache=True)
                    def run(prompt,cap,warmup=False):
                        output=[];events=[];metrics=[];done={};progress=[]
                        with make() as engine:
                            backend=engine.bridge.backend;original=backend.release
                            def retain(h,st):
                                done['state']=st.target;done['context']=st.context.clone();original(h,st)
                            backend.release=retain
                            caps0=target.capabilities;head=dict(target.screen.kernel_sha256)
                            draft0=provider.worker.graph_capabilities
                            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                            try:
                                engine.submit_tokens('e51',prompt,max_tokens=cap)
                                for step in range(1200):
                                    if not engine.active_requests:break
                                    out=engine.step();metrics.extend(engine.last_step_metrics)
                                    for e in out:
                                        output.extend(e.token_ids);events.append(dict(emitted=len(e.token_ids),ttft=e.ttft_seconds,elapsed=e.elapsed_seconds,finish=e.finish_reason))
                                    mem=old.guard(torch);progress.append(mem)
                                    if not warmup and step%32==0:
                                        report['progress']=dict(mode=mode,step=step,output_tokens=len(output),**mem);save()
                                torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
                                assert not engine.active_requests and output and done
                            finally:backend.release=original
                            cap1=target.capabilities;draft1=provider.worker.graph_capabilities
                            nonempty=[e for e in events if e['emitted']];decode=[m for m in metrics if not m['prompt']]
                            record=dict(mode=mode,warmup=warmup,prompt_tokens=len(prompt),output_ids=output,output_tokens=len(output),
                                output_sha256=tasks.token_sha(output),request_ms=wall,ttft_ms=nonempty[0]['ttft']*1000,
                                decode_ms=(nonempty[-1]['elapsed']-nonempty[0]['elapsed'])*1000,
                                decode_tokens=len(output)-nonempty[0]['emitted'],events=events,metrics=metrics,
                                verify_rounds=len(decode),accepted=sum(m['accepted'] for m in decode),observed=sum(m['observed'] for m in decode),
                                phase_ms={k:sum(m[k] for m in decode) for k in ('draft_ms','verify_ms','commit_ms')},
                                graph_replays=cap1['graph_replays']-caps0['graph_replays'],graph_captures=cap1['graph_captures'],
                                graph_fallbacks=cap1['graph_fallbacks'],head_fallbacks=cap1['head_fallbacks'],
                                kernel_hashes_unchanged=head==target.screen.kernel_sha256,
                                draft_captures_before=draft0['draft_graph_captures'],draft_captures_after=draft1['draft_graph_captures'],
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),minimum_host_available_bytes=min(m['MemAvailable'] for m in progress))
                            if record['decode_ms']>0:record['decode_tok_s']=record['decode_tokens']*1000/record['decode_ms']
                            state=done['state'];record['live_state']=live_state(torch,state)
                            record['feature_sha256']=tensor_sha(torch,done['context'])
                            record['state_storage_bytes']=layout.storage_bytes(torch,state)
                            # Scratch from the last attention layer/verify window.
                            last=decode[-1] if decode else None
                            if last:
                                b=target.buffers[last['verified_rows']];s=b.attn
                                record['last_selector_sha256']=tensor_sha(torch,s.ids[:last['verified_rows']])
                                record['last_selector_counts_sha256']=tensor_sha(torch,s.nk[:last['verified_rows']])
                            # Untimed full-logit/feature diagnostic at the final
                            # committed prefix, independently of screened IDs.
                            snap=state.snapshot()
                            try:
                                logits,features,_=target.eager.run(state,[output[-1]],capture=True)
                                record['diagnostic_logits_sha256']=tensor_sha(torch,logits)
                                record['diagnostic_features_sha256']=tensor_sha(torch,features)
                            finally:
                                state.restore(snap)
                            assert record['live_state']==live_state(torch,state),'diagnostic changed committed state'
                            record['retrieval_hits']=[k in session.tokenizer.decode(output,skip_special_tokens=True) for k in identity['retrieval_keys']]
                            report['last_run']=record;save()
                            assert cap1['graph_captures']==caps0['graph_captures']==16 and not any(cap1['graph_fallbacks']) and not cap1['head_fallbacks']
                            assert draft0['draft_graph_captures']==draft1['draft_graph_captures']==10
                            if not warmup:assert record['kernel_hashes_unchanged'],'timed head JIT'
                            return record
                    short=list(session.tokenizer.apply_chat_template([dict(role='user',content='Explain why a cache needs a clear owner and how cancellation affects it.')],
                        tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False))
                    report['active']='warmup:'+mode;save();report['warmups'].append(run(short,32,True));save()
                    if mode=='ring':
                        for phase in ('waiting','prepared_prompt','prepared_decode'):
                            with make() as engine:
                                h=engine.submit_tokens('reuse',short,max_tokens=4)
                                if phase=='prepared_decode':engine.step();assert engine.active_requests==1
                                p=None
                                if phase!='waiting':p=engine.bridge.prepare();assert p.rows;engine.bridge.execute(p)
                                try:engine.cancel(h)
                                finally:
                                    if p is not None:engine.bridge.complete(p)
                                out=list(engine.drain(max_steps=16));assert out[-1].finish_reason=='cancelled'
                                h2=engine.submit_tokens('reuse',short,max_tokens=1);out=list(engine.drain(max_steps=16))
                                assert h2.generation>h.generation and sum(len(e.token_ids) for e in out)==1
                            report['lifecycle'].append(dict(phase=phase,exact=True));save()
                    report['active']='long:'+mode;save();report['runs'].append(run(ids,512));save()
                executor.close();target.close();executor=target=None;gc.collect();torch.cuda.synchronize()
            a,b=report['runs']
            trace=lambda r:[{k:m[k] for k in ('prompt','verified_rows','emitted','accepted','observed','context_rows','rates','head_fallback','head_union_count')} for m in r['metrics']]
            assert a['output_ids']==b['output_ids'],'output mismatch'
            assert a['live_state']==b['live_state'],'complete live state mismatch'
            assert trace(a)==trace(b),'dynamic budget/acceptance trace mismatch'
            for key in ('feature_sha256','last_selector_sha256','last_selector_counts_sha256',
                        'diagnostic_logits_sha256','diagnostic_features_sha256'):
                assert a[key]==b[key],key
            report.update(status='PASS_E51_STUDY',outputs_live_state_trace_features_selectors_exact=True,
                decode_change=b['decode_tok_s']/a['decode_tok_s']-1,request_wall_change=b['request_ms']/a['request_ms']-1,
                state_storage_saved_bytes=a['state_storage_bytes']-b['state_storage_bytes'])
            save();return report
    finally:
        if executor is not None:executor.close()
        if target is not None:
            for owner in list(target.owners):target.release_state(target.owners[owner].workspace)
            target.close()
        if provider is not None:provider.worker.close()
        executor=target=provider=None;gc.collect();torch.cuda.synchronize()
