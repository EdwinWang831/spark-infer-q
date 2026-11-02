"""Uninterrupted long-output target-only versus adaptive DFlash, no fixed arm."""
import gc
import hashlib
import importlib
import json
import time
from collections import Counter

PROMPT=261120
CAP=512


def token_sha(ids):
    return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()


def inputs(session):
    paragraph=('A processing ledger describes caches, queues, and transactions. A cache stores reusable computed results; '
        'a queue records work waiting for execution. Each transaction preserves its own committed state. '
        '缓存保存可复用结果，队列记录待处理任务；事务提交后的状态与未提交的临时计算必须区分。 '
        'The examples below are background material; registry keys are independent facts.\n')
    text=paragraph*6500;a=len(text)//3
    body=('Registry ORION-257 has the exact key MAPLE-7319.\n'+text[:a]+
        '\nRegistry TAU-631 has the exact key RIVER-4826.\n'+text[a:]+
        '\nRegistry LYRA-492 has the exact key CEDAR-9053.\n'
        'Write an extensive English engineering tutorial (at least 1000 words) explaining how to build a '
        'concurrent inference service. Start by listing the three exact registry keys in ORION-257, TAU-631, '
        'LYRA-492 order. Then explain chunked prefill, scheduling fairness, KV-cache ownership, speculative '
        'verification and rejection, cancellation and reuse, and latency measurement. Include a Python '
        'pseudocode example and concrete invariants. Use multiple detailed sections and do not stop after '
        'the keys or a short summary. This tutorial, rather than the repetitive ledger, is the requested answer.')
    raw=session.tokenizer.apply_chat_template([dict(role='user',content=body)],tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_dict=False)
    assert len(raw)>PROMPT
    ids=list(raw[:PROMPT-1024])+list(raw[-1024:]);assert len(ids)==PROMPT
    text=session.tokenizer.decode(ids,skip_special_tokens=True)
    keys=['MAPLE-7319','RIVER-4826','CEDAR-9053'];assert all(k in text for k in keys)
    return ids,dict(prompt_tokens=len(ids),source_tokens=len(raw),input_sha256=token_sha(ids),retrieval_keys=keys,
        construction='bilingual synthetic ledger plus English tutorial task; real chat prefix and final1024 tokens')


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e49_run.py');old=load('scripts/e46_study.py')
    fingerprints=load('scripts/e48_study.py');checks=load('scripts/e47_state.py')
    native_mod=importlib.import_module(tag+'.native_scheduler');shared=importlib.import_module(tag+'.shared_graph')
    optimized=importlib.import_module(tag+'.kaifra_optimized');spec=importlib.import_module(tag+'.kaifra_speculative')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    p=session.root/'jobs/done'/args['study_job'];job=json.loads(p.read_text());capacity=job['result']
    assert job['status']=='PASS' and capacity['status'] in ('STOP_E49_FOUR_CAPACITY','PASS_E49_CAPACITY_LOWER_BOUND')
    assert capacity['target_object_id']==id(session.weights) and capacity['draft_object_id']==id(session.draft)
    ids,identity=inputs(session)
    report.update(status='RUNNING_E49_STUDY',capacity_job_sha256=helper.sha(p),input=identity,
        output_cap=CAP,formats=[],scope='single_request_long_output; no_fixed_arm; no_HTTP_or_four_request_execution',
        quality_gate_promoted=False,formal_timing='submit_through_drain_and_sync; preparation_and_state_hash_outside')
    def save():helper.save(source,report)
    save();expert,digest=helper.expert(session,tag);report['expert_binary_sha256']=digest
    for fmt in ('bf16','int8'):
        target=executor=provider=None
        records=[];format_report=dict(kv_dtype=fmt,warmups=[],runs=records);report['formats'].append(format_report)
        try:
            with torch.inference_mode():
                report['active']=fmt+':prepare';save();old.guard(torch)
                target=shared.SharedGraphTarget.from_session(session,kv_dtype=fmt,slots=1)
                target.warmup();assert target.capabilities['graph_captures']==16
                for rows in range(1,9):
                    x=torch.zeros((rows,target.screen.k),dtype=torch.bfloat16,device=target.screen.weight.device)
                    for count in range(1,129):target.screen.subset(x,torch.arange(count,device=x.device))
                torch.cuda.synchronize();del x
                format_report['head_subset_prewarm_rows']=list(range(1,9))
                format_report['head_subset_prewarm_counts']=list(range(1,129))
                provider,profile=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=262160)
                format_report['profile']=profile
                executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,
                    expert_backend=expert,kv_dtype=fmt),chunk_rows=1024)
                def run(mode,prompt,cap,warmup=False):
                    outputs=[];events=[];metrics=[];plans=[];done={};progress=[]
                    with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7 if mode=='adaptive' else 0) as native:
                        if mode=='adaptive':
                            engine=spec.speculative_engine(native,target,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],
                                mode='adaptive',capacity=262144,max_requests=1,token_budget=1024,tokenizer=session.tokenizer,prefill_executor=executor)
                        else:
                            engine=shared.shared_graph_engine(native,target,eos_ids=session.eos,capacity=262144,token_budget=1024,
                                tokenizer=session.tokenizer,prefill_executor=executor)
                        with engine:
                            backend=engine.bridge.backend;release=backend.release;complete=engine.bridge.complete
                            def retain(h,st):
                                done[h.request_id]=st.target if mode=='adaptive' else st
                                release(h,st)
                            def observe(plan):
                                value=complete(plan)
                                metrics.extend(getattr(plan,'metrics',()))
                                plans.extend(dict(position=r.position,rows=len(r.token_ids),prompt=r.prompt,
                                    keep=(plan.results[i].keep if mode=='adaptive' and plan.results[i] is not None else len(r.token_ids)))
                                    for i,r in enumerate(plan.rows))
                                return value
                            backend.release=retain;engine.bridge.complete=observe
                            before=target.capabilities;hashes=dict(target.screen.kernel_sha256)
                            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                            try:
                                engine.submit_tokens('e49',prompt,max_tokens=cap)
                                for step in range(1200):
                                    if not engine.active_requests:break
                                    for e in engine.step():
                                        outputs.extend(e.token_ids)
                                        events.append(dict(emitted=len(e.token_ids),position=e.processed_position,finish=e.finish_reason,
                                            elapsed_seconds=e.elapsed_seconds,ttft_seconds=e.ttft_seconds))
                                    mem=old.guard(torch)
                                    if not warmup:
                                        progress.append(mem)
                                        if step%32==0 or (events and step%64==0):
                                            report['progress']=dict(kv_dtype=fmt,mode=mode,step=step,output_tokens=len(outputs),**mem);save()
                                torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
                                assert not engine.active_requests and done and events[-1]['finish'] in ('length','stop')
                            finally:
                                backend.release=release;engine.bridge.complete=complete
                            unchanged=hashes==target.screen.kernel_sha256
                            after=target.capabilities
                            nonempty=[e for e in events if e['emitted']];dt=nonempty[-1]['elapsed_seconds']-nonempty[0]['elapsed_seconds']
                            text=session.tokenizer.decode(outputs,skip_special_tokens=True)
                            decode=[m for m in metrics if not m['prompt']]
                            record=dict(mode=mode,warmup=warmup,prompt_tokens=len(prompt),output_cap=cap,output_tokens=len(outputs),
                                input_sha256=token_sha(prompt),output_sha256=token_sha(outputs),output_ids=outputs,generated_text=text,
                                ttft_ms=nonempty[0]['ttft_seconds']*1000,completion_ms=nonempty[-1]['elapsed_seconds']*1000,request_ms=wall,
                                residual_decode_tokens=len(outputs)-nonempty[0]['emitted'],residual_decode_ms=dt*1000,
                                decode_tok_s=(len(outputs)-nonempty[0]['emitted'])/dt if dt>0 else None,
                                finish_reason=events[-1]['finish'],events=events,plans=plans,metrics=metrics,
                                accepted=sum(m['accepted'] for m in decode),observed=sum(m['observed'] for m in decode),
                                verify_rounds=len(decode) if mode=='adaptive' else sum(not r['prompt'] for r in plans),
                                width_histogram=dict(Counter(m['verified_rows'] for m in decode)),
                                draft_ms=sum(m['draft_ms'] for m in decode),verify_ms=sum(m['verify_ms'] for m in decode),commit_ms=sum(m['commit_ms'] for m in decode),
                                graph_captures=after['graph_captures'],graph_replays=after['graph_replays']-before['graph_replays'],
                                graph_fallbacks=after['graph_fallbacks'],head_fallbacks=after['head_fallbacks'],timed_kernel_sources_unchanged=unchanged,
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),minimum_host_available_bytes=min((m['MemAvailable'] for m in progress),default=None))
                            # Large cache transfer is after completion/sync timers.
                            st=done['e49'];record['state_sha256']=fingerprints.state_sha(torch,checks,st)
                            record['final_position']=st.pos
                            record['retrieval_hits']=[k in text for k in identity['retrieval_keys']]
                            report['last_run']=record;save()
                            assert not any(after['graph_fallbacks']) and not after['head_fallbacks']
                            if not warmup:assert unchanged,'formal head JIT changed'
                            return record
                short=session.tokenizer.apply_chat_template([dict(role='user',content='Explain cache ownership in a concurrent inference service, with a small Python example.')],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False,return_dict=False)
                for mode in ('target_only','adaptive'):
                    format_report['warmups'].append(run(mode,list(short),32,True));save()
                for mode in ('target_only','adaptive'):
                    report['active']=fmt+':'+mode;save()
                    records.append(run(mode,ids,CAP));save()
                a,b=records
                format_report['outputs_exact']=a['output_ids']==b['output_ids']
                format_report['committed_state_exact']=a['state_sha256']==b['state_sha256']
                assert format_report['outputs_exact'],'target-only/adaptive output mismatch'
                assert format_report['committed_state_exact'],'target-only/adaptive full state mismatch'
                format_report['decode_speedup']=b['decode_tok_s']/a['decode_tok_s']-1
                format_report['request_wall_change']=b['request_ms']/a['request_ms']-1
                save()
        finally:
            if executor is not None:executor.close()
            if target is not None:
                for owner in list(target.owners):
                    st=target.owners[owner].workspace
                    target.release_state(st)
                target.close()
            target=executor=provider=None
            gc.collect();torch.cuda.synchronize()
        format_report['after_format_allocated_bytes']=torch.cuda.memory_allocated();save()
    report['cross_format_outputs_exact']=report['formats'][0]['runs'][0]['output_ids']==report['formats'][1]['runs'][0]['output_ids']
    report['status']='PASS_E49_SINGLE_LONG_OUTPUT'
    return report
