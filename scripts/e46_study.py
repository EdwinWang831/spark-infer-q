"""One direct256K native request, with an explicitly untimed diagnostic pause."""
import gc
import hashlib
import importlib
import json
from pathlib import Path
import time

PROMPT=262080
OUTPUT=64
CAPACITY=262160
KEYS=('MAPLE-7319','RIVER-4826','CEDAR-9053')


def memory(torch):
    values={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapFree:'))}
    values.update(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
    return values


def guard(torch):
    m=memory(torch)
    if m['MemAvailable']<8*1024**3:raise RuntimeError('host available memory below8GiB reserve: '+str(m))
    return m


def inputs(session):
    paragraph=('A processing ledger describes caches, queues, and transactions. A cache stores reusable computed results; '
               'a queue records work waiting for execution. Each transaction preserves its own committed state. '
               '缓存保存可复用结果，队列记录待处理任务；事务提交后的状态与未提交的临时计算必须区分。 '
               'The operational examples below are background material, and the three registry keys are separate facts.\n')
    text=paragraph*6000
    a=len(text)//3;b=2*len(text)//3
    text=('Registry ORION-257 has the exact key '+KEYS[0]+'.\n'+text[:a]+
          '\nRegistry TAU-631 has the exact key '+KEYS[1]+'.\n'+text[a:b]+text[b:]+
          '\nRegistry LYRA-492 has the exact key '+KEYS[2]+'.\n'
          'Return the three exact registry keys, in ORION-257, TAU-631, LYRA-492 order. Output only the keys, without explanation.')
    raw=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_dict=False)
    assert len(raw)>PROMPT
    # Keep the final256 tokens (late fact, question and actual chat suffix).
    ids=list(raw[:PROMPT-256])+list(raw[-256:]);assert len(ids)==PROMPT
    decoded=session.tokenizer.decode(ids,skip_special_tokens=True)
    assert all(k in decoded for k in KEYS),'retrieval facts were truncated'
    return ids,dict(prompt_tokens=len(ids),source_template_tokens=len(raw),tail_tokens_kept=256,
        token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
        construction='synthetic bilingual ledger; prefix plus final256 actual chat tokens',
        key_character_offsets=[decoded.index(k) for k in KEYS])


def prefix_digest(torch,state,end):
    digest=hashlib.sha256()
    for i,cache in enumerate(state.kc):
        for name,t,rows in (('k',cache.k,end),('v',cache.v,end),('index',state.ikc[i],end),('pool',state.pooled[i],end//state.ratio)):
            digest.update((str(i)+name+str(rows)).encode())
            for at in range(0,rows,8192):
                digest.update(t[at:min(rows,at+8192)].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def tail(state,start):
    result=[]
    for i,k in enumerate(state.kc):
        result.extend((k.k[start:state.pos].clone(),k.v[start:state.pos].clone(),
                       state.ikc[i][start:state.pos].clone(),state.pooled[i][start//state.ratio:state.pos//state.ratio].clone()))
    return result


def exact_state(torch,state,snapshot,cache_tail,start):
    import numpy as np
    got=state.snapshot()
    assert got['pos']==snapshot['pos'] and got['mtp_len']==snapshot['mtp_len']
    assert np.array_equal(got['ple_history'],snapshot['ple_history'])
    assert all(torch.equal(got[k],snapshot[k]) for k in ('rec','conv','ple_tail'))
    other=tail(state,start)
    assert len(other)==len(cache_tail) and all(torch.equal(a,b) for a,b in zip(other,cache_tail))


def selector_check(torch,workspace,capacity):
    import numpy as np
    s=workspace.attn;pos=int(workspace.pos_blk.item());count=min(256,capacity-pos)
    # Last scheduler chunk is960 rows, hence its final attention tile is192.
    count=PROMPT-pos
    assert count==192 and s.qsa and s.nb>32768
    records=[]
    for r in (0,count//2,count-1):
        end=pos+r+1;complete=end//s.ratio
        scores=s.scores[r,:complete].cpu().numpy();assert np.isfinite(scores).all()
        chosen=np.sort(np.lexsort((np.arange(complete),-scores))[:s.budget//s.ratio])
        ids=np.concatenate([(chosen[:,None]*s.ratio+np.arange(s.ratio)).reshape(-1),np.arange(complete*s.ratio,end)])
        n=int(s.nk[r].item());actual=s.ids[r,:n].cpu().numpy()
        assert int(s.sparse[r].item())==1 and np.array_equal(actual,ids),'tiled QSA selector differs from CPU stable topK'
        records.append(dict(position=end,complete_blocks=complete,selected_keys=n,exact=True,
            minimum_key=int(actual.min()),maximum_key=int(actual.max()),selected_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
    return records


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e46_run.py')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    optimized=importlib.import_module(tag+'.kaifra_optimized');native_mod=importlib.import_module(tag+'.native_scheduler')
    spec=importlib.import_module(tag+'.kaifra_speculative')
    ids,identity=inputs(session)
    report.update(status='RUNNING_E46_STUDY',input=identity,prompt_tokens=PROMPT,output_cap=OUTPUT,
        chunks=[],windows=[],selector=[],timing_scope='capacity/correctness study; diagnostics pause after first token; active phase timers only',
        checkpoint_position_limit=262144,paged_kv=False,kv_dtype='bf16',multi_request=False)
    def save():helper.save(source,report)
    save();expert,binary_sha=helper.expert(session,tag);report['expert_binary_sha256']=binary_sha
    target=executor=None
    try:
        with torch.inference_mode():
            guard(torch);torch.cuda.reset_peak_memory_stats()
            report['active']='prepare256K';save()
            target=optimized.optimized_target(session,capacity=CAPACITY)
            target.warmup();assert target.captures==16
            report['prepared_memory']=guard(torch);report['warmup_graph']=target.capabilities;save()
            provider,profile=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=CAPACITY)
            report['profile']=profile
            executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=CAPACITY,expert_backend=expert),chunk_rows=1024)
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                with spec.speculative_engine(native,target,provider,eos_ids=session.eos,prior=session.calibration['conditional_prior'],
                    mode='adaptive',capacity=262144,max_requests=1,token_budget=1024,tokenizer=session.tokenizer,prefill_executor=executor) as engine:
                    captured=[];old_release=engine.bridge.backend.release
                    def release(handle,state):
                        captured.append(dict(target=state.target,start=state.context_start,rows=len(state.context),
                            feature_storage_bytes=state.context.untyped_storage().nbytes(),feature_suffix=state.context.clone()))
                        old_release(handle,state)
                    engine.bridge.backend.release=release
                    try:
                        torch.cuda.synchronize();start=time.perf_counter();handle=engine.submit_tokens('e46-256k',ids,max_tokens=OUTPUT)
                        events=[]
                        for index in range(256):
                            report['active']='prefill:'+str(index+1)+'/256'
                            t=time.perf_counter();new=engine.step();torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
                            events.extend(new);state=engine.bridge.backend.states[handle];mem=guard(torch)
                            report['chunks'].append(dict(index=index,position=state.target.pos,rows=state.target.pos-(report['chunks'][-1]['position'] if index else 0),wall_ms=elapsed,**mem))
                            save()
                            if new:break
                        assert len(report['chunks'])==256 and state.target.pos==PROMPT and len(events)==1 and len(events[0].token_ids)==1
                        report['prefill_active_ms']=(time.perf_counter()-start)*1000
                        report['first_token_ttft_ms']=events[0].ttft_seconds*1000
                        report['prefill_feature_rows']=len(state.context);report['feature_storage_bytes']=state.context.untyped_storage().nbytes()
                        assert len(state.context)==2040 and report['feature_storage_bytes']==104448000
                        base=state.target;prefix=base.snapshot();basepos=base.pos;anchor=events[0].token_ids[0]
                        report['active']='256K selector and Graph diagnostics';save();diagnostic_start=time.perf_counter()
                        report['selector']=selector_check(torch,executor.workspace,CAPACITY)
                        report['prefix_sha256']=prefix_digest(torch,base,basepos);save()
                        # Compare every actual Graph bucket to the full-logit eager
                        # path at the long prefix; compare all changed live state.
                        sample_ids=[anchor]+ids[-7:]
                        for parity in (0,1):
                            for width in range(1,9):
                                keep=width if width%2 else width//2
                                base.restore(prefix)
                                logits,features,b=target.eager.run(base,sample_ids[:width]);target.eager.commit(base,b,width,keep)
                                expected=base.snapshot();expected_tail=tail(base,basepos)
                                base.restore(prefix)
                                if parity:
                                    base.rec[1].copy_(base.rec[0]);base.cur=[1]*len(base.cur)
                                replays=target.replays;g,f,b=target.run(base,sample_ids[:width]);target.commit(base,b,width,keep)
                                assert torch.equal(logits,g) and torch.equal(features,f),'256K Graph full logits/features mismatch'
                                exact_state(torch,base,expected,expected_tail,basepos)
                                assert target.replays==replays+1
                                report['windows'].append(dict(width=width,parity=parity,keep=keep,logits_exact=True,features_exact=True,state_exact=True));save()
                                del logits,features,b,expected,expected_tail,g,f
                        base.restore(prefix);gold=[anchor]
                        for _ in range(OUTPUT-1):
                            if gold[-1] in session.eos:break
                            logits,_=target.eager.step(base,[gold[-1]],capture=False)
                            gold.append(int(logits[-1].argmax()))
                        final=base.snapshot();final_tail=tail(base,basepos);base.restore(prefix)
                        report['diagnostic_ms']=(time.perf_counter()-diagnostic_start)*1000
                        report['oracle_output_tokens']=len(gold);report['oracle_position']=final['pos'];report['active']='256K dynamic decode';save()
                        traces=[];start=time.perf_counter()
                        for _ in range(OUTPUT+4):
                            if not engine.active_requests:break
                            events.extend(engine.step());traces.extend({k:m[k] for k in ('prompt','context_rows','verified_rows','emitted','accepted','observed','rates','head_fallback','head_union_count')} for m in engine.last_step_metrics)
                        torch.cuda.synchronize();report['decode_active_ms']=(time.perf_counter()-start)*1000
                        assert not engine.active_requests and len(captured)==1
                        tokens=[t for e in events for t in e.token_ids]
                        assert tokens==gold,'256K dynamic output differs from same-prefix target-only continuation'
                        data=captured[0];exact_state(torch,data['target'],final,final_tail,basepos)
                        report['prefix_after_sha256']=prefix_digest(torch,data['target'],basepos)
                        assert report['prefix_after_sha256']==report['prefix_sha256'],'committed prefix cache changed'
                        assert data['start']+data['rows']==data['target'].pos and data['feature_storage_bytes']==104448000
                        text=session.tokenizer.decode(tokens,skip_special_tokens=True)
                        report.update(output_tokens=len(tokens),output_sha256=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),
                            generated_text=text,retrieval_keys=list(KEYS),retrieval_hits=[k in text for k in KEYS],oracle_tokens_exact=True,
                            oracle_state_exact=True,prefix_cache_unchanged=True,trace=traces,context_start=data['start'],
                            retained_rows=data['rows'],final_position=data['target'].pos,graph=target.capabilities,
                            decode_active_tps=(len(tokens)-1)*1000/report['decode_active_ms'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            final_active_memory=memory(torch))
                        assert not target.fallbacks and target.head_fallbacks==0 and target.captures==16
                        save()
                        del data,base,prefix,final,final_tail,logits,state,captured
                    finally:engine.bridge.backend.release=old_release
            report['status']='PASS_E46_STUDY';report.pop('active',None);save()
    finally:
        if executor is not None:executor.close()
        if target is not None:
            if target.leased:target.release_state(target.workspace)
            target.close()
        del target,executor
        gc.collect();torch.cuda.synchronize()
    return report
