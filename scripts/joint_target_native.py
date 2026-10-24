"""Real DeepInfer target-only lifecycle and matched split/joint decode timings."""
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import ModuleType

CONTRACT=dict(pairs=2,repetitions=4,modes=['split','joint'],draft='none / NEXTN=0',
    target='same E33 screen, same 80 joint/single-slot Graphs and native scheduler; only decode grouping changes',
    timing='token API wall; full requests and post-both-first-token decode cohort separately; diagnostics/checks/capture excluded',
    scope='two controlled bilingual pairs, four repetitions; not DFlash/HTTP/production throughput')


def run(session,arguments):
    source=Path(arguments['source_root']);manifest=json.loads((source/'manifest.json').read_text())
    def check():
        for n,h in manifest.items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h,n
    check()
    tag='_joint_native_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(source/'spark_infer_q')];sys.modules[tag]=pkg
    target=None
    report=dict(status='RUNNING',contract=CONTRACT,pairs=[],lifecycle=[],source_sha256=manifest,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),model_reloaded=False)
    path=source/'native-result.json'
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    save()
    try:
        runtime=importlib.import_module(tag+'.kaifra_joint_runtime')
        native_module=importlib.import_module(tag+'.native_scheduler')
        module=importlib.import_module(tag+'.kaifra_joint_target')
        head=importlib.import_module(tag+'.screened_head_fused')
        from e3_wide_target import WideTarget
        from spark_infer_q.prompt import prompt_state
        torch=session.torch
        target=module.JointTarget(WideTarget(session.weights,capacity=2048),head.FusedScreenedHead(session.weights.head))
        report['active']='warmup';save();t=time.perf_counter();target.warmup()
        report['graph_warmup_ms']=(time.perf_counter()-t)*1000
        assert len(target.graphs)==80
        solo=[WideTarget(session.weights,capacity=2048) for _ in range(2)]
        tasks=[('code_en','Write a Python function that removes duplicates while preserving order. Return only code.',32),
               ('explain_zh','请用三点解释缓存与队列的区别。',32),
               ('qa_en','Explain how a cache helps repeated reads, using one concrete example.',24),
               ('json_zh','输出一个JSON对象，包含名称为缓存和队列的两个条目，每个条目给出用途。',24)]
        refs=[]
        for index,(name,text,limit) in enumerate(tasks):
            report['active']='reference:'+name;save()
            ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            eager=solo[index%2];st=eager.state();lgs=[];tokens=[];inputs=list(ids)
            for token in ids:
                lg,_=eager.step(st,[token],capture=False);lgs.append(lg.cpu())
            tokens.append(int(lg[-1].argmax()))
            while len(tokens)<limit and tokens[-1] not in session.eos:
                anchor=tokens[-1];inputs.append(anchor)
                lg,_=eager.step(st,[anchor],capture=False);lgs.append(lg.cpu());tokens.append(int(lg[-1].argmax()))
            refs.append(dict(name=name,ids=ids,inputs=inputs,tokens=tokens,logits=lgs,state=st,limit=limit))
        diagnostic_rows=state_checks=0
        with native_module.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=0) as native:
            def run_pair(pair_index,mode,diagnostic=False):
                nonlocal diagnostic_rows,state_checks
                pair=refs[pair_index*2:pair_index*2+2];shadows={};kinds={};traces=[]
                def observe(handle,pos,ids,lg,state):
                    nonlocal diagnostic_rows,state_checks
                    i=kinds[handle.request_id];ref=pair[i];eager=solo[i]
                    shadow=shadows.setdefault(handle,eager.state())
                    assert shadow.pos==pos
                    assert list(ids)==ref['inputs'][pos:pos+len(ids)]
                    for j,token in enumerate(ids):
                        assert torch.equal(lg[j:j+1].cpu(),ref['logits'][pos+j]);diagnostic_rows+=1
                        eager.step(shadow,[token],capture=False)
                    assert not eager.state_difference(state,shadow);state_checks+=1
                with runtime.joint_engine(native,target,eos_ids=session.eos,capacity=256,
                        observer=observe if diagnostic else None,joint_decode=mode=='joint') as engine:
                    execute=engine.bridge.execute
                    def traced(plan):
                        traces.append([(r.handle.request_id,r.position,len(r.token_ids),r.prompt) for r in plan.rows])
                        return execute(plan)
                    engine.bridge.execute=traced
                    before=dict(target.counters);output=[[],[]];first=[None,None];last=[None,None];intervals=[[],[]]
                    t0=time.perf_counter();handles=[];owned=[]
                    for i,ref in enumerate(pair):
                        name='request'+str(i);kinds[name]=i
                        h=engine.submit_tokens(name,ref['ids'],max_tokens=ref['limit']);handles.append(h)
                        owned.append(engine.bridge.requests[name]['state'])
                    cohort=None;cohort_counts=None
                    while engine.active_requests:
                        events=engine.step();now=time.perf_counter()
                        for ev in events:
                            i=kinds[ev.handle.request_id]
                            output[i].extend(ev.token_ids)
                            if ev.token_ids:
                                if first[i] is None:first[i]=now-t0
                                elif last[i] is not None:intervals[i].append((now-last[i])*1000)
                                last[i]=now
                        if cohort is None and all(v is not None for v in first):
                            cohort=time.perf_counter();cohort_counts=[len(x) for x in output]
                    torch.cuda.synchronize();end=time.perf_counter()
                    # Full-reference comparisons are outside both timers.
                    assert output==[r['tokens'] for r in pair]
                    assert all(not solo[i].state_difference(st,pair[i]['state']) for i,st in enumerate(owned))
                    assert not any(target.leased) and not engine.bridge.backend.states
                    assert not engine.bridge.native.has_unfinished_seqs()
                    assert engine.bridge.blocks.get_num_free_blocks()==engine.bridge.block_count-1
                    delta={k:target.counters[k]-before.get(k,0) for k in target.counters}
                    decode_tokens=sum(len(x)-n for x,n in zip(output,cohort_counts))
                    assert decode_tokens>0
                    return dict(mode=mode,full_ms=(end-t0)*1000,decode_ms=(end-cohort)*1000,
                        decode_tokens=decode_tokens,completion_tokens=[len(x) for x in output],ttft_ms=[t*1000 for t in first],
                        tpot_intervals_ms=intervals,counters=delta,trace=traces,exact=True,
                        token_sha256=[hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in output])
            for pi in range(2):
                row=dict(pair=pi,checks=[],timings=[]);report['pairs'].append(row)
                for mode in CONTRACT['modes']:
                    report['active']=dict(pair=pi,mode=mode,phase='correctness');save()
                    result=run_pair(pi,mode,True)
                    row['checks'].append(dict(mode=mode,exact=True,trace=result['trace'],counters=result['counters']))
                assert row['checks'][0]['trace']==row['checks'][1]['trace']
                for rep in range(4):
                    order=CONTRACT['modes'] if (pi+rep)%2==0 else list(reversed(CONTRACT['modes']))
                    for slot,mode in enumerate(order):
                        report['active']=dict(pair=pi,mode=mode,repetition=rep,phase='timing');save()
                        result=run_pair(pi,mode)
                        assert result['trace']==row['checks'][0]['trace']
                        result.update(repetition=rep,order=slot);row['timings'].append(result)
                    save();print(json.dumps(dict(pair=pi,repetition=rep,status='PASS')),flush=True)
            # Native cancellation at the public synchronous/prepared boundaries.
            pair=refs[:2]
            for phase in ('waiting','prepared_prompt','prepared_decode'):
                report['active']='cancel:'+phase;save()
                with runtime.joint_engine(native,target,eos_ids=session.eos,capacity=256) as engine:
                    a=engine.submit_tokens('reuse',pair[0]['ids'],max_tokens=8)
                    b=engine.submit_tokens('survivor',pair[1]['ids'],max_tokens=8)
                    seen={a:[],b:[]};plan=None
                    if phase=='prepared_decode':
                        while not all(seen.values()):
                            for ev in engine.step():seen[ev.handle].extend(ev.token_ids)
                    if phase!='waiting':plan=engine.bridge.prepare()
                    engine.cancel(a)
                    prefix=list(seen[a])
                    if plan is not None:engine.bridge.execute(plan);engine.bridge.complete(plan)
                    for ev in engine.drain():seen[ev.handle].extend(ev.token_ids)
                    assert seen[a]==prefix and seen[b]==pair[1]['tokens'][:8]
                    assert not any(target.leased)
                    c=engine.submit_tokens('reuse',pair[1]['ids'],max_tokens=4)
                    assert c!=a and c.generation>a.generation
                    try:engine.cancel(a)
                    except runtime.BridgeError:pass
                    else:raise AssertionError('stale handle cancelled reused slot')
                    result=[t for ev in engine.drain() for t in ev.token_ids]
                    assert result==pair[1]['tokens'][:4]
                    assert not any(target.leased) and not engine.bridge.native.has_unfinished_seqs()
                    report['lifecycle'].append(dict(phase=phase,survivor_exact=True,cancel_prefix_exact=True,reuse_exact=True,stale_rejected=True))
        summary={}
        for mode in CONTRACT['modes']:
            rows=[t for p in report['pairs'] for t in p['timings'] if t['mode']==mode]
            tokens=sum(t['decode_tokens'] for t in rows);ms=sum(t['decode_ms'] for t in rows)
            full=sum(t['full_ms'] for t in rows);complete=sum(sum(t['completion_tokens']) for t in rows)
            ttft=[x for t in rows for x in t['ttft_ms']];tpot=sorted(x for t in rows for r in t['tpot_intervals_ms'] for x in r)
            summary[mode]=dict(decode_tokens=tokens,decode_ms=ms,decode_tps=tokens*1000/ms,
                full_ms=full,completion_tokens=complete,full_tps=complete*1000/full,
                ttft_p50_ms=statistics.median(ttft),ttft_p95_ms=sorted(ttft)[int(.95*(len(ttft)-1))],
                tpot_p50_ms=statistics.median(tpot),tpot_p95_ms=tpot[int(.95*(len(tpot)-1))])
        gains=[]
        for rep in range(4):
            times={m:sum(t['decode_ms'] for p in report['pairs'] for t in p['timings'] if t['mode']==m and t['repetition']==rep) for m in CONTRACT['modes']}
            gains.append(times['split']/times['joint']-1)
        check();assert len(target.graphs)==80 and not target.fallbacks
        report.update(status='PASS_JOINT_NATIVE_TARGET',summary=summary,decode_speed_gain=summary['joint']['decode_tps']/summary['split']['decode_tps']-1,
            per_repetition_speed_gain=gains,diagnostic_logit_rows=diagnostic_rows,diagnostic_state_checks=state_checks,
            final_capabilities=target.capabilities,finished_unix=time.time())
        report.pop('active',None);save()
        return report
    finally:
        if target is not None:
            session.torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for st,leased in zip(target.states,target.leased):
                if leased:target.release_state(st)
            target.close()
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
