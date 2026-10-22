"""Resident bounded-generation A/B and a separate signal-observation pass."""
import hashlib
import json
from pathlib import Path
import statistics
import time

from e13_tasks import CONTRACT, TASKS


def run(session, arguments):
    contract=dict(CONTRACT,**arguments.get("contract",{}))
    tasks=arguments.get("tasks",TASKS)
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.verification_budget import LaggedAcceptance
    from spark_infer_q.draft_signals import SignalDraft, observed_labels
    from spark_infer_q.moe_cost import union_total
    from e3_mechanism_checks import entry, params
    import spark_q_e3_varlen
    assert hashlib.sha256(Path(spark_q_e3_varlen.__file__).read_bytes()).hexdigest()=='722b6ea835fa3e7e349af7f22ac727045a65af487a35f6709f3838e066bd678e'
    torch, target, base = session.torch, session.target, session.draft
    phase = arguments['phase']
    assert phase in ('signals','integrated')
    cached = None
    if phase == 'integrated':
        from importlib import reload
        from spark_infer_q import draft_context
        cached = reload(draft_context).CachedContextDraft.attach(base)
    signal_draft = SignalDraft.attach(base)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    paths = [Path(__file__),session.root/'e13_tasks.py',
             session.root/'spark_infer_q/draft_context.py',session.root/'spark_infer_q/draft_signals.py',
             session.root/'e3_mechanism_checks.py']
    for name in arguments.get('sources',[]):
        assert Path(name).name==name and name.endswith('.py')
        paths.append(session.root/name)
    hashes = {str(p.relative_to(session.root)):sha(p) for p in paths}
    report = dict(status='RUNNING',phase=phase,contract=contract,tasks=[],source_sha256=hashes,
                  target_object_id=id(session.weights),draft_object_id=id(base),started_unix=time.time())
    artifact=arguments.get('artifact','e13-'+phase+'.json')
    assert Path(artifact).name==artifact and artifact.endswith('.json')
    output = session.root/'artifacts'/artifact
    def save():output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    def emit(item):print(json.dumps(item),flush=True)
    save()
    curve = session.calibration['verify_commit_curve_ms']
    prior = session.calibration['conditional_prior']
    draft_ms = session.calibration['draft_median_ms']['8']
    def generate(task,start,context,first,mode,oracle=None,capture=False,signals=False):
        state=start.clone();ctx=context.clone();tokens=[first];steps=[];logs=[]
        worker = cached if mode == 'cached' else signal_draft if signals else base
        if mode=='cached':cached.begin((task['id'],time.monotonic_ns()))
        policy=LaggedAcceptance(prior);budget=spark_q_e3_varlen.BudgetProbe()
        budget.curve([(r,ms+(draft_ms if r>1 else 0.)) for r,ms in curve])
        for _ in range(5):budget.runtime(1,1,curve[0][1],0.)
        stop=first in session.eos;previous_union=None
        torch.cuda.synchronize();started=time.perf_counter()
        while len(tokens)<contract['output_cap'] and not stop:
            offset=len(tokens)-1;anchor=tokens[-1]
            if mode=='serial':
                lg,_=target.step(state,[anchor],capture=False)
                if capture:logs.append(lg.cpu())
                nxt=int(lg[-1].argmax());tokens.append(nxt);stop=nxt in session.eos
                continue
            width=budget.schedule(params(),[entry(policy.current,policy.previous,process=len(steps)+2,schedule=len(steps)+2)])[0]
            k=min(width-1,contract['output_cap']-len(tokens)-1)
            t=time.perf_counter()
            proposals,info=worker.propose(ctx,anchor) if k else ([],{})
            proposals=proposals[:k]
            torch.cuda.synchronize();dm=(time.perf_counter()-t)*1000
            if signals and k:
                reference,_=base.propose(ctx,anchor)
                assert proposals==reference[:k]
            t=time.perf_counter();lg,ft,b=target.run(state,[anchor]+proposals);vm=(time.perf_counter()-t)*1000
            true_union=None
            if signals and k:
                # Observed route is a label, never a same-step decision input.
                ol,of,ob,routes=target.observe(state,[anchor]+proposals)
                assert torch.equal(lg,ol) and torch.equal(ft,of)
                true_union=union_total(routes);lg,ft,b=ol,of,ob
            pred=lg.argmax(-1).tolist()
            out,keep,_,stop=greedy_result(anchor,proposals,pred,eos_ids=session.eos,
                                         max_new=contract['output_cap']-len(tokens))
            labels=observed_labels(proposals,pred,len(out),session.eos)
            accepted=sum(value for _,value in labels);observed=len(labels)
            t=time.perf_counter();target.commit(state,b,k+1,keep);cm=(time.perf_counter()-t)*1000
            ctx=torch.cat([ctx,ft[:keep]]);tokens.extend(out)
            assert len(ctx)==state.pos==start.pos+len(tokens)-1
            if oracle:
                assert out==oracle['tokens'][offset+1:offset+1+len(out)]
                for j in range(keep):assert torch.equal(lg[j:j+1].cpu(),oracle['logits'][offset+j])
            policy.observe(observed,accepted);budget.runtime(1,k+1,dm+vm+cm,0.)
            row=dict(width=width,verified_rows=k+1,context_rows=state.pos-keep,offset=offset,
                emitted=keep,accepted=accepted,observed=observed,draft_ms=dm,verify_ms=vm,commit_ms=cm,
                previous_union=previous_union,true_union=true_union)
            if signals:
                row['labels']=[dict(depth=d+1,matched=value,signals=info['signals'][d]) for d,value in labels]
            steps.append(row);previous_union=true_union
        torch.cuda.synchronize();elapsed=(time.perf_counter()-started)*1000
        result=dict(tokens=tokens,state=state,logits=logs,decode_ms=elapsed,steps=steps,eos=stop)
        if mode=='cached':result['cache']=dict(bytes=cached.cache_bytes,appended_rows=cached.appended_rows,reused_rows=cached.reused_rows)
        return result
    try:
        for task_index,task in enumerate(tasks):
            ids=session.tokenizer.apply_chat_template([dict(role='user',content=task['prompt'])],tokenize=True,
                add_generation_prompt=True,enable_thinking=task['thinking'],return_dict=False)
            assert len(ids)+contract['output_cap']+8<=target.capacity
            state,ctx,lg=prompt_state(target,ids,chunk_size=8);first=int(lg[-1].argmax())
            golden=generate(task,state,ctx,first,'serial',capture=True)
            rec=dict(**task,prompt_tokens=len(ids),completion_tokens=len(golden['tokens']),eos=golden['eos'],
                     token_sha256=hashlib.sha256(json.dumps(golden['tokens']).encode()).hexdigest(),timings=[],checks=[])
            report['tasks'].append(rec)
            if phase=='signals':
                # Assert the instrumentation's selected path equals the original.
                p,_=base.propose(ctx,first);q,_=signal_draft.propose(ctx,first);assert p==q
                measured=generate(task,state,ctx,first,'original',oracle=golden,signals=True)
                assert measured['tokens']==golden['tokens'] and not target.state_difference(measured['state'],golden['state'])
                rec.update(steps=measured['steps'],selected_path_probe_exact=True,all_committed_logits_state_exact=True)
                del measured
            else:
                # Full oracle runs also warm both implementations before timing.
                for mode in contract['modes']:
                    check=generate(task,state,ctx,first,mode,oracle=golden)
                    assert check['tokens']==golden['tokens'] and not target.state_difference(check['state'],golden['state'])
                    rec['checks'].append(dict(mode=mode,all_committed_logits_exact=True,live_state_exact=True))
                    del check
                for rep in range(contract['repetitions']):
                    modes=contract['modes'] if (rep+task_index)%2==0 else contract['modes'][::-1]
                    for slot,mode in enumerate(modes):
                        result=generate(task,state,ctx,first,mode)
                        assert result['tokens']==golden['tokens'] and not target.state_difference(result['state'],golden['state'])
                        rec['timings'].append(dict(mode=mode,repetition=rep,order=slot,decode_ms=result['decode_ms'],
                            steps=result['steps'],cache=result.get('cache'),tokens_and_state_exact=True))
                        del result
            del golden,state,ctx
            save();emit(dict(e13_phase=phase,task=task['id'],complete_tasks=len(report['tasks'])))
        for name,digest in hashes.items():assert sha(session.root/name)==digest,name
        report.update(status='PASS_E13_'+phase.upper(),finished_unix=time.time(),
            resource=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        save()
        return dict(status=report['status'],tasks=len(report['tasks']),artifact=output.name,
                    target_object_id=id(session.weights),draft_object_id=id(base))
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());save();raise
