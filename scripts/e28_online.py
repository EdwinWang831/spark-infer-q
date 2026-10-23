"""Fixed-seven-draft paired decode with only the five MLP blocks quantized."""
import json
import random
import time

from e28_probe import sha,backend

MODES=('bf16','w8')
CONTRACT=dict(modes=list(MODES),repetitions=4,prompts=8,output_cap=384,capacity=2048,policy='fixed width8, cap/EOS-aware',
    selection='manifest-only salted SHA within eight prespecified E25 family/language strata; reused engineering roots, no new independent corpus claim',
    quantization='five MLP gate/up/down only; symmetric per-channel INT8/FP32 scales; BF16 activations and output',
    timing='decode after first prompt token; draft/F1 features/verify/commit/loop included; no predictor or dynamic budget calls',
    limits=['fixed budget isolates cost and acceptance changes; not a dynamic-policy promotion',
            'only eight reused roots, including two familiar-format synthetic ledgers',
            'BF16 controls coexist; quantized payload is incremental experimental memory, not net saving'])


def order(ti,rep):return list(MODES) if (ti+rep)%2==0 else list(reversed(MODES))


def summarize(tasks,samples=4000):
    assert len(tasks)==len({t['group'] for t in tasks})==8
    costs=[];tokens=[]
    for task in tasks:
        assert len(task['timings'])==8
        row={}
        for mode in MODES:
            ts=[v for v in task['timings'] if v['mode']==mode]
            assert sorted(v['repetition'] for v in ts)==[0,1,2,3]
            assert all(v['exact'] and v['decode_ms']>0 for v in ts)
            row[mode]=sum(v['decode_ms'] for v in ts)
        costs.append(row);tokens.append((task['completion_tokens']-1)*4)
    def point(indices):
        n=sum(tokens[i] for i in indices)
        rates={m:n*1000/sum(costs[i][m] for i in indices) for m in MODES}
        return rates,rates['w8']/rates['bf16']-1
    rates,gain=point(list(range(8)));rng=random.Random(281023)
    draws=sorted(point(rng.choices(range(8),k=8))[1] for _ in range(samples))
    trace={}
    for mode in MODES:
        runs=[v for t in tasks for v in t['timings'] if v['mode']==mode];steps=[s for v in runs for s in v['steps']]
        trace[mode]=dict(decode_ms=sum(v['decode_ms'] for v in runs),rounds=len(steps),
            emitted=sum(s['emitted'] for s in steps),accepted=sum(s['accepted'] for s in steps),observed=sum(s['observed'] for s in steps),
            phase_ms={k:sum(s[k] for s in steps) for k in ('draft_ms','verify_ms','commit_ms')})
    repeats=[]
    for rep in range(4):
        elapsed={m:sum(v['decode_ms'] for t in tasks for v in t['timings'] if v['mode']==m and v['repetition']==rep) for m in MODES}
        repeats.append(dict(repetition=rep,relative_gain=elapsed['bf16']/elapsed['w8']-1))
    return dict(decode_tps=rates,relative_gain=gain,prompt_bootstrap_gain_95=[draws[int(.025*(samples-1))],draws[int(.975*(samples-1))]],
        prompts=8,timed_runs=64,timed_tokens_per_arm=sum(tokens),trace=trace,per_repetition=repeats)


def run(session,arguments):
    from e3_wide_target import WideTarget
    from e24_run import expected_prompt
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.confidence_features import ConfidenceDraft
    from spark_infer_q.draft_backbone import attach_mlp_w8
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.draft_signals import observed_labels
    root=session.root;torch=session.torch;assert not arguments
    seal_path=root/'artifacts/e28-online-seal.json';seal=json.loads(seal_path.read_text());assert seal['contract']==CONTRACT
    def check():
        for label in ('source_sha256','input_sha256'):
            for n,h in seal[label].items():assert sha(root/n)==h,n
    check();ext,identity=backend(root)
    proof=json.loads((root/'artifacts/e28-proposer.json').read_text());assert proof['status']=='PASS_E28_PROPOSER_DIAGNOSTIC'
    assert proof['target_object_id']==id(session.weights) and proof['draft_object_id']==id(session.draft)
    tasks=json.loads((root/'e28_test_tasks.json').read_text())['tasks'];assert len(tasks)==8
    reference={t['id']:t for i in range(3) for t in json.loads((root/f'artifacts/e24-test-{i:02d}.json').read_text())['tasks']}
    output=root/'artifacts/e28-online.json';assert not output.exists()
    report=dict(status='RUNNING',contract=CONTRACT,tasks=[],started_unix=time.time(),seal_sha256=sha(seal_path),identity=identity,
        target_object_id=id(session.weights),draft_object_id=id(session.draft))
    def save():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(output)
    save()
    try:
        target=WideTarget(session.weights,capacity=2048)
        base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
        engines=dict(bf16=ConfidenceDraft.attach(base),w8=attach_mlp_w8(base,lambda x,p:ext.matmul(x,p.values,p.scales)))
        report['incremental_packed_bytes']=sum(v['packed_bytes'] for v in engines['w8'].w8_manifest.values());save()
        def generate(task,start,context,first,mode,oracle=None):
            state,ctx=start.clone(),context.clone();tokens=[first];logs=[];steps=[];stop=first in session.eos
            if mode!='serial':engines[mode].begin((task['id'],mode,time.monotonic_ns()))
            torch.cuda.synchronize();t0=time.perf_counter()
            while len(tokens)<task['output_cap'] and not stop:
                anchor=tokens[-1];offset=len(tokens)-1
                if mode=='serial':
                    lg,_=target.step(state,[anchor],capture=False);nxt=int(lg[-1].argmax());tokens.append(nxt);logs.append(lg.cpu());stop=nxt in session.eos
                    continue
                width=min(8,task['output_cap']-len(tokens));t=time.perf_counter()
                proposals,_=engines[mode].propose(ctx,anchor) if width>1 else ([],{})
                proposals=proposals[:width-1];torch.cuda.synchronize();dm=(time.perf_counter()-t)*1000
                t=time.perf_counter();lg,ft,buf=target.run(state,[anchor]+proposals);vm=(time.perf_counter()-t)*1000
                pred=lg.argmax(-1).tolist()
                emitted,keep,_,stop=greedy_result(anchor,proposals,pred,session.eos,task['output_cap']-len(tokens))
                labels=observed_labels(proposals,pred,len(emitted),session.eos)
                t=time.perf_counter();target.commit(state,buf,width,keep);cm=(time.perf_counter()-t)*1000
                ctx=torch.cat([ctx,ft[:keep]]);tokens.extend(emitted)
                assert len(ctx)==state.pos==start.pos+len(tokens)-1
                if oracle is not None:
                    assert emitted==oracle['tokens'][offset+1:offset+1+len(emitted)]
                    for j in range(keep):assert torch.equal(lg[j:j+1].cpu(),oracle['logits'][offset+j])
                steps.append(dict(context_rows=state.pos-keep,width=width,emitted=keep,accepted=sum(v for _,v in labels),
                    observed=len(labels),draft_ms=dm,verify_ms=vm,commit_ms=cm))
            torch.cuda.synchronize();elapsed=(time.perf_counter()-t0)*1000
            return dict(tokens=tokens,state=state,logits=logs,steps=steps,decode_ms=elapsed)
        for ti,task in enumerate(tasks):
            report['active_task']=dict(index=ti,id=task['id'],phase='prefill');save()
            ids,prompt_sha=expected_prompt(session,task);assert len(ids)+task['output_cap']+8<=2048
            assert prompt_sha==reference[task['id']]['prompt_sha256']
            start,ctx,lg=prompt_state(target,ids,chunk_size=8);first=int(lg[-1].argmax())
            report['active_task']['phase']='serial_reference';save()
            golden=generate(task,start,ctx,first,'serial')
            import hashlib
            token_sha=hashlib.sha256(json.dumps(golden['tokens']).encode()).hexdigest();assert token_sha==reference[task['id']]['token_sha256']
            rec=dict(id=task['id'],group=task['group'],language=task['language'],family=task['family'],prompt_tokens=len(ids),
                prompt_sha256=prompt_sha,completion_tokens=len(golden['tokens']),token_sha256=token_sha,eos=golden['tokens'][-1] in session.eos,
                checks=[],timings=[])
            report['tasks'].append(rec)
            for mode in MODES:
                report['active_task'].update(phase='correctness_and_warmup',mode=mode);save()
                result=generate(task,start,ctx,first,mode,oracle=golden)
                assert result['tokens']==golden['tokens'] and not target.state_difference(result['state'],golden['state'])
                rec['checks'].append(dict(mode=mode,tokens_exact=True,committed_logits_exact=True,live_state_exact=True));del result
            for rep in range(4):
                for slot,mode in enumerate(order(ti,rep)):
                    report['active_task'].update(phase='timing',mode=mode,repetition=rep,slot=slot);save()
                    result=generate(task,start,ctx,first,mode)
                    assert result['tokens']==golden['tokens'] and not target.state_difference(result['state'],golden['state'])
                    rec['timings'].append(dict(mode=mode,repetition=rep,order=slot,decode_ms=result['decode_ms'],steps=result['steps'],exact=True));del result
                save();print(json.dumps(dict(e28_online=task['id'],task_index=ti,repetition=rep)),flush=True)
            del start,ctx,golden
        check();assert session.draft.max_context==1024
        report.update(status='PASS_E28_FIXED_DECODE',summary=summarize(report['tasks']),finished_unix=time.time());report.pop('active_task',None);save()
        return dict(status=report['status'],artifact=output.name,target_object_id=id(session.weights),draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());save();raise
