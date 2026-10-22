"""Measure all feasible verification widths on the frozen training-only hard pool."""
from collections import defaultdict
import json
import math
import statistics
import time
from e16_tasks import TASKS,selected_positions
from e17_run_contract import CONTRACT,sha


def run(session,arguments):
    from e3_wide_target import WideTarget
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_context import CachedContextDraft
    from spark_infer_q.confidence_features import ConfidenceDraft,feature_vector
    from spark_infer_q.confidence_head import choose_rows
    from spark_infer_q.budget_regret import budget_regret,regret_weight
    from spark_infer_q.local_policy import ProgressModel
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.speculative import greedy_result
    root=session.root;torch=session.torch;smoke=bool(arguments.get('smoke',False));shard=int(arguments.get('shard',0))
    assert 0<=shard<8
    pool_path=root/'e17_hard_pool.json';assert sha(pool_path)=='cf80eb26f88c1fb875017d31dc8d6836e2563b95b95747292b9200499327c3ab'
    pool=json.loads(pool_path.read_text())
    assert len(pool['rows'])==512 and all(r['split']=='train' for r in pool['rows'])
    wanted=defaultdict(set)
    for r in pool['rows']:wanted[r['id']].add(r['position'])
    ids=sorted(wanted);assert len(ids)==216
    ids=ids[:1] if smoke else ids[shard*27:(shard+1)*27]
    originals={};datasets={}
    for i in range(12):
        meta=json.loads((root/f'artifacts/e16-collection-{i:02d}.json').read_text());path=root/'artifacts'/meta['dataset']
        assert sha(path)==meta['dataset_sha256'];datasets[path.name]=sha(path)
        for line in path.open():
            rec=json.loads(line)
            if rec['id'] in ids:originals[rec['id']]=rec
    tasks={t['id']:t for t in TASKS};assert all(tasks[i]['split']=='train' for i in ids)
    prior_path=root/'artifacts/e16-selected-model.json';prior=json.loads(prior_path.read_text());model=ProgressModel(prior['metadata'])
    tag='e17-cost-smoke' if smoke else f'e17-cost-{shard:02d}'
    output=root/'artifacts'/(tag+'.json');assert not output.exists()
    names=['e17_cost_audit.py','e17_run_contract.py','e16_tasks.py','e3_target.py','e3_wide_target.py',
           'spark_infer_q/confidence_features.py','spark_infer_q/confidence_head.py','spark_infer_q/budget_regret.py',
           'spark_infer_q/draft_context.py','spark_infer_q/draft_head.py','spark_infer_q/prompt.py','spark_infer_q/speculative.py','spark_infer_q/local_policy.py']
    hashes={n:sha(root/n) for n in names}
    report=dict(status='RUNNING',smoke=smoke,shard=shard,contract=CONTRACT,tasks=[],source_sha256=hashes,
        pool_sha256=sha(pool_path),prior_sha256=sha(prior_path),datasets=datasets,started_unix=time.time(),
        target_object_id=id(session.weights),draft_object_id=id(session.draft),
        cost_scope='paired actual verify+commit medians; frozen paid-draft/feature/CPU cost for attribution; no online speed claim')
    def save():
        p=output.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(output)
    save();target=WideTarget(session.weights,capacity=2048)
    base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    native=CachedContextDraft.attach(base);draft=ConfidenceDraft.attach(base)
    try:
        for ti,task_id in enumerate(ids):
            task=tasks[task_id];original=originals[task_id];source_rows={r['position']:r for r in original['rows']}
            wanted_positions=sorted(wanted[task_id])[:1] if smoke else sorted(wanted[task_id])
            token_ids=session.tokenizer.apply_chat_template([dict(role='user',content=task['prompt'])],tokenize=True,
                add_generation_prompt=True,enable_thinking=task['thinking'],return_dict=False)
            if 'target_prompt_tokens' in task:
                n=task['target_prompt_tokens'];assert len(token_ids)>n;token_ids=token_ids[:n-128]+token_ids[-128:]
            state,ctx,lg=prompt_state(target,token_ids,chunk_size=8);initial=state.clone();tokens=[int(lg[-1].argmax())];features=[ctx]
            while len(tokens)<task['output_cap'] and tokens[-1] not in session.eos:
                lg,ft=target.step(state,[tokens[-1]]);features.append(ft);tokens.append(int(lg[-1].argmax()))
            assert __import__('hashlib').sha256(json.dumps(tokens).encode()).hexdigest()==original['token_sha256'],task_id
            all_features=torch.cat(features);del state,features
            record=dict(id=task_id,group=task['group'],language=task['language'],family=task['family'],positions=[])
            report['tasks'].append(record);native.begin((tag,task_id,'native'));draft.begin((tag,task_id,'featured'))
            prefix_state=initial.clone();replayed=0
            for pos in original['selected_positions']:
                if pos>wanted_positions[-1]:break
                prefix=all_features[:len(token_ids)+pos];anchor=tokens[pos]
                reference,_=native.propose(prefix,anchor);proposals,info=draft.propose(prefix,anchor);assert reference==proposals
                x=feature_vector(info['signals'],len(prefix),pos+1,task['output_cap']-pos-1,task['thinking'])
                assert all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10) for a,b in zip(x,source_rows[pos]['x'])),(task_id,pos,'features changed')
                if pos not in wanted_positions:continue
                while replayed<pos:target.step(prefix_state,[tokens[replayed]],capture=False);replayed+=1
                gold=prefix_state.clone();logs=[];states=[]
                for tok in tokens[pos:pos+min(8,len(tokens)-pos-1)]:
                    gl,_=target.step(gold,[tok],capture=False);logs.append(gl.cpu());states.append(gold.clone())
                row=source_rows[pos];limit=row['max_rows'];samples=[]
                for rep in range(-1,3):
                    order=list(range(1,limit+1))
                    if rep%2:order.reverse()
                    shift=(ti+pos+max(rep,0))%len(order);order=order[shift:]+order[:shift]
                    for width in order:
                        scratch=prefix_state.clone();selected=proposals[:width-1]
                        torch.cuda.synchronize();t=time.perf_counter();lg,ft,buf=target.run(scratch,[anchor]+selected)
                        out,keep,_,_=greedy_result(anchor,selected,lg.argmax(-1).tolist(),session.eos,row['remaining'])
                        target.commit(scratch,buf,width,keep);torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
                        assert out==tokens[pos+1:pos+1+keep] and keep==row['emitted'][width-1]
                        assert torch.equal(lg[:keep].cpu(),torch.cat(logs[:keep]))
                        assert not target.state_difference(scratch,states[keep-1])
                        if rep>=0:samples.append(dict(width=width,repetition=rep,verify_commit_ms=elapsed,emitted=keep,exact=True))
                        del scratch
                c=prior['costs']['long' if row['context_rows']>=768 else 'short'];paid=limit>1
                fee=(c['native_draft_ms']+c['feature_increment_ms']+prior['cpu_predict_ms']) if paid else 0.
                old=[v+fee for v in c['verify_commit_ms']]
                measured=[statistics.median(s['verify_commit_ms'] for s in samples if s['width']==w)+fee for w in range(1,limit+1)]
                actual=measured+old[limit:];rates=model.predict(row['x'])
                old_width=choose_rows(rates,old,limit);recost_width=choose_rows(rates,actual,limit)
                old_loss=budget_regret(row['emitted'],actual,old_width,CONTRACT['reference_rate'],limit)
                recost_loss=budget_regret(row['emitted'],actual,recost_width,CONTRACT['reference_rate'],limit)
                record['positions'].append(dict(position=pos,context_rows=row['context_rows'],remaining=row['remaining'],
                    emitted=row['emitted'],samples=samples,old_rows=old_width,recost_rows=recost_width,
                    actual_costs_ms=actual,old_costs_ms=old,old_actual_regret_ms=old_loss['regret_ms'],
                    recost_regret_ms=recost_loss['regret_ms'],
                    weight=regret_weight(recost_loss['regret_ms'],CONTRACT['old_weight_scale_ms'])))
                del gold,logs,states;save()
            assert len(record['positions'])==len(wanted_positions)
            del initial,prefix_state,ctx,all_features;save()
            print(json.dumps(dict(e17_cost=task_id,done=ti+1,total=len(ids))),flush=True)
        for n,h in hashes.items():assert sha(root/n)==h,n
        assert sha(pool_path)==report['pool_sha256'] and sha(prior_path)==report['prior_sha256']
        report.update(status='PASS_E17_COST_SMOKE' if smoke else 'PASS_E17_COST_SHARD',
            positions=sum(len(t['positions']) for t in report['tasks']),finished_unix=time.time());save()
        return dict(status=report['status'],artifact=output.name,positions=report['positions'],
                    target_object_id=id(session.weights),draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());save();raise
