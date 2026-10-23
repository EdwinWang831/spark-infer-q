"""Linear runtime equivalence, cost accounting, and tokenizer preflight only."""
import json
import time

from e27_contract import CONTRACT,sha,check_seal,write_new


def run(session,arguments):
    from e24_run import expected_prompt
    from e23_train import timing
    from e17_train import expand
    from spark_infer_q.local_policy import ProgressModel
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.confidence_head import choose_rows
    assert not arguments
    root=session.root;check_seal(root);p=root/'artifacts/e27-preflight.json';assert not p.exists()
    threads=session.torch.get_num_threads()
    try:
        session.torch.set_num_threads(2)
        tasks=json.loads((root/'e27_test_tasks.json').read_text())['tasks']
        assert len(tasks)==len({t['group'] for t in tasks})==12
        used={t['group'] for n in ('e17_new_tasks.json','e23_test_tasks.json','e24_all_tasks.json','e26_test_tasks.json')
              for t in json.loads((root/n).read_text())['tasks']}
        assert not used&{t['group'] for t in tasks}
        lengths=[]
        for task in tasks:
            ids,h=expected_prompt(session,task);assert len(ids)+task['output_cap']+8<=2048
            lengths.append(dict(id=task['id'],tokens=len(ids),prompt_sha256=h))
        old=json.loads((root/'artifacts/e17-fit.json').read_text());base=old['models'][old['selected_model']]['metadata']
        original,fast=ProgressModel(base),NumpyLinearProgressModel(base)
        recs=[]
        for prefix,count in [('e17-learning',10),('e24-learning',9)]:
            for i in range(count):
                meta=json.loads((root/f'artifacts/{prefix}-{i:02d}.json').read_text());data=root/'artifacts'/meta['dataset']
                assert sha(data)==meta['dataset_sha256']
                recs.extend(r for r in map(json.loads,data.read_text().splitlines()) if r['split']=='calibration')
        cal=expand(recs,'runtime_equivalence_only');assert len(cal)==10054
        times=timing(dict(linear_original=original,linear_fast=fast),cal);times['linear_live']=times['linear_fast']
        max_error=0.;decisions=0
        for row in cal:
            a,b=original.predict(row['x']),fast.predict(row['x']);max_error=max(max_error,max(abs(x-y) for x,y in zip(a,b)))
            c=old['costs']['long' if row['context_rows']>=768 else 'short']
            for name in ('linear_original','linear_fast'):
                paid=c['native_draft_ms']+c['feature_increment_ms']+times[name]['cpu_predict_ms'] if row['max_rows']>1 else 0.
                curve=[v+paid for v in c['verify_commit_ms']]
                decisions+=choose_rows(a,curve,row['max_rows'])!=choose_rows(b,curve,row['max_rows'])
        assert max_error<1e-12 and decisions==0
        dest=root/'artifacts/e27-linear-model.json';write_new(dest,dict(metadata=base,feature_group='f1',experimental=True))
        report=dict(status='PASS_E27_PREFLIGHT',contract=CONTRACT,seal_sha256=sha(root/'artifacts/e27-seal.json'),
            preflight=dict(prompts=12,lengths=lengths,new_target_calls=0),calibration_rows=len(cal),
            linear_max_probability_error=max_error,linear_changed_actions=decisions,model_sha256=sha(dest),timings=times,
            costs=old['costs'],target_object_id=id(session.weights),draft_object_id=id(session.draft),finished_unix=time.time())
        check_seal(root);write_new(p,report)
        return dict(status='PASS_E27_PREFLIGHT',artifact=p.name,preflight_sha256=sha(p),
                    target_object_id=id(session.weights),draft_object_id=id(session.draft))
    finally:session.torch.set_num_threads(threads)
