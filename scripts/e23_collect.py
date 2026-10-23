"""Independent test producer, opened only after E23 calibration selection."""
import json
import time

from e23_contract import check_seal, sha, write_new


def run(session, arguments):
    root=session.root;check_seal(root)
    fit_path=root/'artifacts/e23-fit.json';fit=json.loads(fit_path.read_text())
    assert fit['status']=='PASS_E23_FIT' and fit['eligible_fresh_test']
    assert fit['selected_model']=='decision_regret'
    assert sha(root/'artifacts/e23-selected-model.json')==fit['selected_model_sha256']
    validation=json.loads((root/'artifacts/e21-collector-validation.json').read_text())
    assert validation['status']=='PASS_E21_FAST_COLLECTOR' and validation['gate_pass']
    tasks=json.loads((root/'e23_test_tasks.json').read_text())['tasks']
    assert len(tasks)==72 and len({t['group'] for t in tasks})==60
    assert all(t['split']=='test' for t in tasks)
    shard=arguments['shard'];assert type(shard) is int and 0<=shard<3
    tasks=tasks[shard*24:(shard+1)*24]
    from e3_wide_target import WideTarget
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_context import CachedContextDraft
    from spark_infer_q.confidence_features import ConfidenceDraft
    from spark_infer_q.local_policy import ProgressModel
    from spark_infer_q.collection_budget import BudgetedTrajectoryDraft
    from spark_infer_q.collector import collect_record
    target=WideTarget(session.weights,capacity=2048)
    base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    native=CachedContextDraft.attach(base);featured=ConfidenceDraft.attach(base)
    prior=json.loads((root/'artifacts/e16-selected-model.json').read_text())
    trajectory=BudgetedTrajectoryDraft(ConfidenceDraft.attach(base),ProgressModel(prior['metadata']),
        prior['costs'],prior['cpu_predict_ms'],session.calibration['conditional_prior'])
    tag=f'e23-test-{shard:02d}';path=root/f'artifacts/{tag}.json';data=root/f'artifacts/{tag}-rows.jsonl'
    assert not path.exists() and not data.exists()
    report=dict(status='RUNNING',tasks=[],dataset=data.name,shard=shard,
        fit_sha256=sha(fit_path),manifest_sha256=sha(root/'e23_test_tasks.json'),
        seal_sha256=sha(root/'artifacts/e23-seal.json'),producer='e23-sealed-test-v1',
        started_unix=time.time(),target_object_id=id(session.weights),draft_object_id=id(session.draft))
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(path)
    data.touch();save()
    for task in tasks:
        trajectory.configure(task,calibrate=False)
        rec,transient=collect_record(session,target,native,featured,trajectory,task,mode='speculative',calibrate=False)
        with data.open('a') as handle:handle.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n')
        report['tasks'].append({k:v for k,v in rec.items() if k not in ('rows','cost_samples')})
        report['tasks'][-1]['positions']=len(rec['rows'])
        del transient
        save();print(json.dumps(dict(e23_test=task['id'],done=len(report['tasks']),total=len(tasks))),flush=True)
    check_seal(root)
    assert sha(fit_path)==report['fit_sha256'] and session.draft.max_context==1024
    report.update(status='PASS_E23_TEST_COLLECTION',dataset_sha256=sha(data),finished_unix=time.time(),
                  positions=sum(t['positions'] for t in report['tasks']))
    save()
    return dict(status=report['status'],artifact=path.name,positions=report['positions'],
                target_object_id=id(session.weights),draft_object_id=id(session.draft))
