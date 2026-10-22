"""Finite resident E17 chain: paired costs -> learning -> fit -> sealed test -> online."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


FILES=['e17_pipeline.py','e17_run_contract.py','e17_cost_audit.py','e17_collect.py','e17_train.py','e17_online.py',
        'e17_online_tasks.py','e16_tasks.py','e15_metrics.py','spark_infer_q/regret_head.py',
        'spark_infer_q/budget_regret.py','spark_infer_q/confidence_head.py','spark_infer_q/confidence_features.py',
        'spark_infer_q/local_policy.py','spark_infer_q/draft_context.py','spark_infer_q/draft_head.py',
        'spark_infer_q/prompt.py','spark_infer_q/speculative.py','spark_infer_q/verification_budget.py',
        'e3_target.py','e3_wide_target.py','e3_mechanism_checks.py','e17_new_tasks.json','e17_hard_pool.json']


def main(root):
    output=root/'artifacts/e17-pipeline.json';assert not output.exists()
    hashes={n:digest(root/n) for n in FILES}
    before=json.loads((root/'worker-status.json').read_text())
    report=dict(status='RUNNING',started_unix=time.time(),source_sha256=hashes,worker_before=before,jobs=[],
        retained_preflight_failure='035-e17-cost-smoke.json: missing helper import; fixed without model reload',
        stage_order=['training_only_costs','new_learning_labels','fit_and_calibration_selection','new_test_labels','heldout_gate','online_if_pass'])
    def save():
        p=output.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2)+'\n');p.replace(output)
    def check():
        worker=json.loads((root/'worker-status.json').read_text())
        assert worker['pid']==before['pid'] and worker['load_count']==before['load_count']==1
        assert worker['status'] in ('READY','RUNNING'),worker
        for n,h in hashes.items():assert digest(root/n)==h,n
        return worker
    def wait_for(name):
        path=root/'jobs/done'/name
        while not path.exists():check();time.sleep(5)
        j=json.loads(path.read_text());assert j['status']=='PASS',j
        assert j['worker_pid']==before['pid'] and j['load_count']==1
        return j
    def submit(name,module,arguments,expected):
        check();assert not any((root/'jobs'/s/name).exists() for s in ('pending','running','done'))
        job=dict(module=module,sha256=hashes[module+'.py'],arguments=arguments)
        p=root/'jobs/pending'/name;p.with_suffix('.tmp').write_text(json.dumps(job));p.with_suffix('.tmp').replace(p)
        report['active_job']=name;save();j=wait_for(name);assert j['result']['status']==expected,j
        report['jobs'].append(dict(name=name,request=job,result=j));save();return j
    save()
    try:
        for name,expected in [('036-e17-cost-smoke-retry.json','PASS_E17_COST_SMOKE'),('037-e17-cost-00.json','PASS_E17_COST_SHARD')]:
            j=wait_for(name);assert j['result']['status']==expected
            report['jobs'].append(dict(name=name,result=j));save()
        for shard in range(1,8):submit(f'{37+shard:03d}-e17-cost-{shard:02d}.json','e17_cost_audit',dict(shard=shard),'PASS_E17_COST_SHARD')
        submit('045-e17-label-smoke.json','e17_collect',dict(smoke=True),'PASS_E17_LABEL_SMOKE')
        for shard in range(10):submit(f'{46+shard:03d}-e17-learning-{shard:02d}.json','e17_collect',dict(phase='learning',shard=shard),'PASS_E17_COLLECTION_SHARD')
        submit('056-e17-fit.json','e17_train',dict(stage='fit'),'PASS_E17_FIT')
        # Source test labels are not generated until the calibration selection is sealed.
        for shard in range(2):submit(f'{57+shard:03d}-e17-test-{shard:02d}.json','e17_collect',dict(phase='test',shard=shard),'PASS_E17_COLLECTION_SHARD')
        result=submit('059-e17-evaluation.json','e17_train',dict(stage='evaluate'),'PASS_E17_EVALUATION')
        if result['result']['gate_pass']:
            submit('060-e17-online.json','e17_online',{},'PASS_E17_ONLINE');report['status']='PASS_E17_CHAIN'
        else:report['status']='COMPLETE_OFFLINE_GATE_FAILED';report['online']='NOT_RUN'
        for _ in range(6):
            worker=check()
            if worker['status']=='READY':break
            time.sleep(1)
        report.update(worker_after=worker,finished_unix=time.time());report.pop('active_job',None);save()
    except Exception as exc:
        report.update(status='STOPPED_ON_FAILURE',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root.resolve())
