"""One finite E16 experiment chain. Does not load models, retry, or restart worker."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    output=root/'artifacts/e16-pipeline.json'
    assert not output.exists(), 'do not overwrite an existing experiment chain'
    names=['e16_tasks.py','e16_collect.py','e16_train.py','e16_online.py','e16_online_tasks.py','e16_pipeline.py',
           'e15_metrics.py','spark_infer_q/confidence_features.py','spark_infer_q/confidence_head.py',
           'spark_infer_q/draft_fidelity.py','spark_infer_q/draft_context.py','spark_infer_q/draft_head.py',
           'spark_infer_q/local_policy.py','spark_infer_q/prompt.py','spark_infer_q/speculative.py',
           'spark_infer_q/verification_budget.py','e3_target.py','e3_wide_target.py','e3_mechanism_checks.py']
    hashes={n:digest(root/n) for n in names}
    before=json.loads((root/'worker-status.json').read_text())
    report=dict(status='RUNNING',started_unix=time.time(),source_sha256=hashes,
                resident_pid=before['pid'],load_count=before['load_count'],worker_before=before,jobs=[])
    def save():
        temporary=output.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(output)
    def check():
        worker=json.loads((root/'worker-status.json').read_text())
        assert worker['pid']==before['pid'] and worker['load_count']==before['load_count']==1
        assert worker['status'] in ('READY','RUNNING'), worker
        for n,h in hashes.items():assert digest(root/n)==h,n
        return worker
    def wait_for(name):
        path=root/'jobs/done'/name
        while not path.exists():check();time.sleep(5)
        result=json.loads(path.read_text());assert result['status']=='PASS',result
        assert result['worker_pid']==before['pid'] and result['load_count']==1
        return result
    def submit(name,module,arguments,expected):
        check()
        assert not any((root/'jobs'/state/name).exists() for state in ('pending','running','done'))
        job=dict(module=module,sha256=hashes[module+'.py'],arguments=arguments)
        p=root/'jobs/pending'/name;p.with_suffix('.tmp').write_text(json.dumps(job));p.with_suffix('.tmp').replace(p)
        report['active_job']=name;save()
        result=wait_for(name)
        assert result['result']['status']==expected,result
        report['jobs'].append(dict(name=name,request=job,result=result));save()
        return result
    save()
    try:
        # E15-head ablation has its own predeclared source and must complete first.
        wait_for('017-e16-ablation.json')
        preflight=json.loads((root/'artifacts/e16-tokenization-preflight.json').read_text())
        assert preflight['status']=='PASS' and len(preflight['tasks'])==600
        submit('018-e16-smoke.json','e16_collect',dict(smoke=True),'PASS_E16_SMOKE')
        for shard in range(12):
            submit(f'{19+shard:03d}-e16-collect-{shard:02d}.json','e16_collect',dict(shard=shard),'PASS_E16_COLLECTION_SHARD')
        result=submit('031-e16-train.json','e16_train',{},'PASS_E16_TRAINING')
        if result['result']['gate_pass']:
            submit('032-e16-online.json','e16_online',{},'PASS_E16_ONLINE')
            report['status']='PASS_E16_CHAIN'
        else:
            report['status']='COMPLETE_OFFLINE_GATE_FAILED';report['online']='NOT_RUN'
        # Worker may need one loop tick to publish READY after the done file.
        for _ in range(6):
            worker=check()
            if worker['status']=='READY':break
            time.sleep(1)
        report.update(worker_after=worker,finished_unix=time.time());report.pop('active_job',None);save()
    except Exception as exc:
        report.update(status='STOPPED_ON_FAILURE',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time())
        save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root.resolve())
