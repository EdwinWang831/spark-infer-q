"""Finite expansion chain after E23; serialized use of the resident models."""
import argparse
import json
from pathlib import Path
import time

from e24_contract import check_seal,sha,write_new


def main(root):
    seal=check_seal(root);before=json.loads((root/'worker-status.json').read_text())
    assert before['load_count']==1 and before['status'] in ('READY','RUNNING')
    path=root/'artifacts/e24-pipeline.json';assert not path.exists()
    report=dict(status='WAITING_FOR_E23',jobs=[],started_unix=time.time(),
                seal_sha256=sha(root/'artifacts/e24-seal.json'),fixed_objective=seal['fixed_objective'])
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(path)
    def check():
        check_seal(root);state=json.loads((root/'worker-status.json').read_text())
        assert state['pid']==before['pid'] and state['load_count']==1 and state['status'] in ('READY','RUNNING')
        return state
    def submit(name,args,expected):
        state=check()
        while state['status']!='READY':time.sleep(3);state=check()
        assert not list((root/'jobs/pending').glob('*.json')) and not list((root/'jobs/running').glob('*.json'))
        assert not (root/'jobs/done'/name).exists()
        p=root/'jobs/pending'/name;temp=p.with_suffix('.tmp')
        write_new(temp,dict(module='e24_run',sha256=seal['source_sha256']['e24_run.py'],arguments=args));temp.replace(p)
        report.update(status='RUNNING',active_job=name);save()
        done=root/'jobs/done'/name
        while not done.exists():check();time.sleep(5)
        result=json.loads(done.read_text());assert result['status']=='PASS',result
        assert result['worker_pid']==before['pid'] and result['load_count']==1 and result['result']['status']==expected
        report['jobs'].append(result);save();return result['result']
    save()
    try:
        while True:
            check();prior=json.loads((root/'artifacts/e23-pipeline.json').read_text())
            if prior['status'].startswith('COMPLETE_'):break
            assert prior['status']=='RUNNING',prior
            time.sleep(15)
        report['e23_terminal_status']=prior['status']
        submit('071-e24-preflight.json',dict(stage='preflight'),'PASS_E24_PREFLIGHT')
        submit('072-e24-validate.json',dict(stage='validate'),'PASS_E24_PRODUCER_VALIDATION')
        for shard in range(9):
            submit(f'{73+shard:03d}-e24-learning-{shard:02d}.json',dict(stage='collect',phase='learning',shard=shard),'PASS_E24_COLLECTION')
        fit=submit('082-e24-fit.json',dict(stage='fit'),'PASS_E24_FIT')
        if fit['eligible_fresh_test']:
            for shard in range(3):
                submit(f'{83+shard:03d}-e24-test-{shard:02d}.json',dict(stage='collect',phase='test',shard=shard),'PASS_E24_COLLECTION')
            evaluation=submit('086-e24-evaluation.json',dict(stage='evaluate'),'PASS_E24_EVALUATION')
            report['status']='COMPLETE_FRESH_GATE_PASSED' if evaluation['gate_pass'] else 'COMPLETE_FRESH_GATE_FAILED'
        else:report['status']='COMPLETE_CALIBRATION_NOT_IMPROVED'
        report.update(finished_unix=time.time(),worker_after=check());report.pop('active_job',None);save()
    except Exception as exc:
        report.update(status='STOPPED_ON_FAILURE',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root.resolve())
