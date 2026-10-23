"""Finite objective comparison on the existing serialized resident worker."""
import argparse
import json
from pathlib import Path
import time

from e23_contract import check_seal, sha, write_new


def main(root):
    seal=check_seal(root);worker=json.loads((root/'worker-status.json').read_text())
    assert worker['status']=='READY' and worker['load_count']==1
    path=root/'artifacts/e23-pipeline.json';assert not path.exists()
    report=dict(status='RUNNING',started_unix=time.time(),seal_sha256=sha(root/'artifacts/e23-seal.json'),
                jobs=[],worker_before=worker)
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(path)
    def check():
        check_seal(root);state=json.loads((root/'worker-status.json').read_text())
        assert state['pid']==worker['pid'] and state['load_count']==1
        assert state['status'] in ('READY','RUNNING')
        return state
    def submit(name,module,args,expected):
        state=check()
        while state['status']!='READY':time.sleep(2);state=check()
        assert not list((root/'jobs/pending').glob('*.json'))
        assert not list((root/'jobs/running').glob('*.json'))
        assert not (root/'jobs/done'/name).exists()
        request=dict(module=module,sha256=seal['source_sha256'][module+'.py'],arguments=args)
        p=root/'jobs/pending'/name;temp=p.with_suffix('.tmp');write_new(temp,request);temp.replace(p)
        report['active_job']=name;save()
        done=root/'jobs/done'/name
        while not done.exists():check();time.sleep(3)
        result=json.loads(done.read_text());assert result['status']=='PASS',result
        assert result['result']['status']==expected and result['worker_pid']==worker['pid'] and result['load_count']==1
        report['jobs'].append(result);save();return result['result']
    save()
    try:
        submit('065-e23-preflight.json','e23_train',dict(stage='preflight'),'PASS_E23_PREFLIGHT')
        fit=submit('066-e23-fit.json','e23_train',dict(stage='fit'),'PASS_E23_FIT')
        if fit['eligible_fresh_test']:
            for shard in range(3):
                submit(f'{67+shard:03d}-e23-test-{shard:02d}.json','e23_collect',dict(shard=shard),'PASS_E23_TEST_COLLECTION')
            evaluation=submit('070-e23-evaluation.json','e23_train',dict(stage='evaluate'),'PASS_E23_EVALUATION')
            report['status']='COMPLETE_FRESH_GATE_PASSED' if evaluation['gate_pass'] else 'COMPLETE_FRESH_GATE_FAILED'
        else:
            report['status']='COMPLETE_CALIBRATION_NOT_IMPROVED'
        report.update(finished_unix=time.time(),worker_after=check());report.pop('active_job',None);save()
    except Exception as exc:
        report.update(status='STOPPED_ON_FAILURE',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root.resolve())
