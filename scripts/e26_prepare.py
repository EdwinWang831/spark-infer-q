"""Seal new sources/data, then submit a finite CPU fit and conditional online job."""
import argparse
import json
from pathlib import Path
import time

from e26_contract import CONTRACT,sha,write_new,check_seal


def submit(root,name,module,arguments):
    state=json.loads((root/'worker-status.json').read_text())
    assert state['status']=='READY' and state['load_count']==1
    for area in ('pending','running'):
        assert not list((root/'jobs'/area).glob('*.json'))
    for area in ('pending','running','done'):
        assert not (root/'jobs'/area/name).exists()
    p=root/'jobs/pending'/name;tmp=p.with_suffix('.tmp')
    write_new(tmp,dict(module=module,sha256=sha(root/(module+'.py')),arguments=arguments));tmp.replace(p)


def prepare(root):
    from e24_contract import check_seal as check24
    prior=check24(root);state=json.loads((root/'worker-status.json').read_text())
    assert state['status']=='READY' and state['load_count']==1
    assert json.loads((root/'artifacts/e24-pipeline.json').read_text())['status']=='COMPLETE_FRESH_GATE_PASSED'
    assert not (root/'artifacts/e26-seal.json').exists() and not (root/'artifacts/e26-fit.json').exists()
    sources=set(prior['source_sha256'])|{'e26_contract.py','e26_run.py','e26_online.py','e26_prepare.py','e26_tasks.py',
        'spark_infer_q/residual_head.py','tests/test_residual_head.py'}
    inputs=set(prior['input_sha256'])|{'e26_test_tasks.json','artifacts/e24-seal.json','artifacts/e24-fit.json'}
    inputs|={f'artifacts/e24-learning-{i:02d}{suffix}' for i in range(9) for suffix in ('.json','-rows.jsonl')}
    seal=dict(contract=CONTRACT,created_unix=time.time(),resident_pid=state['pid'],worker_before=state,
        source_sha256={n:sha(root/n) for n in sorted(sources)},input_sha256={n:sha(root/n) for n in sorted(inputs)})
    write_new(root/'artifacts/e26-seal.json',seal);check_seal(root)
    submit(root,'088-e26-fit.json','e26_run',{})
    print(json.dumps(dict(status='FIT_SUBMITTED',seal_sha256=sha(root/'artifacts/e26-seal.json'),pid=state['pid'],load_count=1)))


def online(root):
    check_seal(root)
    job=json.loads((root/'jobs/done/088-e26-fit.json').read_text())
    assert job['status']=='PASS' and job['result']['status']=='PASS_E26_FIT'
    p=root/'artifacts/e26-fit.json';fit=json.loads(p.read_text());assert sha(p)==job['result']['fit_sha256']
    if not fit['eligible_online']:
        print(json.dumps(dict(status='COMPLETE_CALIBRATION_NOT_IMPROVED',selected_alpha=fit['selected_alpha'])));return
    assert not (root/'artifacts/e26-online.json').exists()
    submit(root,'089-e26-online.json','e26_online',dict(fit_sha256=sha(p)))
    print(json.dumps(dict(status='ONLINE_SUBMITTED',selected_alpha=fit['selected_alpha'],prompts=12,timed_runs=108)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--submit-online',action='store_true')
    a=p.parse_args();(online if a.submit_online else prepare)(a.root.resolve())
