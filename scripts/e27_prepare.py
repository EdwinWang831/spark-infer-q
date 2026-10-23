"""Submit the linear-only checks and a source-bound actual decode comparison."""
import argparse
import json
from pathlib import Path
import time

from e27_contract import CONTRACT,sha,check_seal,write_new


def submit(root,name,module,args):
    w=json.loads((root/'worker-status.json').read_text());assert w['status']=='READY' and w['load_count']==1
    for area in ('pending','running'):assert not list((root/'jobs'/area).glob('*.json'))
    for area in ('pending','running','done'):assert not (root/'jobs'/area/name).exists()
    p=root/'jobs/pending'/name;tmp=p.with_suffix('.tmp')
    write_new(tmp,dict(module=module,sha256=sha(root/(module+'.py')),arguments=args));tmp.replace(p)


def main(root,online=False):
    if online:
        seal=check_seal(root);j=json.loads((root/'jobs/done/090-e27-preflight.json').read_text())
        assert j['status']=='PASS' and j['result']['status']=='PASS_E27_PREFLIGHT'
        p=root/'artifacts/e27-preflight.json';assert sha(p)==j['result']['preflight_sha256']
        assert not (root/'artifacts/e27-online.json').exists()
        submit(root,'091-e27-online.json','e27_online',dict(preflight_sha256=sha(p)))
        print(json.dumps(dict(status='ONLINE_SUBMITTED',prompts=12,timed_runs=108)));return
    from e24_contract import check_seal as check24
    old=check24(root);w=json.loads((root/'worker-status.json').read_text());assert w['status']=='READY' and w['load_count']==1
    assert not (root/'artifacts/e27-seal.json').exists()
    sources=set(old['source_sha256'])|{'e27_contract.py','e27_run.py','e27_online.py','e27_prepare.py','e27_tasks.py',
        'spark_infer_q/linear_runtime.py','spark_infer_q/past_cost.py','tests/test_linear_runtime_cost.py'}
    inputs=set(old['input_sha256'])|{'e27_test_tasks.json','e26_test_tasks.json','artifacts/e24-seal.json'}
    inputs|={f'artifacts/e24-learning-{i:02d}{suffix}' for i in range(9) for suffix in ('.json','-rows.jsonl')}
    inputs|={f'artifacts/e17-learning-{i:02d}.json' for i in range(10)}
    seal=dict(contract=CONTRACT,worker_before=w,resident_pid=w['pid'],created_unix=time.time(),
        source_sha256={n:sha(root/n) for n in sorted(sources)},input_sha256={n:sha(root/n) for n in sorted(inputs)})
    write_new(root/'artifacts/e27-seal.json',seal);check_seal(root)
    submit(root,'090-e27-preflight.json','e27_run',{})
    print(json.dumps(dict(status='PREFLIGHT_SUBMITTED',seal_sha256=sha(root/'artifacts/e27-seal.json'),load_count=1)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--submit-online',action='store_true')
    a=p.parse_args();main(a.root.resolve(),a.submit_online)
