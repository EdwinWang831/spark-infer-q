"""Submit the prespecified paired budget control after the graph comparison."""
import argparse
import json
from pathlib import Path
import time
from e28_probe import sha
from e29_budget import CONTRACT


def main(root):
    worker=json.loads((root/'worker-status.json').read_text());assert worker['status']=='READY' and worker['load_count']==1
    for area in ('pending','running'):assert not list((root/'jobs'/area).glob('*.json'))
    report=json.loads((root/'artifacts/e29-online.json').read_text());assert report['status']=='PASS_E29_ONLINE'
    old=json.loads((root/'artifacts/e29-online-seal.json').read_text())
    sources=set(old['source_sha256'])|{'e29_budget.py','e29_submit_budget.py','tests/test_e29_budget.py'}
    inputs=set(old['input_sha256'])|{'artifacts/e29-online.json','artifacts/e29-online-seal.json'}
    seal=dict(contract=CONTRACT,worker_before=worker,created_unix=time.time(),
        source_sha256={n:sha(root/n) for n in sorted(sources)},input_sha256={n:sha(root/n) for n in sorted(inputs)},external_source_sha256=old['external_source_sha256'])
    with (root/'artifacts/e29-budget-seal.json').open('x') as h:h.write(json.dumps(seal,indent=2)+'\n')
    name='102-e29-budget.json';assert not (root/'jobs/done'/name).exists() and not (root/'artifacts/e29-budget.json').exists()
    p=root/'jobs/pending'/name;tmp=p.with_suffix('.tmp')
    with tmp.open('x') as h:h.write(json.dumps(dict(module='e29_budget',sha256=sha(root/'e29_budget.py'),arguments={}))+'\n')
    tmp.replace(p);print(json.dumps(dict(status='SUBMITTED',job=name,timed_runs=32,load_count=1)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
