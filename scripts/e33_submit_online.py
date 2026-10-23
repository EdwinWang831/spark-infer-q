"""Freeze normal generation A/B only after the complete screened head passes its gate."""
import argparse
import json
from pathlib import Path
import time
from e28_probe import sha
from e33_online import CONTRACT


def main(root):
    worker=json.loads((root/'worker-status.json').read_text());assert worker['status']=='READY' and worker['load_count']==1
    for area in ('pending','running'):assert not list((root/'jobs'/area).glob('*.json'))
    assert json.loads((root/'artifacts/e33-fused-probe.json').read_text())['status']=='PASS_E33_HEAD_GATE'
    old=json.loads((root/'artifacts/e33-fused-seal.json').read_text())
    for key in ('source_sha256','input_sha256'):
        for n,h in old[key].items():assert sha(root/n)==h,n
    sources=set(old['source_sha256'])|{'e33_target.py','e33_online.py','e33_submit_online.py'}
    inputs=set(old['input_sha256'])|{'artifacts/e33-fused-seal.json','artifacts/e33-fused-probe.json','artifacts/e29-online.json','e28_test_tasks.json'}
    seal=dict(contract=CONTRACT,created_unix=time.time(),worker_before=worker,source_sha256={n:sha(root/n) for n in sorted(sources)},input_sha256={n:sha(root/n) for n in sorted(inputs)},external_source_sha256=old['external_source_sha256'])
    with (root/'artifacts/e33-online-seal.json').open('x') as f:f.write(json.dumps(seal,indent=2)+'\n')
    name='155-e33-online.json'
    for area in ('pending','running','done'):assert not (root/'jobs'/area/name).exists()
    p=root/'jobs/pending'/name;tmp=p.with_suffix('.tmp')
    with tmp.open('x') as f:f.write(json.dumps(dict(module='e33_online',sha256=sha(root/'e33_online.py'),arguments={}))+'\n')
    tmp.replace(p);print(json.dumps(dict(status='SUBMITTED',job=name,load_count=1)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
