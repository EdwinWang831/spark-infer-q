"""Seal the single E33 screening format and resident inputs before execution."""
import argparse
import json
from pathlib import Path
import time
from e28_probe import sha
from e33_fused_probe import CONTRACT


def main(root):
    worker=json.loads((root/'worker-status.json').read_text());assert worker['status']=='READY' and worker['load_count']==1
    for area in ('pending','running'):assert not list((root/'jobs'/area).glob('*.json'))
    old=json.loads((root/'artifacts/e33-seal.json').read_text())
    for key in ('source_sha256','input_sha256'):
        for n,h in old[key].items():assert sha(root/n)==h,n
    sources=set(old['source_sha256'])|{'e33_fused_head.py','e33_fused_probe.py','e33_fused_prepare.py'}
    seal=dict(contract=CONTRACT,created_unix=time.time(),worker_before=worker,
        source_sha256={n:sha(root/n) for n in sorted(sources)},input_sha256={**old['input_sha256'],'artifacts/e33-seal.json':sha(root/'artifacts/e33-seal.json'),'artifacts/e33-probe.json':sha(root/'artifacts/e33-probe.json')},external_source_sha256=old['external_source_sha256'])
    with (root/'artifacts/e33-fused-seal.json').open('x') as f:f.write(json.dumps(seal,indent=2)+'\n')
    name='154-e33-fused.json'
    for area in ('pending','running','done'):assert not (root/'jobs'/area/name).exists()
    p=root/'jobs/pending'/name;tmp=p.with_suffix('.tmp')
    with tmp.open('x') as f:f.write(json.dumps(dict(module='e33_fused_probe',sha256=sha(root/'e33_fused_probe.py'),arguments={}))+'\n')
    tmp.replace(p);print(json.dumps(dict(status='SUBMITTED',job=name,load_count=1)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
