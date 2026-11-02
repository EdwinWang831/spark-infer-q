"""Freeze E35 on top of immutable E33 inputs without modifying old jobs."""
import argparse
import json
from pathlib import Path
import time
from e28_probe import sha
from e35_probe import CONTRACT


def main(root):
    worker=json.loads((root/'worker-status.json').read_text())
    assert worker['status']=='READY' and worker['load_count']==1
    for area in ('pending','running'):
        assert not list((root/'jobs'/area).glob('*.json'))
    old=json.loads((root/'artifacts/e33-online-seal.json').read_text())
    for key in ('source_sha256','input_sha256'):
        for n,h in old[key].items():assert sha(root/n)==h,n
    sources=set(old['source_sha256'])|{'e35_overlap.py','e35_probe.py','e35_prepare.py','e32_probe_v2.py'}
    # The isolation helper was already frozen by E32; do not trust a mutable copy.
    e32=json.loads((root/'artifacts/e32-v2-seal.json').read_text())
    assert sha(root/'e32_probe_v2.py')==e32['source_sha256']['e32_probe_v2.py']
    seal=dict(contract=CONTRACT,created_unix=time.time(),worker_before=worker,
        source_sha256={n:sha(root/n) for n in sorted(sources)},input_sha256=old['input_sha256'],
        external_source_sha256=old['external_source_sha256'])
    with (root/'artifacts/e35-seal.json').open('x') as f:f.write(json.dumps(seal,indent=2)+'\n')
    name='189-e35-moe-overlap.json'
    for area in ('pending','running','done'):assert not (root/'jobs'/area/name).exists()
    p=root/'jobs/pending'/name;tmp=p.with_suffix('.tmp')
    with tmp.open('x') as f:f.write(json.dumps(dict(module='e35_probe',sha256=sha(root/'e35_probe.py'),arguments={}))+'\n')
    tmp.replace(p)
    print(json.dumps(dict(status='SUBMITTED',job=name)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
