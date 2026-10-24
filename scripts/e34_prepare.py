"""Seal E34 beside immutable E33 sources and enqueue a serial resident job."""
import argparse
import json
from pathlib import Path
import time
from e28_probe import sha
from e34_probe import CONTRACT


def main(root):
    worker=json.loads((root/'worker-status.json').read_text())
    assert worker['status']=='READY' and worker['load_count']==1
    for area in ('pending','running'):
        assert not list((root/'jobs'/area).glob('*.json'))
    old=json.loads((root/'artifacts/e33-online-seal.json').read_text())
    for key in ('source_sha256','input_sha256'):
        for n,h in old[key].items():assert sha(root/n)==h,n
    sources=set(old['source_sha256'])|{'e34_device_head.py','e34_probe.py','e34_prepare.py'}
    seal=dict(contract=CONTRACT,created_unix=time.time(),worker_before=worker,
        source_sha256={n:sha(root/n) for n in sorted(sources)},input_sha256=old['input_sha256'],
        external_source_sha256=old['external_source_sha256'])
    with (root/'artifacts/e34-seal.json').open('x') as f:
        f.write(json.dumps(seal,indent=2)+'\n')
    name='185-e34-device-head.json'
    for area in ('pending','running','done'):
        assert not (root/'jobs'/area/name).exists()
    p=root/'jobs/pending'/name
    tmp=p.with_suffix('.tmp')
    with tmp.open('x') as f:
        f.write(json.dumps(dict(module='e34_probe',sha256=sha(root/'e34_probe.py'),arguments={}))+'\n')
    tmp.replace(p)
    print(json.dumps(dict(status='SUBMITTED',job=name)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    main(p.parse_args().root)
