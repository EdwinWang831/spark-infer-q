"""Seal one layout candidate after the current-path hotspot profile."""
import argparse
import json
from pathlib import Path
import time
from e28_probe import sha
from e30_layout_probe import CONTRACT


def main(root):
    w=json.loads((root/'worker-status.json').read_text());assert w['status']=='READY' and w['load_count']==1
    for area in ('pending','running'):assert not list((root/'jobs'/area).glob('*.json'))
    assert json.loads((root/'artifacts/e30-profile.json').read_text())['status']=='PASS_E30_PROFILE'
    old=json.loads((root/'artifacts/e30-profile-seal.json').read_text());sources=set(old['source_sha256'])|{'e30_head_layout.py','e30_layout_probe.py','e30_submit_layout.py'}
    inputs=set(old['input_sha256'])|{'artifacts/e30-profile-seal.json','artifacts/e30-profile.json'}
    seal=dict(contract=CONTRACT,created_unix=time.time(),worker_before=w,source_sha256={n:sha(root/n) for n in sorted(sources)},input_sha256={n:sha(root/n) for n in sorted(inputs)},external_source_sha256=old['external_source_sha256'])
    with (root/'artifacts/e30-layout-seal.json').open('x') as h:h.write(json.dumps(seal,indent=2)+'\n')
    name='104-e30-layout.json';assert not (root/'jobs/done'/name).exists();p=root/'jobs/pending'/name;tmp=p.with_suffix('.tmp')
    with tmp.open('x') as h:h.write(json.dumps(dict(module='e30_layout_probe',sha256=sha(root/'e30_layout_probe.py'),arguments={}))+'\n')
    tmp.replace(p);print(json.dumps(dict(status='SUBMITTED',job=name,load_count=1)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
