"""Seal and submit the single E31 candidate after build, preserving earlier jobs."""
import argparse
import json
from pathlib import Path
import time
from e28_probe import sha
from e31_probe import CONTRACT


def main(root):
    worker=json.loads((root/'worker-status.json').read_text());assert worker['status']=='READY' and worker['load_count']==1
    for area in ('pending','running'):assert not list((root/'jobs'/area).glob('*.json'))
    build=json.loads((root/'artifacts/e31-build.json').read_text());assert sha(build['binary'])==build['binary_sha256']
    old=json.loads((root/'artifacts/e30-layout-seal.json').read_text());sources=set(old['source_sha256'])|{'e31_build.py','e31_bindings.cu','e31_probe.py','e31_prepare.py','e31_sass.py'}
    inputs=set(old['input_sha256'])|{'artifacts/e31-build.json','artifacts/e31-sass.json','artifacts/e30-layout-seal.json'}
    sources|={'build/e31/cuda/nvfp4/experts.cu','build/e31/cuda/experts.cuh'}
    for key in ('source_sha256','input_sha256'):
        for n,h in old[key].items():assert sha(root/n)==h,n
    seal=dict(contract=CONTRACT,created_unix=time.time(),worker_before=worker,source_sha256={n:sha(root/n) for n in sorted(sources)},input_sha256={n:sha(root/n) for n in sorted(inputs)},external_source_sha256=old['external_source_sha256'])
    with (root/'artifacts/e31-seal.json').open('x') as f:f.write(json.dumps(seal,indent=2)+'\n')
    name='106-e31-scale.json'
    for area in ('pending','running','done','failed'):assert not (root/'jobs'/area/name).exists()
    path=root/'jobs/pending'/name;tmp=path.with_suffix('.tmp')
    with tmp.open('x') as f:f.write(json.dumps(dict(module='e31_probe',sha256=sha(root/'e31_probe.py'),arguments={}))+'\n')
    tmp.replace(path);print(json.dumps(dict(status='SUBMITTED',job=name,load_count=1)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
