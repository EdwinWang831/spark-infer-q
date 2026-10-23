"""Seal E32 compact staging only after the worker is idle."""
import argparse
import json
from pathlib import Path
import time
from e28_probe import sha
from e32_probe_v2 import CONTRACT


def main(root):
    worker=json.loads((root/'worker-status.json').read_text());assert worker['status']=='READY' and worker['load_count']==1
    for area in ('pending','running'):assert not list((root/'jobs'/area).glob('*.json'))
    build=json.loads((root/'artifacts/e32-build.json').read_text());assert sha(build['binary'])==build['binary_sha256']
    old=json.loads((root/'artifacts/e32-seal.json').read_text())
    for key in ('source_sha256','input_sha256'):
        for n,h in old[key].items():assert sha(root/n)==h,n
    sources=set(old['source_sha256'])|{'e32_build.py','e32_bindings.cpp','e32_probe_v2.py','e32_prepare_v2.py','build/e32/cuda/nvfp4/qmmf.cu','build/e32/cuda/kernels/qmm_frag.cuh'}
    inputs=set(old['input_sha256'])|{'artifacts/e32-build.json','artifacts/e32-seal.json','artifacts/e32-probe.json','jobs/done/111-e32-compact.json'}
    seal=dict(contract=CONTRACT,created_unix=time.time(),worker_before=worker,source_sha256={n:sha(root/n) for n in sorted(sources)},input_sha256={n:sha(root/n) for n in sorted(inputs)},external_source_sha256=old['external_source_sha256'])
    with (root/'artifacts/e32-v2-seal.json').open('x') as f:f.write(json.dumps(seal,indent=2)+'\n')
    name='114-e32-compact-v2.json'
    for area in ('pending','running','done'):assert not (root/'jobs'/area/name).exists()
    p=root/'jobs/pending'/name;tmp=p.with_suffix('.tmp')
    with tmp.open('x') as f:f.write(json.dumps(dict(module='e32_probe_v2',sha256=sha(root/'e32_probe_v2.py'),arguments={}))+'\n')
    tmp.replace(p);print(json.dumps(dict(status='SUBMITTED',job=name,load_count=1)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
