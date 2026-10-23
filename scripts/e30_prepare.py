"""Seal the frozen W8 target profile, without touching prior experiment files."""
import argparse
import json
from pathlib import Path
import time
from e28_probe import sha
from e30_profile import CONTRACT


def main(root):
    w=json.loads((root/'worker-status.json').read_text());assert w['status']=='READY' and w['load_count']==1
    for area in ('pending','running'):assert not list((root/'jobs'/area).glob('*.json'))
    old=json.loads((root/'artifacts/e29-budget-seal.json').read_text())
    tasks=json.loads((root/'e29_calibration_tasks.json').read_text())['tasks']
    selected=[]
    for family,lang in [('external_qa','en'),('code','en'),('long_context','en'),('long_context','zh')]:
        choices=[t for t in tasks if t['family']==family and t['language']==lang];assert len(choices)==1;selected+=choices
    manifest=root/'e30_profile_tasks.json'
    with manifest.open('x') as h:h.write(json.dumps(dict(tasks=selected),indent=2,ensure_ascii=False)+'\n')
    sources=set(old['source_sha256'])|{'e30_profile.py','e30_prepare.py'}
    inputs=set(old['input_sha256'])|{'e30_profile_tasks.json','artifacts/e29-budget-seal.json','artifacts/e29-budget.json'}
    seal=dict(contract=CONTRACT,created_unix=time.time(),worker_before=w,
        source_sha256={n:sha(root/n) for n in sorted(sources)},input_sha256={n:sha(root/n) for n in sorted(inputs)},external_source_sha256=old['external_source_sha256'])
    with (root/'artifacts/e30-profile-seal.json').open('x') as h:h.write(json.dumps(seal,indent=2)+'\n')
    name='103-e30-profile.json';assert not (root/'jobs/done'/name).exists()
    p=root/'jobs/pending'/name;tmp=p.with_suffix('.tmp')
    with tmp.open('x') as h:h.write(json.dumps(dict(module='e30_profile',sha256=sha(root/'e30_profile.py'),arguments={}))+'\n')
    tmp.replace(p);print(json.dumps(dict(status='SUBMITTED',job=name,load_count=1)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
