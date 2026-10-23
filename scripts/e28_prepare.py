"""Freeze eight existing engineering roots and submit one fixed-budget comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from e25_online import get_tasks
from e28_online import CONTRACT
from e28_probe import sha


def main(root):
    state=json.loads((root/'worker-status.json').read_text());assert state['status']=='READY' and state['load_count']==1
    proof=json.loads((root/'artifacts/e28-proposer.json').read_text());assert proof['status']=='PASS_E28_PROPOSER_DIAGNOSTIC'
    assert sum(t['proposer_ms']['w8'] for t in proof['cases'])<sum(t['proposer_ms']['bf16'] for t in proof['cases'])
    candidates=get_tasks(json.loads((root/'e24_test_tasks.json').read_text())['tasks'])
    tasks=[]
    for family,lang in [('external_qa','en'),('external_qa','zh'),('code','en'),('math','en'),('multi_turn','en'),('multi_turn','zh'),('long_context','en'),('long_context','zh')]:
        pool=[t for t in candidates if t['family']==family and t['language']==lang]
        tasks.append(min(pool,key=lambda t:hashlib.sha256(('e28-fixed-v1/'+t['id']).encode()).hexdigest()))
    tasks.sort(key=lambda t:hashlib.sha256(('e28-order-v1/'+t['id']).encode()).hexdigest())
    assert len(tasks)==len({t['group'] for t in tasks})==8
    name='098-e28-online.json'
    for area in ('pending','running'):assert not list((root/'jobs'/area).glob('*.json'))
    for area in ('pending','running','done'):assert not (root/'jobs'/area/name).exists()
    assert not (root/'artifacts/e28-online.json').exists()
    with (root/'e28_test_tasks.json').open('x') as h:h.write(json.dumps(dict(tasks=tasks),indent=2,ensure_ascii=False)+'\n')
    sources=['e28_online.py','e28_prepare.py','e28_probe.py','e25_online.py','draft_w8_kernel.cu',
        'spark_infer_q/draft_backbone.py','spark_infer_q/dflash2_reference.py','spark_infer_q/dflash2_block.py',
        'spark_infer_q/draft_context.py','spark_infer_q/confidence_features.py','spark_infer_q/draft_head.py',
        'spark_infer_q/speculative.py','spark_infer_q/draft_signals.py','spark_infer_q/prompt.py','e3_target.py','e3_wide_target.py','e24_run.py']
    inputs=['e28_test_tasks.json','e24_test_tasks.json','artifacts/e28-build.json','artifacts/e28-kernel.json','artifacts/e28-proposer.json']
    inputs += [f'artifacts/e24-test-{i:02d}.json' for i in range(3)]
    seal=dict(contract=CONTRACT,created_unix=time.time(),worker_before=state,source_sha256={n:sha(root/n) for n in sources},input_sha256={n:sha(root/n) for n in inputs})
    with (root/'artifacts/e28-online-seal.json').open('x') as h:h.write(json.dumps(seal,indent=2)+'\n')
    p=root/'jobs/pending'/name;tmp=p.with_suffix('.tmp')
    with tmp.open('x') as h:h.write(json.dumps(dict(module='e28_online',sha256=sha(root/'e28_online.py'),arguments={}))+'\n')
    tmp.replace(p);print(json.dumps(dict(status='SUBMITTED',job=name,prompts=8,timed_runs=64,worker_pid=state['pid'],load_count=1)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
