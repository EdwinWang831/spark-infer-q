"""Seal new E29 work without overwriting any earlier experiment source."""
import argparse
import json
from pathlib import Path
import time

from e28_probe import sha


def main(root, stage):
    state=json.loads((root/'worker-status.json').read_text())
    assert state['status']=='READY' and state['load_count']==1
    for area in ('pending','running'):
        assert not list((root/'jobs'/area).glob('*.json'))
    dependencies=set(json.loads((root/'artifacts/e28-online-seal.json').read_text())['source_sha256'])
    sources=dependencies|{'e29_graph.py','e29_probe.py','e29_prepare.py'}
    if stage=='sweep':sources|={'e29_sweep.py','e29_analysis.py','spark_infer_q/linear_runtime.py','spark_infer_q/local_policy.py','spark_infer_q/confidence_head.py','e12_routes.py','spark_infer_q/moe_cost.py'}
    inputs=['artifacts/e28-build.json','artifacts/e28-online.json']
    if stage=='sweep':inputs+=['e29_calibration_tasks.json','artifacts/e27-linear-model.json','artifacts/e27-preflight.json']
    seal=dict(stage=stage,created_unix=time.time(),worker_before=state,
        source_sha256={n:sha(root/n) for n in sorted(sources)},input_sha256={n:sha(root/n) for n in inputs})
    with (root/f'artifacts/e29-{stage}-seal.json').open('x') as h:h.write(json.dumps(seal,indent=2)+'\n')
    name={'probe':'099-e29-probe.json','sweep':'100-e29-sweep.json'}[stage]
    p=root/'jobs/pending'/name;assert not (root/'jobs/done'/name).exists()
    tmp=p.with_suffix('.tmp')
    with tmp.open('x') as h:h.write(json.dumps(dict(module='e29_'+stage,sha256=sha(root/f'e29_{stage}.py'),arguments={}))+'\n')
    tmp.replace(p);print(json.dumps(dict(status='SUBMITTED',job=name,pid=state['pid'],load_count=1)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['probe','sweep'],required=True);a=p.parse_args();main(a.root,a.stage)
