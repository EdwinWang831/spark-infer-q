"""Freeze calibration-only costs and the complete three-arm online protocol."""
import argparse
import json
from pathlib import Path
import time

from e28_probe import sha
from e29_analysis import summarize
from e29_online import CONTRACT


def main(root):
    worker=json.loads((root/'worker-status.json').read_text());assert worker['status']=='READY' and worker['load_count']==1
    for area in ('pending','running'):assert not list((root/'jobs'/area).glob('*.json'))
    sweep_path=root/'artifacts/e29-sweep.json';sweep=json.loads(sweep_path.read_text());assert sweep['status']=='PASS_E29_WIDTH_SWEEP'
    probe=json.loads((root/'artifacts/e29-probe.json').read_text());assert probe['status']=='PASS_E29_GRAPH_PROBE'
    summary=summarize(sweep['samples']);assert json.loads(json.dumps(summary))==sweep['summary']
    # No test timings enter these estimates; no policy hyperparameter sweep.
    costs=dict(source_sha256=sha(sweep_path),modes={m:{b:dict(verify_commit_ms=summary[m]['verify_commit_ms'][b],paid_ms=summary[m]['paid_ms'][b]) for b in ('short','long')} for m in ('bf16','w8')})
    cost_path=root/'artifacts/e29-frozen-costs.json'
    with cost_path.open('x') as h:h.write(json.dumps(costs,indent=2)+'\n')
    old=json.loads((root/'artifacts/e29-sweep-seal.json').read_text())
    sources=set(old['source_sha256'])|{'e29_online.py','e29_submit_online.py','tests/test_e29_verification.py'}
    inputs=set(old['input_sha256'])|{'artifacts/e29-sweep.json','artifacts/e29-probe.json','artifacts/e29-frozen-costs.json','artifacts/e29-probe-seal.json','artifacts/e29-sweep-seal.json','e28_test_tasks.json'}
    # Verify the actual external source tree against the loader's pinned manifest.
    native=root.parent/'e2-load/reference/tensorfold';pin=json.loads((root.parent/'e2-load/artifacts/source-manifest.json').read_text())
    external={n:sha(native/n) for n in pin}
    assert external=={n:v['sha256'] for n,v in pin.items()}
    seal=dict(contract=CONTRACT,created_unix=time.time(),worker_before=worker,
        source_sha256={n:sha(root/n) for n in sorted(sources)},input_sha256={n:sha(root/n) for n in sorted(inputs)},external_source_sha256=external)
    with (root/'artifacts/e29-online-seal.json').open('x') as h:h.write(json.dumps(seal,indent=2)+'\n')
    name='101-e29-online.json';assert not (root/'jobs/done'/name).exists() and not (root/'artifacts/e29-online.json').exists()
    p=root/'jobs/pending'/name;tmp=p.with_suffix('.tmp')
    with tmp.open('x') as h:h.write(json.dumps(dict(module='e29_online',sha256=sha(root/'e29_online.py'),arguments={}))+'\n')
    tmp.replace(p);print(json.dumps(dict(status='SUBMITTED',job=name,prompts=8,timed_runs=96,load_count=1)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
