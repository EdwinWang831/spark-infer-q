"""Independent read-only replay of the completed fresh E23 test."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from e17_train import expand,replay_positions,prior_comparison
from spark_infer_q.confidence_head import headroom_capture
from spark_infer_q.local_policy import ProgressModel
from spark_infer_q.mlp_runtime import NumpyMLPProgressModel


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root,old_artifacts,out):
    a=root/'artifacts'
    fit=json.loads((a/'e23-fit.json').read_text());evaluation=json.loads((a/'e23-evaluate.json').read_text())
    assert evaluation['status']=='PASS_E23_EVALUATION' and not evaluation['gate_pass']
    assert evaluation['fit_sha256']==sha(a/'e23-fit.json')
    assert fit['selected_model_sha256']==sha(a/'e23-selected-model.json')
    records=[];files={}
    for i in range(3):
        meta_path=a/f'e23-test-{i:02d}.json';meta=json.loads(meta_path.read_text())
        assert meta['status']=='PASS_E23_TEST_COLLECTION' and meta['fit_sha256']==sha(a/'e23-fit.json')
        p=a/meta['dataset'];assert sha(p)==meta['dataset_sha256']==evaluation['datasets'][p.name]
        files[p.name]=sha(p);records.extend(json.loads(line) for line in p.read_text().splitlines())
    tasks=json.loads((root/'e23_test_tasks.json').read_text())['tasks']
    assert [r['id'] for r in records]==[t['id'] for t in tasks]
    assert len(records)==72 and len({r['group'] for r in records})==60
    rows=expand(records,'e23_fresh_test');assert len(rows)==evaluation['positions']==4442
    for row in rows:
        assert row['y']==[int(row['emitted'][-1]>d+1) for d in range(7)]
        expected=[math.log1p(row['context_rows']),math.log1p(row['position']+1),math.log1p(row['remaining'])]
        assert all(abs(x-y)<1e-13 for x,y in zip(row['x'][21:24],expected))
    outputs={}
    for name in ('bce_reference','decision_regret'):
        p=a/f'e23-{name}-model.json';assert sha(p)==fit['model_sha256'][name]
        model=NumpyMLPProgressModel(json.loads(p.read_text())['metadata'])
        outputs[name]=replay_positions(rows,model,'f1',fit['costs'],fit['timings'][name]['cpu_predict_ms'])
    old=json.loads((old_artifacts/'e17-fit.json').read_text())
    assert sha(old_artifacts/'e17-fit.json')==json.loads((a/'e23-seal.json').read_text())['input_sha256']['artifacts/e17-fit.json']
    linear=replay_positions(rows,ProgressModel(old['models'][old['selected_model']]['metadata']),
        'f1',fit['costs'],fit['timings']['linear_reference']['cpu_predict_ms'])
    point=headroom_capture(outputs['decision_regret'],outputs['decision_regret'][0]['costs_ms'],fit['fixed_rows'])
    for key in ('policy','fixed','oracle','captured'):assert abs(point[key]-evaluation['test'][key])<1e-12,key
    versus_bce=prior_comparison(outputs['decision_regret'],outputs['bce_reference'])
    versus_linear=prior_comparison(outputs['decision_regret'],linear)
    for actual,label in ((versus_bce,'versus_bce'),(versus_linear,'versus_linear')):
        assert abs(actual['relative_gain']-evaluation[label]['relative_gain'])<1e-12
        assert all(abs(x-y)<1e-12 for x,y in zip(actual['group_bootstrap_gain_95'],evaluation[label]['group_bootstrap_gain_95']))
    report=dict(status='PASS_E23_INDEPENDENT_CPU_AUDIT',auditor_sha256=sha(Path(__file__)),
        dataset_sha256=files,fit_sha256=sha(a/'e23-fit.json'),evaluation_sha256=sha(a/'e23-evaluate.json'),
        prompts=72,roots=60,positions=4442,reproduced=point,versus_bce=versus_bce,versus_linear=versus_linear,
        limits='independent host CPU replay/semantic and SHA checks; no new Target calls, fitting or online throughput')
    out.mkdir(parents=True,exist_ok=True)
    (out/'2026-10-23-independent-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','old_artifacts','out'):p.add_argument(name,type=Path)
    a=p.parse_args();print(audit(a.root,a.old_artifacts,a.out)['status'])
