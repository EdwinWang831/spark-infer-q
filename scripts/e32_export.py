"""Export the completed compact-staging screen, preserving its collector failure."""
import argparse
import json
from pathlib import Path
from e28_probe import sha
from e32_probe_v2 import summarize


def main(raw,out):
    report=json.loads((raw/'artifacts/e32-v2-probe.json').read_text());seal=json.loads((raw/'artifacts/e32-v2-seal.json').read_text())
    assert report['status'] in ('PASS_E32_COMPACT_GATE','REJECT_E32_COMPACT_PERFORMANCE')
    assert sha(raw/'artifacts/e32-v2-seal.json')==report['seal_sha256']
    for key in ('source_sha256','input_sha256'):
        for name,h in seal[key].items():assert sha(raw/name)==h,name
    assert summarize(report['cases'])==report['summary']
    job=json.loads((raw/'jobs/done/114-e32-compact-v2.json').read_text());assert job['status']=='PASS'
    failed=json.loads((raw/'jobs/done/111-e32-compact.json').read_text());assert failed['status']=='FAIL' and failed['error']=='AssertionError: '
    v1=json.loads((raw/'artifacts/e32-probe.json').read_text())
    assert len(v1['cases'])==72 and len(v1['preflight'])==224 and len(v1['trajectory'])==8
    assert all(c['outputs_exact'] and c['guards_exact'] for c in v1['cases'])
    assert all(t['logits_features_live_state_exact'] for t in v1['trajectory'])
    assert "len(report['cases'])==96" in failed['traceback']
    for name,h in seal['external_source_sha256'].items():assert sha(raw/'external/tensorfold'/name)==h,name
    prior=json.loads((raw/'artifacts/e30-profile.json').read_text())
    for key in ('target_object_id','draft_object_id'):assert report[key]==prior[key]
    before,after=seal['worker_before'],json.loads((raw/'worker-status.json').read_text())
    assert before['pid']==after['pid']==job['worker_pid'] and before['load_count']==after['load_count']==1
    public={k:report[k] for k in ('status','contract','started_unix','finished_unix','seal_sha256','baseline_binary_sha256','preflight','separate_stream_exact','cases','trajectory','summary')}
    public.update(build={k:v for k,v in report['build'].items() if k!='binary'},
        model_residency=dict(same_worker=True,same_target_object=True,same_draft_object=True,load_count=1,
            before={k:before[k] for k in ('completed_jobs','failed_jobs','allocated_bytes')},after={k:after[k] for k in ('completed_jobs','failed_jobs','allocated_bytes')}),
        collector_v1=dict(status='FAIL_INCOMPLETE_CAPTURE',real_cases=72,expected_cases=96,diagnostic_steps=8,preflight_cases=224,
            reason='MoE4.shared_act retained original class-method globals, bypassing the private module hook; final case-count assertion failed; original record preserved',
            kernel_numerics_failure=False,source_sha256=sha(raw/'e32_probe.py')),
        source_sha256=seal['source_sha256'],input_sha256=seal['input_sha256'],external_source_sha256=seal['external_source_sha256'],no_full_decode_measurement=True)
    assert len(public['cases'])==96 and len(public['preflight'])==224 and len(public['trajectory'])==8
    assert sum(len(c['timings']) for c in public['cases'])==2304
    assert sum(t['calls'] for c in public['cases'] for t in c['timings'])==12672
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(public,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=public['status'],summary=public['summary']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.raw,a.out)
