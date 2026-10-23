"""Export E31 evidence without prompts, model payloads or absolute runtime paths."""
import argparse
import json
from pathlib import Path
from e28_probe import sha
from e31_probe import summarize


def main(raw,out):
    report=json.loads((raw/'artifacts/e31-probe.json').read_text());seal=json.loads((raw/'artifacts/e31-seal.json').read_text())
    assert report['status'] in ('PASS_E31_EXPERT_GATE','REJECT_E31_EXPERT_PERFORMANCE')
    assert sha(raw/'artifacts/e31-seal.json')==report['seal_sha256']
    for key in ('source_sha256','input_sha256'):
        for name,h in seal[key].items():assert sha(raw/name)==h,name
    assert summarize(report['cases'])==report['summary']
    job=json.loads((raw/'jobs/done/106-e31-scale.json').read_text());assert job['status']=='PASS'
    before,after=seal['worker_before'],json.loads((raw/'worker-status.json').read_text())
    assert before['pid']==after['pid'] and before['load_count']==after['load_count']==1
    prior=json.loads((raw/'artifacts/e30-profile.json').read_text())
    assert report['target_object_id']==prior['target_object_id'] and report['draft_object_id']==prior['draft_object_id']
    build={k:v for k,v in report['build'].items() if k!='binary'}
    public={k:report[k] for k in ('status','contract','started_unix','finished_unix','seal_sha256','baseline_binary_sha256','scale_decode','cases','trajectory','summary')}
    public.update(build=build,sass=json.loads((raw/'artifacts/e31-sass.json').read_text()),
        model_residency=dict(same_worker=True,load_count=1,same_target_object=True,same_draft_object=True,
            before={k:before[k] for k in ('completed_jobs','failed_jobs','allocated_bytes')},after={k:after[k] for k in ('completed_jobs','failed_jobs','allocated_bytes')}),
        source_sha256=seal['source_sha256'],input_sha256=seal['input_sha256'],external_source_sha256=seal['external_source_sha256'],
        no_full_decode_measurement=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:f.write(json.dumps(public,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=public['status'],summary=public['summary']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.raw,a.out)
