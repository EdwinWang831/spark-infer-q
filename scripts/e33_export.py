"""Verify the three immutable E33 stages and export prompt-free evidence."""
import argparse
import json
from pathlib import Path
from e28_probe import sha
from e33_probe import summarize as component_summary
from e33_online import summarize as online_summary


def main(raw,out):
    names=('e33','e33-fused','e33-online');reports=[];seals=[]
    for name in names:
        seal_path=raw/'artifacts'/(name+'-seal.json');seal=json.loads(seal_path.read_text());seals.append(seal)
        for kind in ('source_sha256','input_sha256'):
            for n,h in seal[kind].items():assert sha(raw/n)==h,n
        for n,h in seal['external_source_sha256'].items():assert sha(raw/'external/tensorfold'/n)==h,n
        report=json.loads((raw/'artifacts'/(name+('-probe.json' if name!='e33-online' else '.json'))).read_text())
        assert report['seal_sha256']==sha(seal_path)
        if reports:assert all(report[k]==reports[0][k] for k in ('target_object_id','draft_object_id'))
        reports.append(report)
    for report in reports[:2]:
        assert len(report['cases'])==32 and len(report['preflight'])==11 and report['separate_stream_exact']
        assert component_summary(report['cases'],'primitive_timings',('bf16','q8'))==report['primitive_summary']
        assert component_summary(report['cases'],'full_timings',('bf16','screened'))==report['full_summary']
        assert all(c['argmax_exact'] and c['original_winner_retained'] and c['bound_violations']==0 and c['subset_exact'] for c in report['cases'])
    online=reports[2];assert online['status'] in ('PASS_E33_ONLINE_GAIN','PASS_E33_ONLINE_NO_CONFIRMED_GAIN')
    assert online_summary(online['tasks'])==online['summary'] and online['captures']==dict(original=16,screened=16)
    assert len({t['group'] for t in online['tasks']})==8
    for task in online['tasks']:
        assert task['decision_trace_exact'] and len(task['checks'])==2 and all(all(c[k] for k in ('tokens_exact','committed_logits_exact','committed_features_exact','live_state_exact','head_argmax_exact')) for c in task['checks'])
        assert all(t['exact'] for t in task['timings'])
        def policy(steps):return [{k:s[k] for k in ('context_rows','width','emitted','accepted','observed','rates')} for s in steps]
        first=policy(task['timings'][0]['steps']);assert all(policy(t['steps'])==first for t in task['timings'])
    assert not {c['id'] for c in reports[1]['cases']}.intersection(t['id'] for t in online['tasks'])
    after=json.loads((raw/'worker-status.json').read_text());jobs=[]
    for name in ('115-e33-screen.json','154-e33-fused.json','155-e33-online.json'):
        job=json.loads((raw/'jobs/done'/name).read_text());assert job['status']=='PASS' and job['load_count']==1 and job['worker_pid']==after['pid'];jobs.append(job)
    assert all(s['worker_before']['load_count']==after['load_count']==1 and s['worker_before']['pid']==after['pid'] for s in seals)
    prior=json.loads((raw/'artifacts/e30-profile.json').read_text())
    assert all(reports[0][k]==prior[k] for k in ('target_object_id','draft_object_id'))
    def clean(report):
        return {k:v for k,v in report.items() if k not in ('target_object_id','draft_object_id','w8_identity')}
    public=dict(status=online['status'],initial=clean(reports[0]),fused=clean(reports[1]),online=clean(online),
        environment=json.loads((raw/'artifacts/e33-environment.json').read_text()),
        residency=dict(same_worker=True,same_target_object=True,same_draft_object=True,load_count=1,all_experiment_jobs_pass=True,
            before={k:seals[0]['worker_before'][k] for k in ('completed_jobs','failed_jobs','allocated_bytes')},
            after={k:after[k] for k in ('completed_jobs','failed_jobs','allocated_bytes')},jobs=[dict(job=j['job'],status=j['status'],started_unix=j['started_unix'],finished_unix=j['finished_unix']) for j in jobs]),
        source_seals={n:sha(raw/'artifacts'/(n+'-seal.json')) for n in names},
        source_sha256=seals[-1]['source_sha256'],input_sha256=seals[-1]['input_sha256'],external_source_sha256=seals[-1]['external_source_sha256'],exporter_sha256=sha(Path(__file__)))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(public,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=online['status'],summary=online['summary']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.raw,a.out)
