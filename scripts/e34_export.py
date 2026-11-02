"""Validate immutable E34 evidence and retain the failed attempt and cost rejection."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from e33_fused_probe import summarize


def head_gate(cases):
    if len(cases)!=32 or len({c['id'] for c in cases})!=4:
        raise ValueError('four roots / 32 windows required')
    for root in {c['id'] for c in cases}:
        if sorted(c['step'] for c in cases if c['id']==root)!=list(range(8)):
            raise ValueError('missing or repeated window')
    for c in cases:
        if not c['check']['exact'] or c['check']['fallback']:
            raise ValueError('real-window exact/no-fallback check failed')
        times=c['full_timings']
        if sorted((t['mode'],t['repetition']) for t in times)!=sorted(
                (m,r) for m in ('before','after') for r in range(6)):
            raise ValueError('missing or repeated timing')
        if any(not math.isfinite(t['ms']) or t['ms']<=0 for t in times):
            raise ValueError('positive finite timings required')
    s=summarize(cases,'full_timings',('before','after'))
    passed=s['time_reduction']>=.05 and s['root_bootstrap_gain_95'][0]>0 and min(r['speed_gain'] for r in s['per_root'])>=-.02
    return ('PASS_E34_HEAD_GATE' if passed else 'REJECT_E34_HEAD_COST'),s


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(raw):
    checkout=Path(__file__).resolve().parents[1]
    seals=[]
    for stage in ('e34','e34-v2'):
        seal=json.loads((raw/'artifacts'/(stage+'-seal.json')).read_text())
        for key in ('source_sha256','input_sha256'):
            for n,h in seal[key].items():assert digest(raw/n)==h,n
        for n,h in seal['external_source_sha256'].items():
            assert digest(raw/'external/tensorfold'/n)==h,n
        seals.append(seal)
    for n,h in {**seals[0]['source_sha256'],**seals[1]['source_sha256']}.items():
        if n.startswith('e34'):
            assert digest(checkout/'scripts'/n)==h,n
    names=('185-e34-device-head.json','186-e34-diagnose.json',
           '187-e34-device-head-v2.json','188-e34-audit.json')
    jobs=[json.loads((raw/'jobs/done'/n).read_text()) for n in names]
    first,diagnostic,probe,audit=jobs
    assert first['status']=='FAIL' and first['error']=='AssertionError: separate_stream'
    assert all(j['status']=='PASS' for j in jobs[1:])
    report=json.loads((raw/'artifacts/e34-v2-probe.json').read_text())
    status,summary=head_gate(report['cases'])
    assert report['status']==probe['result']['status']==status
    assert summary==report['summary']==probe['result']['summary']
    assert report['seal_sha256']==digest(raw/'artifacts/e34-v2-seal.json')
    assert len(report['preflight'])==15 and all(p['exact'] for p in report['preflight'])
    assert len(report['compaction'])==5 and all(c['stable'] and c['guards'] for c in report['compaction'])
    assert any(c['count']>2048 for c in report['compaction'])
    assert audit['result']['status']=='PASS_E34_RESIDENT_AUDIT'
    assert audit['result']['allocated_bytes']==77594734592
    initial=json.loads((raw/'artifacts/e34-probe.json').read_text())
    for key in ('target_object_id','draft_object_id'):
        assert initial[key]==report[key]==probe['result'][key]==audit['result'][key]
    worker=json.loads((raw/'worker-status.json').read_text())
    assert worker['status']=='READY' and worker['load_count']==1
    assert all(j['worker_pid']==worker['pid'] and j['load_count']==1 for j in jobs)
    assert all(s['worker_before']['pid']==worker['pid'] for s in seals)
    assert worker['allocated_bytes']==seals[0]['worker_before']['allocated_bytes']==77594734592
    clean={k:v for k,v in report.items() if k not in ('target_object_id','draft_object_id','w8_identity')}
    return dict(status=status,initial_failure=dict(job=names[0],error=first['error'],
        diagnosis=diagnostic['result'],cause='captured external ranks tensor was not retained; reused storage triggered spurious nonfinite fallback',
        correction='v2 retains per-width ranks storage for Graph lifetime; original sealed attempt unchanged'),
        component=clean,source_sha256={**seals[0]['source_sha256'],**seals[1]['source_sha256']},
        input_sha256=seals[1]['input_sha256'],external_source_sha256=seals[1]['external_source_sha256'],
        residency=dict(same_worker=True,same_model_objects=True,load_count=1,model_reloaded=False,
            final_status='READY',before={k:seals[0]['worker_before'][k] for k in ('completed_jobs','failed_jobs','allocated_bytes')},
            after={k:worker[k] for k in ('completed_jobs','failed_jobs','allocated_bytes')}),
        jobs=[dict(job=j['job'],status=j['status'],sha256=digest(raw/'jobs/done'/j['job'])) for j in jobs],
        online_decode_run=False,default_promoted=False,exporter_sha256=digest(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=export(args.raw)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],summary=result['component']['summary']),indent=2))
