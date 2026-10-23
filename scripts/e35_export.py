"""Recheck real MoE timings, actual kernel overlap and resident source identity."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from e35_probe import CONTRACT,summarize
from e35_trace import analyze


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(cases):
    roots={c['id'] for c in cases}
    if len(cases)!=24 or len(roots)!=4:raise ValueError('24 windows / four roots required')
    for root in roots:
        if sorted((c['step'],c['layer']) for c in cases if c['id']==root)!=[(s,l) for s in range(2) for l in (0,23,47)]:
            raise ValueError('missing or repeated layer/window')
    for c in cases:
        if not c['outputs_graph_exact']:raise ValueError('MoE exactness required')
        expected=sorted((r,m,i) for r in CONTRACT['regimes'] for m in CONTRACT['modes'] for i in range(6))
        if sorted((t['regime'],t['mode'],t['repetition']) for t in c['timings'])!=expected:
            raise ValueError('missing or repeated timing')
        if any(not math.isfinite(t['ms']) or t['ms']<=0 or t['calls']!=CONTRACT['calls_per_interval'][t['regime']] for t in c['timings']):
            raise ValueError('invalid timing or cache-regime accounting')
    result=summarize(cases);s=result['evict_128MiB']
    passed=s['time_reduction']>=.05 and s['root_bootstrap_gain_95'][0]>0 and min(r['speed_gain'] for r in s['per_root'])>=-.02
    return ('PASS_E35_MOE_GATE' if passed else 'REJECT_E35_MOE_COST'),result


def export(raw):
    checkout=Path(__file__).resolve().parents[1]
    seal=json.loads((raw/'artifacts/e35-seal.json').read_text())
    for key in ('source_sha256','input_sha256'):
        for n,h in seal[key].items():assert digest(raw/n)==h,n
    for n,h in seal['external_source_sha256'].items():assert digest(raw/'external/tensorfold'/n)==h,n
    for n,h in seal['source_sha256'].items():
        if n.startswith('e35'):assert digest(checkout/'scripts'/n)==h,n
    report=json.loads((raw/'artifacts/e35-probe.json').read_text())
    job=json.loads((raw/'jobs/done/189-e35-moe-overlap.json').read_text())
    audit=json.loads((raw/'jobs/done/190-e35-audit.json').read_text())
    assert job['status']==audit['status']=='PASS'
    assert report['contract']==CONTRACT and report['seal_sha256']==digest(raw/'artifacts/e35-seal.json')
    status,summary=gate(report['cases'])
    assert report['status']==job['result']['status']==status and report['summary']==job['result']['summary']==summary
    assert len(report['preflight'])==96 and all(c['outputs_graph_exact'] and c['guards_exact'] for c in report['preflight'])
    assert report['separate_caller_stream_exact']
    assert len(report['trajectory'])==8 and all(c['logits_features_live_state_exact'] for c in report['trajectory'])
    assert len(report['traces'])==8
    timelines=[]
    for t in report['traces']:
        path=raw/'artifacts'/t['file'];assert digest(path)==t['sha256']
        timelines.append(dict(**t,analysis=analyze(json.loads(path.read_text()))))
    for root in {t['id'] for t in timelines}:
        rows={t['mode']:t['analysis'] for t in timelines if t['id']==root}
        assert set(rows)=={'serial','overlap'}
        assert rows['serial']['kernel_calls']==rows['overlap']['kernel_calls']==19
        assert rows['serial']['routed_shared_overlap_us']==0
        assert rows['overlap']['routed_streams']!=rows['overlap']['shared_streams']
    assert audit['result']['status']=='PASS_E35_RESIDENT_AUDIT'
    for k in ('target_object_id','draft_object_id'):assert report[k]==job['result'][k]==audit['result'][k]
    worker=json.loads((raw/'worker-status.json').read_text());before=seal['worker_before']
    assert worker['status']=='READY' and worker['load_count']==before['load_count']==1
    assert worker['pid']==before['pid']==job['worker_pid']==audit['worker_pid']
    assert worker['allocated_bytes']==before['allocated_bytes']==audit['result']['allocated_bytes']==77594734592
    clean={k:v for k,v in report.items() if k not in ('target_object_id','draft_object_id','w8_identity')}
    clean['traces']=timelines
    return dict(status=status,component=clean,source_sha256=seal['source_sha256'],input_sha256=seal['input_sha256'],
        external_source_sha256=seal['external_source_sha256'],
        residency=dict(same_worker=True,same_model_objects=True,load_count=1,model_reloaded=False,final_status='READY',
            before={k:before[k] for k in ('completed_jobs','failed_jobs','allocated_bytes')},
            after={k:worker[k] for k in ('completed_jobs','failed_jobs','allocated_bytes')}),
        jobs=[dict(job=j['job'],status=j['status'],sha256=digest(raw/'jobs/done'/j['job'])) for j in (job,audit)],
        online_decode_run=False,default_promoted=False,exporter_sha256=digest(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=export(args.raw);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],summary=result['component']['summary']),indent=2))
