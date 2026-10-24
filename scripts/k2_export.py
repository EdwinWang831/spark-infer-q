"""Check and export K2 native/GPU/online resident evidence without private paths."""
import argparse
import hashlib
import json
from pathlib import Path


def checked_job(raw,name,status):
    job=json.loads((raw/'jobs/done'/name).read_text())
    assert job['status']=='PASS' and job['result']['status']==status
    return job


def export(raw,checkout):
    cpu=checked_job(raw,'150-k2-dynamic-only-cpu-v1.json','PASS_K2_NATIVE_CPU')
    gpu=checked_job(raw,'151-k2-dynamic-only-gpu-v1.json','PASS_K2_NATIVE_GPU')
    online=checked_job(raw,'152-k2-dynamic-only-online-v1.json','PASS_K2_ONLINE')
    audit=checked_job(raw,'153-k2-dynamic-only-audit-v1.json','PASS_K2_RESIDENT_AUDIT')
    worker=json.loads((raw/'worker-status.json').read_text())
    assert worker['status']=='READY'
    assert len({x['worker_pid'] for x in (cpu,gpu,online,audit)} | {worker['pid']})==1
    assert all(x['load_count']==1 for x in (cpu,gpu,online,audit)) and worker['load_count']==1
    c,g,o=(x['result'] for x in (cpu,gpu,online))
    assert len(c['cases'])==15 and len(g['cases'])==18
    assert all(x['status']=='PASS' for x in c['cases']+g['cases']+o['http_cases'])
    assert g['full_logit_rows']>0 and g['live_state_checks']>0
    assert all(x['same_target_object'] and x['same_draft_object'] and not x['model_reloaded'] for x in (g,o))
    assert {x['case'] for x in o['http_cases']}=={'stream','cancel'}
    assert len(o['timings'])==27 and o['serial_state_checks']==36
    assert all(x['exact_tokens'] and x['exact_state'] for x in o['timings'])
    for result in (g,o):
        for name,sha in result['source_sha256'].items():
            assert hashlib.sha256((checkout/name).read_bytes()).hexdigest()==sha,name
    assert len(g['legacy_postdraft_widths'])==3
    assert sum(x.get('legacy_budget_widths_exact',False) for x in g['cases'])==3
    failed=json.loads((raw/'jobs/done/105-k2-native-cpu-v1.json').read_text())
    assert failed['status']=='FAIL'
    resident={k:v for k,v in audit['result'].items() if k not in ('target_object_id','draft_object_id')}
    assert resident['native_group_released'] and resident['native_environment_restored']
    return dict(status='PASS_K2_INTEGRATED',cpu=c,gpu=g,online=o,resident=resident,
        same_worker=True,load_count=1,
        historical_native_lagged_evidence='2026-10-23-native-dflash.json',
        historical_postdraft_evidence='2026-10-23-postdraft-e17.json',
        final_worker_status='READY',model_reloaded=False,
        prior_failures=[dict(job=failed['job'],status='FAIL',error=failed['error'],
                            cause='two-request test configured with one request slot; corrected before GPU')],
        raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((raw/'jobs/done').glob('*k2*.json'))})


def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    report=export(a.raw,Path(__file__).resolve().parents[1])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],rows=report['gpu']['full_logit_rows'],
                         state_checks=report['gpu']['live_state_checks'],online_timings=len(report['online']['timings']))))


if __name__=='__main__':main()
