"""Verify and sanitize the staged numerical, lifecycle and live HTTP gates."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    def read(name):return json.loads((args.raw/name).read_text())
    numerical,lifecycle,http=map(read,('numerical.json','lifecycle.json','http.json'))
    live,job,cpu,package=map(read,('live-start.json','lifecycle-job.json','cpu-tests.json','package.json'))
    startup,resident=map(read,('startup.json','resident-status.json'))
    assert startup['status']=='READY' and startup['load_count']==startup['model_ready_events']==1
    assert startup['loaded_layers']==48
    assert startup['pid']==job['worker_pid']==resident['pid'] and resident['load_count']==1
    assert resident['last_job']=='901-engine-v02-resume-v3.json'
    assert resident['allocated_bytes']==lifecycle['after_previous_cleanup']['allocated_bytes']
    assert numerical['status']=='FAIL_ENGINE_V02_LONG'
    assert numerical['error']=='AssertionError: could not construct bounded long chat prompt'
    assert lifecycle['status']=='PASS_ENGINE_V02_LONG' and http['status']=='PASS_ENGINE_V02_HTTP'
    assert http['client_sha256']==hashlib.sha256((Path(__file__).parent/'k4batch_engine_http.py').read_bytes()).hexdigest()
    assert len(http['retained_client_failures'])==2
    for failure,name in zip(http['retained_client_failures'],('http-first.json','http-timeout.json'),strict=True):
        assert failure['raw_sha256']==hashlib.sha256((args.raw/name).read_bytes()).hexdigest()
        assert failure['error']==read(name)['error']
    assert lifecycle['resume_of_sha256']==hashlib.sha256((args.raw/'numerical.json').read_bytes()).hexdigest()
    assert job['status']=='PASS' and job['load_count']==1
    assert numerical['worker_pid']==lifecycle['worker_pid']==job['worker_pid']
    assert numerical['target_object_id']==lifecycle['target_object_id']
    assert numerical['draft_object_id']==lifecycle['draft_object_id']
    assert len(numerical['component_cases'])==80
    assert all(all(row[k] for k in ('logits_exact','features_exact','state_exact')) for row in numerical['component_cases'])
    assert numerical['coincident_populated_positions']==[32704]*4
    assert len(numerical['matches'])==4
    assert all(all(value for key,value in row.items() if key!='index') for row in numerical['matches'])
    assert lifecycle['resumed_lifecycle']==dict(actual_active_requests=4,survivors_exact=3,
        cancellation=True,same_id_reuse=True,stale_cancel_rejected=True)
    manifest=read('live-manifest.json');checkout=Path(__file__).resolve().parents[1]
    production={name:digest for name,digest in manifest.items() if name.startswith('spark_infer_q/')}
    for name,digest in production.items():
        assert numerical['source_sha256'][name]==lifecycle['source_sha256'][name]==digest,name
        assert hashlib.sha256((checkout/name).read_bytes()).hexdigest()==digest,name
    assert live['source_sha256']==manifest
    assert cpu['status']=='PASS' and cpu['passed']+cpu['skipped']==cpu['tests']
    assert cpu['log_sha256']==hashlib.sha256((args.raw/'cpu-tests.log').read_bytes()).hexdigest()
    assert package['status']=='PASS_PACKAGE' and all(package[key] for key in
        ('production_source_matches_live','third_party_notice_packaged','installed_cli_outside_checkout'))
    samples=numerical['memory_samples']
    assert all(row['peak_reserved_bytes']<=88*1024**3 and row['host_available_bytes']>=8*1024**3 for row in samples)
    long,queued=http['runs'];assert long['name']=='four-long-uninstrumented'
    assert long['observed_peaks']['active']==4
    peaks=queued['observed_peaks']
    assert peaks['active']==4 and peaks['streams']==8 and peaks['full_running_with_four_waiting']
    assert queued['overflow_429']
    for index,row in enumerate(long['requests']):
        control=numerical['target_only_controls'][str(index)]
        assert row['text']==control['text'] and row['usage']==dict(prompt_tokens=32704,
            completion_tokens=len(control['tokens']),total_tokens=32704+len(control['tokens']))
    assert http['cancellation']['actual_active']==4 and http['cancellation']['survivors_exact']==3
    assert http['disconnect_recovered']
    health=http['after'];assert health['status']=='ready'
    assert health['active_requests']==health['open_streams']==health['pending_submissions']==0
    assert health['metrics']['worker_failures_total']==health['metrics']['requests_failed_total']==0
    assert health['capabilities']['graph_captures']==64 and not any(health['capabilities']['graph_fallbacks'])
    numerical_http=next(row for row in numerical['runs'] if row['name']=='four-long-http')
    result=dict(status='PASS_ENGINE_V02',date='2026-11-02',cpu=cpu,packaging=package,
        scope=dict(running_requests=4,max_context_including_output=32768,waiting_stream_slots=4,
            target_kv='bf16',gpu_batch=False,http_async_owner=True,decoding='greedy',
            draft='DFlash2 W8 MLP / Q4 head / 2040-row history / private projection cache',
            costs_recalibrated=False,production_stress_tested=False),
        validation=dict(component_cases=numerical['component_cases'],matches=numerical['matches'],
            coincident_populated_probe_positions=numerical['coincident_populated_positions'],
            http_max_live_states=numerical_http['concurrency']['max_live_states'],
            http_max_populated_32k_states=numerical_http['concurrency']['max_populated_states'],
            over_capacity_rejected=numerical['over_capacity_rejected'],
            resumed_lifecycle=lifecycle['resumed_lifecycle'],load_count=1,
            loaded_target_layers=48,model_ready_events=1,
            numerical_model_reloaded=False,retained_fixture_failure=lifecycle['retained_fixture_failure']),
        memory=dict(peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in samples),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in samples),
            minimum_host_available_bytes=min(r['host_available_bytes'] for r in samples),
            before_numerical=numerical['before'],after_previous_cleanup=lifecycle['after_previous_cleanup'],
            post_lifecycle_allocated_bytes=resident['allocated_bytes'],
            startup_admission=live['worker']['capabilities']['memory_admission']),
        live_http=http,production_source_sha256=production,
        raw_sha256={name:hashlib.sha256((args.raw/name).read_bytes()).hexdigest() for name in
            ('numerical.json','lifecycle.json','http.json','live-start.json','lifecycle-job.json',
             'startup.json','resident-status.json')})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],cpu=cpu['tests'],component_cases=80,
        http_requests=health['metrics']['requests_completed_total'],live_wall_seconds=long['wall_seconds'])))


if __name__=='__main__':main()
