"""Recompute the paired prefill gain and verify four-request 64K evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    def read(name):return json.loads((args.raw/name).read_text())
    speed,context,http=map(read,('prefill.json','context.json','http.json'))
    cpu,package,live=map(read,('cpu.json','package.json','live-start.json'))
    resident,context_job,prefill_job=map(read,('resident-status.json','context-job.json','prefill-job.json'))
    assert speed['status']=='PASS_ENGINE_PREFILL' and context['status']=='PASS_ENGINE_CONTEXT'
    assert http['status']=='PASS_ENGINE_V03_HTTP'
    assert len(speed['numeric'])==3
    assert all(all(row[k] for k in ('logits_exact','features_exact','state_exact')) and
        len(row['windows'])==8 and all(win['exact'] for win in row['windows']) for row in speed['numeric'])
    a,b,c,d=speed['runs'];assert [row['arm'] for row in speed['runs']]==['baseline','tile1024','tile1024','baseline']
    reduction=1-(b['wall_seconds']+c['wall_seconds'])/(a['wall_seconds']+d['wall_seconds'])
    assert abs(reduction-speed['promotion']['wall_reduction'])<1e-12 and reduction>=.05
    assert all(y['wall_seconds']<x['wall_seconds'] and y['last_ttft_seconds']<=1.02*x['last_ttft_seconds']
               for x,y in ((a,b),(d,c)))
    for row in speed['runs']:
        assert all(row[k]==a[k] for k in ('outputs','final','trace'))
        assert all(len(values)==64 for values in row['outputs'].values())
        assert row['head_kernels_unchanged'] and row['draft_captures_added']==0
        assert row['capabilities']['graph_captures']==64
        assert not any(row['capabilities']['graph_fallbacks'])
        if row['arm']=='tile1024':assert row['expert_calls']>0
    assert context['prefill_gate_sha256']==hashlib.sha256((args.raw/'prefill.json').read_bytes()).hexdigest()
    assert context['worker_pid']==speed['worker_pid']
    assert resident['pid']==context_job['worker_pid']==prefill_job['worker_pid']==context['worker_pid']
    assert resident['load_count']==context_job['load_count']==prefill_job['load_count']==1
    assert context_job['status']==prefill_job['status']=='PASS'
    assert live['model_reloaded'] is False
    assert context['target_object_id']==speed['target_object_id'] and context['draft_object_id']==speed['draft_object_id']
    assert context['coincident_populated_positions']==[65472]*4
    assert len(context['numeric'])==16 and all(row['exact'] for row in context['numeric'])
    assert len(context['matches'])==4 and all(all(v for k,v in row.items() if k!='index') for row in context['matches'])
    assert context['http']['population']['max_live_states']==4
    assert context['over_capacity_rejected'] and context['cancellation_and_reuse']
    for row in (speed,context):
        assert not row['model_reloaded']
        for sample in row['memory_samples']:
            assert sample['peak_reserved_bytes']<=88*1024**3 and sample['host_available_bytes']>=8*1024**3
    # The next source adds the requested capacity bound and corrects mirrored
    # ring admission. All model/execution operators stay byte-identical.
    for name,digest in speed['source_sha256'].items():
        if name.startswith('spark_infer_q/') and name not in ('spark_infer_q/engine_server.py','spark_infer_q/cli.py'):
            assert context['source_sha256'][name]==digest,name
    root=Path(__file__).resolve().parents[1]
    production={name:digest for name,digest in context['source_sha256'].items() if name.startswith('spark_infer_q/')}
    for name,digest in production.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==live['source_sha256'][name]==digest,name
    assert http['expected_prefill_rows']==1024
    assert len(http['runs'])==2 and http['runs'][0]['observed_peaks']['active']==4
    assert http['runs'][1]['overflow_429'] and http['runs'][1]['observed_peaks']['full_running_with_four_waiting']
    for index,row in enumerate(http['runs'][0]['requests']):
        control=context['target_only_controls'][str(index)]
        assert row['text']==control['text']
        assert row['usage']==dict(prompt_tokens=65472,completion_tokens=len(control['tokens']),
                                 total_tokens=65472+len(control['tokens']))
    assert http['cancellation']['survivors_exact']==3 and http['disconnect_recovered']
    assert http['after']['status']=='ready' and http['after']['metrics']['worker_failures_total']==0
    assert http['after']['active_requests']==http['after']['open_streams']==http['after']['pending_submissions']==0
    assert http['after']['capabilities']['context_capacity']==65536
    assert cpu['status']=='PASS' and cpu['tests']==cpu['passed']+cpu['skipped']
    assert cpu['log_sha256']==hashlib.sha256((args.raw/'cpu.log').read_bytes()).hexdigest()
    assert package['status']=='PASS_PACKAGE' and package['production_source_matches_live']
    assert package['installed_cli_outside_checkout'] and package['third_party_notice_packaged']
    baseline_ms={k:(a['phase_ms'][k]+d['phase_ms'][k])/2 for k in a['phase_ms']}
    fast_ms={k:(b['phase_ms'][k]+c['phase_ms'][k])/2 for k in a['phase_ms']}
    evidence=dict(status='PASS_ENGINE_V03',date='2026-11-02',cpu=cpu,packaging=package,
        scope=dict(running_requests=4,total_tokens_per_request=65536,target_kv='bf16',gpu_batch=False,
                   prefill_rows=1024,prefill_experts=speed['expert_binary_sha256'],dynamic_draft=True,
                   cost_calibration_changed=False,production_stress_tested=False),
        residency=dict(load_count=1,same_model_process=True,same_target_and_draft_objects=True),
        prefill_32k=dict(wall_reduction=reduction,baseline_mean_seconds=(a['wall_seconds']+d['wall_seconds'])/2,
            candidate_mean_seconds=(b['wall_seconds']+c['wall_seconds'])/2,
            baseline_mean_phase_ms=baseline_ms,candidate_mean_phase_ms=fast_ms,
            runs=[{k:row[k] for k in ('name','arm','wall_seconds','last_ttft_seconds','max_output_gaps_seconds',
                'phase_ms','expert_calls','expert_fallbacks','head_kernels_unchanged','memory')} for row in speed['runs']],
            numeric=speed['numeric'],tokens_state_features_dynamic_trace_exact=True),
        context_64k=dict(coincident_populated_positions=context['coincident_populated_positions'],
            numeric=context['numeric'],matches=context['matches'],target_only_controls=context['target_only_controls'],
            http=context['http'],cancellation_and_reuse=True,over_capacity_rejected=True),
        memory=dict(peak_allocated_bytes=max(s['peak_allocated_bytes'] for s in context['memory_samples']),
            peak_reserved_bytes=max(s['peak_reserved_bytes'] for s in context['memory_samples']),
            minimum_host_available_bytes=min(s['host_available_bytes'] for s in context['memory_samples']),
            startup_admission=context['capabilities']['memory_admission'],mirrored_feature_ring_charged=True),
        live_http=http,production_source_sha256=production,
        raw_sha256={name:hashlib.sha256((args.raw/name).read_bytes()).hexdigest()
                    for name in ('prefill.json','context.json','http.json','live-start.json',
                                 'resident-status.json','context-job.json','prefill-job.json')})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(evidence,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=evidence['status'],capacity=65536,wall_reduction=reduction,cpu=cpu['tests'])))


if __name__=='__main__':main()
