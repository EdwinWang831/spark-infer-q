"""Recompute the paired 128K prefill result and verify deployed profile evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    def read(name):return json.loads((args.raw/name).read_text())
    study,reference,http=map(read,('study.json','reference.json','http.json'))
    live,resident,cpu,package=map(read,('live-start.json','resident-status.json','cpu.json','package.json'))
    failure=read('collector-failed.json');job=read('study-job.json')
    assert study['status'] in ('PASS_ENGINE_COMPACT','REJECT_ENGINE_COMPACT_COST')
    assert reference['status']=='PASS_ENGINE_CONTEXT'
    assert study['reference_gate_sha256']==hashlib.sha256((args.raw/'reference.json').read_bytes()).hexdigest()
    assert study['worker_pid']==reference['worker_pid']==resident['pid']==job['worker_pid']
    assert study['target_object_id']==reference['target_object_id'] and study['draft_object_id']==reference['draft_object_id']
    assert resident['load_count']==job['load_count']==1 and job['status']=='PASS'
    assert not study['model_reloaded'] and not live['model_reloaded']
    assert len(study['numeric'])==3
    for row in study['numeric']:
        assert all(row[key] for key in ('logits_exact','features_exact','state_exact'))
        assert len(row['windows'])==8 and all(item['exact'] for item in row['windows']+row['feature_suffixes'])
        assert row['mirrored_bytes']==104448000 and row['linear_bytes']==52224000
    a,b,c,d=study['runs'];assert [row['arm'] for row in study['runs']]==['baseline','candidate','candidate','baseline']
    reduction=1-(b['wall_seconds']+c['wall_seconds'])/(a['wall_seconds']+d['wall_seconds'])
    both=all(y['wall_seconds']<x['wall_seconds'] for x,y in ((a,b),(d,c)))
    ttft=all(y['last_ttft_seconds']<=1.02*x['last_ttft_seconds'] for x,y in ((a,b),(d,c)))
    promoted=reduction>=.05 and both and ttft
    assert abs(reduction-study['promotion']['wall_reduction'])<1e-12
    assert study['promotion']==dict(wall_reduction=reduction,both_pairs_faster=both,last_ttft_within_2_percent=ttft,passed=promoted)
    assert (study['status']=='PASS_ENGINE_COMPACT')==promoted
    for row in study['runs']:
        assert all(row[key]==a[key] for key in ('outputs','final','trace'))
        for key,values in row['outputs'].items():
            gold=reference['target_only_controls'][key]
            assert len(values)==64 and values==gold['tokens']
            assert row['final'][key]['state']==gold['state'] and row['final'][key]['feature_sha256']==gold['feature_sha256']
        expected=104448000 if row['arm']=='baseline' else 52224000
        assert len(row['feature_storage_bytes'])==4 and all(value==expected for value in row['feature_storage_bytes'].values())
        allocated=next(sample for sample in study['memory_samples'] if sample['phase']==row['name']+':allocated')
        # The retained context rows and five layers of projected K/V are fixed
        # 2048-row BF16 buffers, totaling 52428800 bytes per request.
        forecast=allocated['allocated_bytes']+4*(expected+52428800)+1024**3
        assert row['admission']['forecast_peak_bytes']==forecast<=88*1024**3
        assert row['admission']['conservative_host_remaining_bytes']==allocated['host_available_bytes']-(forecast-allocated['allocated_bytes'])
        assert row['admission']['conservative_host_remaining_bytes']>=8*1024**3
        assert row['admission']['admission_fits'] and row['max_live_states']==4
        assert row['head_kernels_unchanged'] and row['draft_captures_added']==0
        assert row['capabilities']['graph_captures']==64 and not any(row['capabilities']['graph_fallbacks'])
        assert not row['capabilities']['head_fallbacks']
        if row['arm']=='candidate':assert row['expert_calls']>0
    for row in study['memory_samples']:
        assert row['peak_reserved_bytes']<=88*1024**3 and row['host_available_bytes']>=8*1024**3
    assert failure['status']=='FAIL_ENGINE_COMPACT' and failure['error'].startswith('RuntimeError: long-context memory admission refused:')
    assert not failure['runs'] and len(failure['warmups'])==2 and len(failure['numeric'])==3
    assert http['status']=='PASS_ENGINE_V04_HTTP'
    expected_chunk,expected_layout=(1024,'linear') if promoted else (512,'mirrored')
    assert http['expected_prefill_rows']==expected_chunk
    caps=http['after']['capabilities']
    assert caps['context_capacity']==131072 and caps['max_running_requests']==4
    assert caps['target_kv_dtype']=='bf16' and caps['gpu_batch'] is False
    assert caps['prefill_chunk_rows']==expected_chunk and caps['feature_history_layout']==expected_layout
    assert caps['service_version']=='0.4.1' and live['metadata']['profile']['feature_history_layout']==expected_layout
    if promoted:assert caps['prefill_expert_sha256']==study['expert_binary_sha256']
    assert caps['memory_admission']['admission']['admission_fits']
    assert http['runs'][0]['observed_peaks']['active']==4
    assert http['runs'][1]['overflow_429'] and http['runs'][1]['observed_peaks']['full_running_with_four_waiting']
    for index,response in enumerate(http['runs'][0]['requests']):
        assert response['text']==reference['target_only_controls'][str(index)]['text']
        assert response['usage']==dict(prompt_tokens=131008,completion_tokens=64,total_tokens=131072)
    assert http['cancellation']['survivors_exact']==3 and http['disconnect_recovered']
    assert http['after']['status']=='ready' and http['after']['metrics']['worker_failures_total']==0
    assert all(http['after'][key]==0 for key in ('active_requests','open_streams','pending_submissions'))
    root=Path(__file__).resolve().parents[1]
    production={name:digest for name,digest in study['source_sha256'].items() if name.startswith('spark_infer_q/')}
    for name,digest in production.items():
        assert live['source_sha256'][name]==digest==hashlib.sha256((root/name).read_bytes()).hexdigest(),name
    assert cpu['status']=='PASS' and cpu['tests']==cpu['passed']+cpu['skipped']
    assert cpu['log_sha256']==hashlib.sha256((args.raw/'cpu.log').read_bytes()).hexdigest()
    assert package['status']=='PASS_PACKAGE' and package['version']=='0.4.1'
    assert all(package[key] for key in ('production_source_matches_live','installed_cli_outside_checkout','third_party_notice_packaged'))
    evidence=dict(status='PASS_ENGINE_V05' if promoted else 'REJECT_ENGINE_V05_COST',date='2026-11-03',cpu=cpu,packaging=package,
        scope=dict(running_requests=4,input_tokens_per_request=131008,output_tokens_per_request=64,target_kv='bf16',gpu_batch=False,
            baseline=dict(prefill_rows=512,experts='original',feature_layout='mirrored'),
            candidate=dict(prefill_rows=1024,expert_sha256=study['expert_binary_sha256'],feature_layout='linear'),
            promoted=promoted,sustained_stress_tested=False,predictor_and_costs_retrained=False),
        residency=dict(load_count=1,same_process=True,same_target_and_draft_objects=True),
        paired=dict(promotion=study['promotion'],baseline_mean_seconds=(a['wall_seconds']+d['wall_seconds'])/2,
            candidate_mean_seconds=(b['wall_seconds']+c['wall_seconds'])/2,
            numeric=study['numeric'],runs=study['runs'],warmups=study['warmups'],
            baseline_mean_phase_ms={key:(a['phase_ms'][key]+d['phase_ms'][key])/2 for key in a['phase_ms']},
            candidate_mean_phase_ms={key:(b['phase_ms'][key]+c['phase_ms'][key])/2 for key in a['phase_ms']}),
        memory=dict(feature_bytes_saved_per_request=52224000,feature_bytes_saved_four_requests=208896000,
            peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in study['memory_samples']),
            peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in study['memory_samples']),
            minimum_host_available_bytes=min(row['host_available_bytes'] for row in study['memory_samples'])),
        live_http=http,production_source_sha256=production,
        retained_collector_failure=dict(status=failure['status'],error=failure['error'],source_sha256=failure['source_sha256']),
        raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.raw.glob('*.json'))})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(evidence,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=evidence['status'],wall_reduction=reduction,live_seconds=http['runs'][0]['wall_seconds'])))


if __name__=='__main__':main()
