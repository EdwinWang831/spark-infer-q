"""Verify 128K numeric, capacity, resident HTTP and packaging evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    def read(name):
        return json.loads((args.raw/name).read_text())

    initial, dedup, capacity = map(read, ('capacity-initial.json', 'capacity-dedup.json', 'capacity.json'))
    index, context, http = map(read, ('index.json', 'context.json', 'http.json'))
    live, resident, cpu, package = map(read, ('live-start.json', 'resident-status.json', 'cpu.json', 'package.json'))
    failed = read('index-failed.json')
    context_failed = read('context-failed.json')
    assert initial['status'] == dedup['status'] == capacity['status'] == 'PASS_CAPACITY_PROBE'
    assert not initial['profiles'][0]['admission']['admission_fits']
    assert not initial['profiles'][1]['admission']['admission_fits']
    assert not dedup['profiles'][0]['admission']['admission_fits']
    assert capacity['profiles'][0]['admission']['admission_fits']
    assert failed['status'] == 'FAIL_ENGINE_INDEX'
    assert failed['error'] == 'AssertionError: could not construct bounded long chat prompt'
    assert context_failed['status'] == 'FAIL_ENGINE_CONTEXT'
    assert context_failed['error'].startswith('OutOfMemoryError: CUDA out of memory.')
    assert context_failed['coincident_populated_positions'] == [131008]*4
    assert len(context_failed['numeric']) == 16
    assert index['status'] == 'PASS_ENGINE_INDEX'
    assert index['full_prefix']['position'] == 131008
    assert index['full_prefix']['chunks'] == 256
    assert index['full_prefix']['all_chunk_logits_and_features_exact']
    assert index['full_prefix']['all_valid_state_exact']
    assert len(index['prefill']) == 4 and len(index['rollback']) == 3
    assert all(row['exact'] for row in index['prefill']+index['rollback']+index['numeric'])
    assert len(index['numeric']) == 16
    assert context['status'] == 'PASS_ENGINE_CONTEXT' and context['capacity'] == 131072
    assert context['index_gate_sha256'] == hashlib.sha256((args.raw/'index.json').read_bytes()).hexdigest()
    assert context['coincident_populated_positions'] == [131008]*4
    assert len(context['numeric']) == 16 and all(row['exact'] for row in context['numeric'])
    assert len(context['matches']) == 4
    assert all(all(value for key,value in row.items() if key != 'index') for row in context['matches'])
    assert context['over_capacity_rejected'] and context['cancellation_and_reuse']
    assert context['http']['population']['max_live_states'] == 4
    assert context['capabilities']['index_ring_rows'] == 2048
    assert context['capabilities']['prefill_chunk_rows'] == 512
    assert context['capabilities']['memory_admission']['admission']['admission_fits']
    assert http['status'] == 'PASS_ENGINE_V04_HTTP' and http['expected_prefill_rows'] == 512
    assert http['after']['capabilities']['context_capacity'] == 131072
    assert http['after']['capabilities']['target_kv_dtype'] == 'bf16'
    assert http['after']['capabilities']['gpu_batch'] is False
    assert http['runs'][0]['observed_peaks']['active'] == 4
    assert http['runs'][1]['overflow_429']
    assert http['runs'][1]['observed_peaks']['full_running_with_four_waiting']
    assert http['cancellation']['survivors_exact'] == 3 and http['disconnect_recovered']
    assert http['after']['status'] == 'ready' and http['after']['metrics']['worker_failures_total'] == 0
    assert all(http['after'][key] == 0 for key in ('active_requests','open_streams','pending_submissions'))
    for index_number, response in enumerate(http['runs'][0]['requests']):
        control = context['target_only_controls'][str(index_number)]
        assert len(control['tokens']) == 64 and response['text'] == control['text']
        assert response['usage'] == dict(prompt_tokens=131008, completion_tokens=64, total_tokens=131072)
    for name in ('capacity-job.json','index-job.json','context-job.json'):
        job = read(name)
        assert job['status'] == 'PASS' and job['worker_pid'] == resident['pid'] and job['load_count'] == 1
    assert resident['load_count'] == 1 and resident['pid'] == context['worker_pid'] == index['worker_pid']
    assert index['target_object_id'] == context['target_object_id']
    assert index['draft_object_id'] == context['draft_object_id']
    for result in (index, context, live):
        assert result['model_reloaded'] is False
    production = {name:digest for name,digest in context['source_sha256'].items()
                  if name.startswith('spark_infer_q/')}
    root = Path(__file__).resolve().parents[1]
    for name, digest in production.items():
        assert index['source_sha256'][name] == live['source_sha256'][name] == digest
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    samples = index['memory_samples']+context['memory_samples']
    assert all(row['peak_reserved_bytes'] <= 88*1024**3 and row['host_available_bytes'] >= 8*1024**3
               for row in samples)
    assert cpu['status'] == 'PASS' and cpu['tests'] == cpu['passed']+cpu['skipped']
    assert cpu['log_sha256'] == hashlib.sha256((args.raw/'cpu.log').read_bytes()).hexdigest()
    assert package['status'] == 'PASS_PACKAGE' and package['version'] == '0.4.0'
    assert package['production_source_matches_live'] and package['installed_cli_outside_checkout']
    assert package['third_party_notice_packaged']
    decode = context['isolated_dynamic']
    isolated_tokens = sum(row['decode_tokens'] for row in decode.values())
    isolated_seconds = sum(row['decode_seconds'] for row in decode.values())
    reclaimed = dedup['before_fused_weight_views']['allocated_bytes']-dedup['after_fused_weight_views']['allocated_bytes']
    evidence = dict(status='PASS_ENGINE_V04', date='2026-11-03', cpu=cpu, packaging=package,
        scope=dict(running_requests=4,total_tokens_per_request=131072,input_tokens_per_request=131008,
            output_tokens_per_request=64,target_kv='bf16',index_ring_rows=2048,prefill_rows=512,
            dynamic_draft=True,gpu_batch=False,cost_calibration_changed=False,sustained_stress_tested=False),
        residency=dict(load_count=1,same_model_process=True,same_target_and_draft_objects=True),
        capacity=dict(initial_profiles=initial['profiles'],after_exact_weight_views=dedup['profiles'],
            final_profile=capacity['profiles'][0],duplicate_draft_bytes_reclaimed=reclaimed),
        full_index_reference=dict(prefill=index['prefill'],rollback=index['rollback'],
            full_prefix=index['full_prefix'],numeric=index['numeric']),
        context_128k=dict(coincident_populated_positions=context['coincident_populated_positions'],
            numeric=context['numeric'],matches=context['matches'],target_only_controls=context['target_only_controls'],
            http=context['http'],cancellation_and_reuse=True,over_capacity_rejected=True),
        isolated_dynamic_decode=dict(tokens=isolated_tokens,seconds=isolated_seconds,tokens_per_second=isolated_tokens/isolated_seconds,
            runs={key:{name:row[name] for name in ('decode_seconds','decode_tokens','decode_scope')}
                  for key,row in decode.items()},
            scope='serial isolated adapter loops, excluding prefill, first token, native Scheduler and HTTP'),
        memory=dict(peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in context['memory_samples']),
            peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in context['memory_samples']),
            minimum_host_available_bytes=min(row['host_available_bytes'] for row in context['memory_samples']),
            startup_admission=context['capabilities']['memory_admission']),
        live_http=http,production_source_sha256=production,
        retained_failures=[{key:failure[key] for key in ('status','error','phase','source_sha256')}
                           for failure in (failed,context_failed)],
        raw_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(args.raw.glob('*.json'))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=evidence['status'],capacity=131072,
                         live_cohort_seconds=http['runs'][0]['wall_seconds'])))


if __name__ == '__main__':
    main()
