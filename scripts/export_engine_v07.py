"""Recompute four-request prompt-batch gates from retained GPU/HTTP evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw

    def read(name):
        return json.loads((raw/name).read_text())

    inventory, numeric, study, reference = map(read, ('memory.json', 'numeric.json', 'study.json', 'reference.json'))
    job, resident, live, http, cpu = map(read,
        ('study-job.json', 'resident-status.json', 'service-status.json', 'http.json', 'cpu.json'))
    assert inventory['status'] == 'PASS_BATCH_MEMORY' and inventory['native_mode_roundtrip']
    assert numeric['status'] == 'PASS_BATCH_NUMERIC'
    assert study['status'] == 'PASS_BATCH_STUDY'
    assert reference['status'] == 'PASS_ENGINE_CONTEXT' and job['status'] == 'PASS'
    assert study['reference_gate_sha256'] == digest(raw/'reference.json')
    assert study['numeric_gate_sha256'] == digest(raw/'numeric.json')
    assert inventory['control_binary_sha256'] == study['prompt_control_sha256']
    assert inventory['source_sha256']['scripts/k4batch_prompt_control.cpp'] == study['source_sha256']['scripts/k4batch_prompt_control.cpp']
    assert resident['load_count'] == job['load_count'] == 1
    assert job['worker_pid'] == resident['pid'] == study['worker_pid'] == numeric['worker_pid'] == inventory['worker_pid']
    for result in (inventory, numeric, study):
        assert not result['model_reloaded']
        assert all(result[key] == reference[key] for key in ('target_object_id', 'draft_object_id'))
    assert study['inputs'] == reference['inputs'] and len(study['inputs']) == 4
    assert [row['request_rows'] for row in numeric['numeric']] == [
        [1, 17, 1023, 1024], [1024]*4, [257, 511, 769, 1024]]
    assert all(row['exact'] and len(row['requests']) == 4 for row in numeric['numeric'])
    assert all(len(request['followup']) == 8 for case in numeric['numeric'] for request in case['requests'])
    totals = {arm: sum(row['wall_seconds'] for row in numeric['timings'] if row['arm'] == arm)
        for arm in ('baseline', 'candidate')}
    assert len(numeric['timings']) == 8 and totals == numeric['totals']
    assert abs(numeric['reduction']-(1-totals['candidate']/totals['baseline'])) < 1e-12
    a, b, c, d = study['runs']
    assert [row['arm'] for row in study['runs']] == ['baseline', 'candidate', 'candidate', 'baseline']
    pairs = ((a, b), (d, c))
    reduction = 1-(b['wall_seconds']+c['wall_seconds'])/(a['wall_seconds']+d['wall_seconds'])
    both = all(y['wall_seconds'] < x['wall_seconds'] for x, y in pairs)
    ttft = all(y['last_ttft_seconds'] <= 1.02*x['last_ttft_seconds'] for x, y in pairs)
    gaps = all(max(y['max_output_gaps_seconds'].values()) <= 1.05*max(x['max_output_gaps_seconds'].values())
        for x, y in pairs)
    promoted = reduction >= .10 and both and ttft and gaps
    assert study['promotion'] == dict(wall_reduction=reduction, both_pairs_faster=both,
        last_ttft_within_2_percent=ttft, output_gaps_within_5_percent=gaps, passed=promoted)
    assert (study['status'] == 'PASS_BATCH_STUDY') == promoted
    for row in study['runs']:
        assert all(row[field] == a[field] for field in ('outputs', 'final', 'trace'))
        for key, tokens in row['outputs'].items():
            gold = reference['target_only_controls'][key]
            assert len(tokens) == 64 and tokens == gold['tokens']
            assert row['final'][key]['state'] == gold['state']
            assert row['final'][key]['feature_sha256'] == gold['feature_sha256']
        assert row['head_kernels_unchanged'] and row['draft_captures_added'] == 0
        assert row['capabilities']['graph_captures'] == 64 and not any(row['capabilities']['graph_fallbacks'])
        assert row['capabilities']['head_fallbacks'] == 0
        assert row['admission']['admission_fits'] and row['admission']['forecast_peak_bytes'] <= 88*1024**3
        assert all(value == 52224000 for value in row['feature_storage_bytes'].values())
        prompt = [plan for plan in row['plans'] if plan['prompt']]
        assert sum(plan['rows'] for plan in prompt) == 4*131008
        if row['arm'] == 'candidate':
            assert len(prompt) == 128 and all(plan['requests'] == 4 for plan in prompt)
            assert all(max(plan['request_rows']) <= 1024 for plan in prompt)
            assert row['prefill_capabilities']['prefill_gpu_batch'] is True
            assert row['prefill_capabilities']['prefill_workspace_rows'] == 4096
        else:
            assert len(prompt) == 512 and max(plan['requests'] for plan in prompt) == 2
            assert sum(plan['requests'] == 2 for plan in prompt) == 3
    for i in (0, 2):
        left, right = study['warmups'][i:i+2]
        assert all(left[field] == right[field] for field in ('outputs', 'final', 'trace'))
    left, right = study['lifecycle']
    assert left['outputs']['0'] == right['outputs']['0'] == []
    for key in ('1', '2', '3'):
        assert all(left[field][key] == right[field][key] for field in ('outputs', 'final', 'trace'))
    samples = numeric['memory']+study['memory']
    for row in samples:
        assert row['reserved_bytes'] <= 88*1024**3 and row['host_available_bytes'] >= 8*1024**3
    assert live['status'] == 'READY' and not live['model_reloaded']
    caps = live['worker']['capabilities']
    assert caps['max_running_requests'] == 4 and caps['context_capacity'] == 131072
    assert caps['gpu_batch'] is False and caps['target_kv_dtype'] == 'bf16'
    assert caps['prefill_chunk_rows'] == 1024 and caps['feature_history_layout'] == 'linear'
    assert caps['memory_admission']['admission']['admission_fits']
    assert promoted and caps['prefill_gpu_batch'] is True and caps['prefill_batch_requests'] == 4
    assert http['status'] == 'PASS_ENGINE_V04_HTTP' and http['expected_prefill_batch_requests'] == 4
    assert http['cancellation']['survivors_exact'] == 3 and http['disconnect_recovered']
    assert http['runs'][1]['overflow_429']
    after = http['after']
    assert after['status'] == 'ready' and after['metrics']['worker_failures_total'] == 0
    assert all(after[key] == 0 for key in ('active_requests', 'open_streams', 'pending_submissions'))
    assert cpu['status'] == 'PASS' and cpu['tests'] == cpu['passed']+cpu['skipped']
    assert cpu['log_sha256'] == digest(raw/'cpu.log')
    root = Path(__file__).resolve().parents[1]
    sources = {name: sha for name, sha in study['source_sha256'].items() if name.startswith('spark_infer_q/')}
    for name, sha in sources.items():
        assert digest(root/name) == sha, name
    package = read('package.json')
    assert package['status'] == 'PASS' and package['python_sources_verified'] == len(sources)
    assert all(package[key] for key in ('isolated_cli_version', 'isolated_cli_help', 'third_party_notice_exact'))
    for label, expected in (('study', study['source_sha256']), ('live', live['source_sha256'])):
        archived = raw/(label+'-source')
        assert read(label+'-source/manifest.json') == expected
        for name, sha in expected.items():
            assert digest(archived/name) == sha, name
    failure = read('study-v1-failure.json')
    assert failure['status'] == 'FAIL_BATCH_STUDY' and not failure['runs']
    evidence = dict(status='PASS_ENGINE_V07', date='2026-11-03',
        scope=dict(running_requests=4, input_tokens_per_request=131008, output_tokens_per_request=64,
            target_kv='bf16', prefill_batch_requests=4, per_request_chunk_rows=1024,
            workspace_rows=4096, decode_gpu_batch=False, feature_layout='linear',
            promoted=promoted, sustained_stress_tested=False),
        residency=dict(load_count=1, same_process=True, same_target_and_draft_objects=True),
        workspace=dict(baseline_storage_bytes=inventory['cuda_storage_bytes'],
            candidate_layout=numeric['workspace_layout'], native_headers=inventory['native_header_sha256'],
            control_binary_sha256=inventory['control_binary_sha256']),
        component=dict(numeric=numeric['numeric'], timings=numeric['timings'], reduction=numeric['reduction']),
        paired=dict(promotion=study['promotion'], baseline_mean_seconds=(a['wall_seconds']+d['wall_seconds'])/2,
            candidate_mean_seconds=(b['wall_seconds']+c['wall_seconds'])/2, runs=study['runs'],
            warmups=study['warmups'], cancellation=study['lifecycle']),
        memory=dict(peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in samples),
            peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in samples),
            minimum_host_available_bytes=min(row['host_available_bytes'] for row in samples)),
        retained_failure=dict(status=failure['status'], error=failure['error'], formal_runs=0),
        service=dict(prefill_gpu_batch=caps.get('prefill_gpu_batch', False),
            source_sha256={name: sha for name, sha in live['source_sha256'].items() if name.startswith('spark_infer_q/')},
            http=http, capabilities=caps), cpu=cpu, package=package, candidate_source_sha256=sources,
        source_archive_sha256={p.name: digest(p) for p in sorted(raw.glob('*-source.tar.gz'))},
        raw_sha256={str(p.relative_to(raw)): digest(p) for p in sorted(raw.rglob('*.json'))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(status=evidence['status'], wall_reduction=reduction, promoted=promoted)))


if __name__ == '__main__': main()
