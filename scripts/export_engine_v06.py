"""Verify 128K profiling, rejected QSA candidates and restored-service evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from e37_accounting import analyze_trace, family


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kernel_family(name):
    if name.startswith('fn_tile4_'):
        return 'routed_experts'
    if name in ('_scores', '_select', '_select_tiles', '_pool', '_chunks', '_merge', '_prep', '_attn_gate'):
        return 'attention'
    return family(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw

    def read(name):
        return json.loads((raw/name).read_text())

    profile = read('profile/engine-hotspots-result.json')
    assert profile['status'] == 'PASS_ENGINE_HOTSPOTS'
    assert len(profile['profiles']) == 6 and len(profile['events']) == 6
    assert len(profile['checks']) == 2
    assert all(all(row[k] for k in ('output_exact', 'state_exact', 'features_exact')) for row in profile['checks'])
    profiles = []
    for row in profile['profiles']:
        path = raw/'profile'/row['trace']
        assert digest(path) == row['trace_sha256']
        trace = json.loads(path.read_text())
        assert analyze_trace(trace, row['root']) == row['cupti']
        profiles.append(dict(index=row['index'], position=row['position'], rows=row['rows'],
            trace_sha256=row['trace_sha256'], cupti=analyze_trace(trace, row['root'], classifier=kernel_family)))

    hc = read('probes/hc/engine-hc-probe-result.json')
    assert hc['status'] == 'PASS_ENGINE_HC_PROBE' and len(hc['numeric']) == 3
    assert all(row['exact'] for row in hc['numeric']) and len(hc['timings']) == 16
    hc_totals = {arm: sum(row['wall_seconds'] for row in hc['timings'] if row['arm'] == arm)
        for arm in ('baseline', 'candidate')}
    assert hc_totals == hc['totals']
    assert abs(hc['wall_reduction']-(1-hc_totals['candidate']/hc_totals['baseline'])) < 1e-12
    rejected = read('probes/qsa-v1/engine-qsa-probe-result.json')
    assert rejected['status'] == 'FAIL_ENGINE_QSA_PROBE' and rejected['error'] == 'AssertionError: scores:17:4095'
    probes = []
    for version in (2, 3, 4, 5):
        probe = read(f'probes/qsa-v{version}/engine-qsa-probe-result.json')
        assert probe['status'] == 'PASS_ENGINE_QSA_PROBE'
        assert len(probe['numeric']) == 5 and all(row['exact'] for row in probe['numeric'])
        totals = {arm: sum(row['gpu_ms'] for row in probe['timings'] if row['arm'] == arm)
            for arm in ('baseline', 'candidate')}
        assert totals == probe['totals_gpu_ms']
        assert abs(probe['reduction']-(1-totals['candidate']/totals['baseline'])) < 1e-12
        if version >= 4:
            assert all(row['exact'] for row in probe['selector_checks'])
        probes.append(dict(version=version, numeric=probe['numeric'], timings=probe['timings'],
            reduction=probe['reduction'], selector_checks=probe.get('selector_checks', []),
            selector_timings=probe.get('selector_timings', []), source_sha256=probe['source_sha256']['spark_infer_q/prefill_qsa.py']))

    study, reference, job, resident = map(read, ('study.json', 'reference.json', 'study-job.json', 'resident-status.json'))
    assert study['status'] in ('PASS_ENGINE_QSA', 'REJECT_ENGINE_QSA_COST')
    assert reference['status'] == 'PASS_ENGINE_CONTEXT' and job['status'] == 'PASS'
    assert study['reference_gate_sha256'] == digest(raw/'reference.json')
    assert study['worker_pid'] == profile['worker_pid'] == reference['worker_pid'] == job['worker_pid'] == resident['pid']
    assert resident['load_count'] == job['load_count'] == 1
    assert not study['model_reloaded'] and not profile['model_reloaded']
    for key in ('target_object_id', 'draft_object_id'):
        assert study[key] == profile[key] == reference[key]
    assert study['inputs'] == reference['inputs']
    assert [row['input_tokens'] for row in study['numeric']] == [1024, 1025, 4097, 32769]
    for row in study['numeric']:
        assert all(row[k] for k in ('logits_exact', 'features_exact', 'state_exact'))
        assert len(row['windows']) == 8 and all(item['exact'] for item in row['windows']+row['feature_suffixes'])
        assert row['baseline_bytes'] == row['candidate_bytes'] == 52224000
    a, b, c, d = study['runs']
    assert [row['arm'] for row in study['runs']] == ['baseline', 'candidate', 'candidate', 'baseline']
    pairs = ((a, b), (d, c))
    reduction = 1-(b['wall_seconds']+c['wall_seconds'])/(a['wall_seconds']+d['wall_seconds'])
    both = all(y['wall_seconds'] < x['wall_seconds'] for x, y in pairs)
    ttft = all(y['last_ttft_seconds'] <= 1.02*x['last_ttft_seconds'] for x, y in pairs)
    gaps = all(max(y['max_output_gaps_seconds'].values()) <= 1.05*max(x['max_output_gaps_seconds'].values()) for x, y in pairs)
    promoted = reduction >= .03 and both and ttft and gaps
    assert study['promotion'] == dict(wall_reduction=reduction, both_pairs_faster=both,
        last_ttft_within_2_percent=ttft, output_gaps_within_5_percent=gaps, passed=promoted)
    assert (study['status'] == 'PASS_ENGINE_QSA') == promoted
    for row in study['runs']:
        assert all(row[k] == a[k] for k in ('outputs', 'final', 'trace'))
        for key, tokens in row['outputs'].items():
            gold = reference['target_only_controls'][key]
            assert len(tokens) == 64 and tokens == gold['tokens']
            assert row['final'][key]['state'] == gold['state']
            assert row['final'][key]['feature_sha256'] == gold['feature_sha256']
        assert row['head_kernels_unchanged'] and row['draft_captures_added'] == 0
        assert row['capabilities']['graph_captures'] == 64 and not any(row['capabilities']['graph_fallbacks'])
        assert row['capabilities']['head_fallbacks'] == 0 and row['max_live_states'] == 4
        assert all(value == 52224000 for value in row['feature_storage_bytes'].values())
        assert row['prefill_capabilities']['prefill_qsa'] == ('reference' if row['arm'] == 'baseline' else 'grouped')
        sample = next(s for s in study['memory_samples'] if s['phase'] == row['name']+':allocated')
        forecast = sample['allocated_bytes']+4*(52224000+52428800)+1024**3
        assert row['admission']['forecast_peak_bytes'] == forecast <= 88*1024**3
        assert row['admission']['admission_fits'] and row['admission']['conservative_host_remaining_bytes'] >= 8*1024**3
    for row in study['memory_samples']:
        assert row['peak_reserved_bytes'] <= 88*1024**3 and row['host_available_bytes'] >= 8*1024**3

    # The complete-request candidate missed its predeclared cost gate.
    # The following selector screen therefore remains a component experiment.
    assert not promoted
    register = read('register.json')
    assert register['status'] == 'PASS_ENGINE_QSA_PROBE' and register['register_selector']
    assert len(register['numeric']) == 8 and all(row['exact'] for row in register['numeric'])
    register_totals = {arm: sum(row['gpu_ms'] for row in register['timings'] if row['arm'] == arm)
        for arm in ('baseline', 'candidate')}
    assert register_totals == register['totals_gpu_ms']
    assert abs(register['reduction']-(1-register_totals['candidate']/register_totals['baseline'])) < 1e-12
    assert register['reduction'] < 0
    assert register['target_object_id'] == study['target_object_id']
    assert register['draft_object_id'] == study['draft_object_id'] and not register['model_reloaded']
    for label, result in (('study', study), ('register', register)):
        manifest = read(label+'-manifest.json')
        assert manifest == result['source_sha256']
        for name in manifest:
            path = raw/(label+'-source')/name
            if path.exists():
                assert digest(path) == manifest[name], name

    restored, live, cpu = map(read, ('restoration.json', 'restored-service-status.json', 'cpu.json'))
    assert restored['status'] == 'PASS_ENGINE_V06_RESTORATION'
    assert restored['output_exact'] and restored['usage_exact']
    assert restored['load_count'] == 1 and restored['worker_pid'] == study['worker_pid']
    assert restored['usage'] == dict(prompt_tokens=221, completion_tokens=64, total_tokens=285)
    assert not live['model_reloaded']
    caps = restored['after']['capabilities']
    assert caps['service_version'] == '0.4.1'
    assert caps['context_capacity'] == 131072 and caps['max_running_requests'] == 4
    assert caps['target_kv_dtype'] == 'bf16' and caps['gpu_batch'] is False
    assert caps['prefill_chunk_rows'] == 1024 and caps['feature_history_layout'] == 'linear'
    assert caps['prefill_expert_sha256'] == study['expert_binary_sha256']
    assert caps['memory_admission']['admission']['admission_fits']
    assert restored['after']['status'] == 'ready' and restored['after']['metrics']['worker_failures_total'] == 0
    assert all(restored['after'][k] == 0 for k in ('active_requests', 'open_streams', 'pending_submissions'))
    assert cpu['status'] == 'PASS' and cpu['tests'] == cpu['passed']+cpu['skipped']
    assert cpu['log_sha256'] == digest(raw/'cpu.log')
    root = Path(__file__).resolve().parents[1]
    restored_commit = 'c3f0103496f65ec24a2f8973cc28802a885c5b73'
    production = {n: h for n, h in live['source_sha256'].items() if n.startswith('spark_infer_q/')}
    assert len(production) == 81
    for name, deployed_digest in production.items():
        sealed = subprocess.check_output(['git', 'show', restored_commit+':'+name], cwd=root)
        assert hashlib.sha256(sealed).hexdigest() == deployed_digest, name
    candidate_sources = {name: study['source_sha256'][name] for name in (
        'spark_infer_q/bounded_index.py', 'spark_infer_q/prefill.py',
        'spark_infer_q/prefill_tensorfold.py', 'spark_infer_q/prefill_qsa.py')}
    for name, tested_digest in candidate_sources.items():
        assert digest(root/name) == tested_digest, name
    assert probes[-1]['source_sha256'] == candidate_sources['spark_infer_q/prefill_qsa.py']
    evidence = dict(status='REJECT_ENGINE_V06_COST', date='2026-11-03',
        scope=dict(running_requests=4, input_tokens_per_request=131008, output_tokens_per_request=64,
            target_kv='bf16', gpu_batch=False, prefill_rows=1024, feature_layout='linear',
            baseline_qsa='reference', candidate_qsa='grouped_scores_and_radix', promoted=promoted,
            sustained_stress_tested=False, predictor_and_costs_retrained=False),
        residency=dict(load_count=1, same_process=True, same_target_and_draft_objects=True),
        profiles=profiles, events=profile['events'], component_probes=probes,
        retained_candidates=dict(hc_prompt_wall_reduction=hc['wall_reduction'], hc_promoted=False,
            automatic_broadcast=dict(status=rejected['status'], error=rejected['error']),
            register_selector=dict(numeric=register['numeric'], timings=register['timings'],
                reduction=register['reduction'], full_model_paired_tested=False, promoted=False)),
        paired=dict(promotion=study['promotion'], baseline_mean_seconds=(a['wall_seconds']+d['wall_seconds'])/2,
            candidate_mean_seconds=(b['wall_seconds']+c['wall_seconds'])/2, numeric=study['numeric'],
            runs=study['runs'], warmups=study['warmups'],
            baseline_mean_phase_ms={k: (a['phase_ms'][k]+d['phase_ms'][k])/2 for k in a['phase_ms']},
            candidate_mean_phase_ms={k: (b['phase_ms'][k]+c['phase_ms'][k])/2 for k in a['phase_ms']}),
        memory=dict(peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in study['memory_samples']),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in study['memory_samples']),
            minimum_host_available_bytes=min(r['host_available_bytes'] for r in study['memory_samples'])),
        restoration=dict(status=restored['status'], service_version='0.4.1',
            source_commit=restored_commit, output_exact=True, usage_exact=True,
            http_seconds=restored['http_seconds'], usage=restored['usage'],
            health_after=restored['after'], production_source_sha256=production,
            scope='one short HTTP replay against frozen v0.4.1 response'),
        cpu=cpu, candidate_source_sha256=candidate_sources,
        raw_sha256={str(p.relative_to(raw)): digest(p) for p in sorted(raw.rglob('*.json'))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(status=evidence['status'], wall_reduction=reduction,
        restored_http_seconds=restored['http_seconds'])))


if __name__ == '__main__':
    main()
