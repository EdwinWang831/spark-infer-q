"""Validate and sanitize v0.1 session/cold-start evidence; no GPU access."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    data = json.loads((args.raw/'gpu-and-startup.json').read_text())
    cpu = json.loads((args.raw/'cpu-tests.json').read_text())
    package = json.loads((args.raw/'package.json').read_text())
    assert package['status'] == 'PASS_PACKAGE'
    assert all(package[key] is True for key in ('production_source_matches_boot',
        'third_party_notice_packaged','installed_cli_outside_checkout'))
    assert cpu['status'] == 'PASS' and cpu['passed']+cpu['skipped'] == cpu['tests']
    assert hashlib.sha256((args.raw/'cpu-tests.log').read_bytes()).hexdigest() == cpu['log_sha256']
    first, stable = data['session_v1'], data['session_v2']
    for report in (first, stable):
        assert report['status'] == 'PASS_ENGINE_V01_SESSION'
        assert report['model_reloaded'] is False and report['session_identity_preserved'] is True
        assert all(report['counts'][key] > 0 for key in ('logit_rows','feature_rows','state_checks','prompt_checks'))
        assert [case['mode'] for case in report['cases']] == ['none','adaptive']
        assert report['cases'][0]['outputs'] == report['cases'][1]['outputs']
        for case in report['cases']:
            assert all(case[key] is True for key in ('outputs_exact','cancellation','disconnect','recovery'))
            assert case['capabilities']['joint_forwards'] > 0
            assert case['metrics']['worker_failures_total'] == 0
            assert case['metrics']['requests_failed_total'] == 0
    assert stable['allocated_before'] == stable['allocated_after']
    assert first['allocated_after'] == stable['allocated_before']
    cold = data['cold_start']
    assert cold['status'] == 'PASS_ENGINE_V01_COLD_START'
    assert cold['model_loaded_once'] and len(data['ready_records']) == 1
    assert data['loaded_layer_events'].count(48) == 1 and max(data['loaded_layer_events']) == 48
    assert cold['health']['status'] == 'ready'
    assert cold['health']['metrics']['requests_completed_total'] == 10
    assert cold['health']['metrics']['admission_rejected_total'] == 1
    assert cold['health']['capabilities']['joint_forwards'] > 0
    assert cold['health']['active_requests'] == cold['health']['open_streams'] == 0
    checkout = Path(__file__).resolve().parents[1]
    for name, digest in data['boot_manifest'].items():
        if name.startswith('spark_infer_q/'):
            assert hashlib.sha256((checkout/name).read_bytes()).hexdigest() == digest, name
            if name != 'spark_infer_q/cli.py':
                assert stable['source_sha256'][name] == digest, name
    ready = data['ready_records'][0]
    public_cold = {key: cold[key] for key in ('status','pid','model_loaded_once','completed','health')}
    public_cold.update(native=ready['metadata']['native'],
        profile={key: ready['metadata']['profile'][key] for key in
                 ('model_sha256','cost_sha256','w8_source_sha256','w8_binary_sha256')},
        checkpoint_revision=data['checkpoint_revision'], loaded_target_layers=48,
        ready_events=len(data['ready_records']))
    evidence = dict(status='PASS_ENGINE_V01', date='2026-11-02', cpu=cpu, packaging=package,
        session_first=first, session_stable=stable, cold_start=public_cold,
        boot_source_sha256=data['boot_manifest'],
        scope=dict(running_requests=2, max_context_including_output=1536,
            target_kv='bf16', decoding='greedy', draft='sequential DFlash2, W8 MLP, Q4 head',
            throughput_measured=False, long_duration_stress_tested=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(status=evidence['status'], cpu=cpu['tests'],
        stable_counts=stable['counts'], cold_pid=cold['pid'])))


if __name__ == '__main__':
    main()
