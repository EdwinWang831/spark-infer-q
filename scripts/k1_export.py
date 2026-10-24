"""Export checked, path-free K1 resident evidence; retain failures separately."""
import argparse
import hashlib
import json
from pathlib import Path


CASES = {'single_eos_rows1', 'single_eos_rows8', 'two_requests_interleaved',
         'max_tokens_one_pending', 'cancel_waiting_reuse', 'cancel_after_output_reuse',
         'invalid_and_full_admission'}


def export(raw, checkout):
    preflight = json.loads((raw/'jobs/done/092-k1-native-preflight-v2.json').read_text())
    gpu = json.loads((raw/'jobs/done/094-k1-native-gpu-v3.json').read_text())
    http = json.loads((raw/'jobs/done/095-k1-http-gpu.json').read_text())
    status = json.loads((raw/'worker-status.json').read_text())
    result = gpu['result']
    assert preflight['status'] == gpu['status'] == http['status'] == 'PASS'
    assert http['result']['status'] == 'PASS_K1_HTTP_GPU'
    assert {x['case'] for x in http['result']['cases']} == {'stream','cancel'}
    assert len(http['result']['cases']) == 2
    assert all(x['status']=='PASS' and x['complete_sse'] for x in http['result']['cases'])
    assert http['result']['same_target_object'] and http['result']['same_draft_object']
    assert not http['result']['model_reloaded']
    assert http['result']['full_logit_rows'] > 0 and http['result']['live_state_checks'] > 0
    assert http['result']['native']['core_sha256'] == result['native']['core_sha256']
    assert result['status'] == 'PASS_K1_NATIVE_GPU'
    assert preflight['result']['repeated_open'] and preflight['result']['borrowed_group_preserved']
    assert status['status'] == 'READY'
    assert preflight['worker_pid'] == gpu['worker_pid'] == http['worker_pid'] == status['pid']
    assert preflight['load_count'] == gpu['load_count'] == http['load_count'] == status['load_count'] == 1
    assert result['same_target_object'] and result['same_draft_object'] and not result['model_reloaded']
    assert {x['case'] for x in result['cases']} == CASES
    assert len(result['cases']) == len(CASES)
    assert result['full_logit_rows'] > 0 and result['live_state_checks'] > 0
    for case in result['cases']:
        assert case['status'] == 'PASS'
        assert case['free_logical_blocks'] == case['expected_free_logical_blocks']
        assert case['completions']
    for name, sha in result['source_sha256'].items():
        assert hashlib.sha256((checkout/name).read_bytes()).hexdigest() == sha, name
    for name, sha in http['result']['source_sha256'].items():
        assert hashlib.sha256((checkout/name).read_bytes()).hexdigest() == sha, name
    failures = []
    for name in ('091-k1-native-gpu.json', '093-k1-native-gpu-v2.json'):
        failed = json.loads((raw/'jobs/done'/name).read_text())
        assert failed['status'] == 'FAIL'
        failures.append(dict(job=name, status='FAIL', error=failed['error']))
    return dict(status='PASS_K1_NATIVE_GPU', native=result['native'],
        cases=result['cases'], http=http['result'], full_logit_rows=result['full_logit_rows'],
        live_state_checks=result['live_state_checks'], source_sha256=result['source_sha256'],
        same_worker=True, load_count=1, same_target_object=True, same_draft_object=True,
        final_worker_status='READY', repeated_native_open=True, borrowed_group_preserved=True,
        model_reloaded=False, prior_failures=failures, limits=result['limits'],
        raw_job_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((raw/'jobs/done').glob('*k1*.json'))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = export(args.raw, Path(__file__).resolve().parents[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], full_logit_rows=result['full_logit_rows'],
                          live_state_checks=result['live_state_checks'])))


if __name__ == '__main__': main()
