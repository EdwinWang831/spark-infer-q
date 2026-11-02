"""Verify sealed E37 sources, paired measurements, traces and post-return audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.e36_export import _finite_tree, _hash, _public, _require, digest
from scripts.e37_accounting import analyze_trace, summarize_timings


JOBS = {'measure-v1': '270-e37-measure-v1.json', 'audit-v1': '271-e37-audit-v1.json'}
SEED = '246-e36-candidate-v1.json'
SEED_SHA = '542d558988acb4c6a91c8b6dc8a73dbf567004036cf3ec973dadb16c0de19853'
CASES = dict(code=512, english=64, mixed=512, new_json=128)


def summarize_measurement(result, seed):
    _finite_tree(result)
    _require(result.get('status') == 'PASS_E37_PREFILL_PROFILE', 'measurement failed')
    _require(result.get('default_promoted') is False and result.get('tf32_disabled') is True
             and result.get('upstream_functions_unchanged') is True
             and result.get('model_reloaded') is False
             and result.get('timing_scope') == 'prompt_only_token_API', 'scope/restoration differs')
    _require(result.get('seed_job') == SEED and result.get('seed_sha256') == SEED_SHA,
             'frozen seed differs')
    inputs = result['inputs']
    _require([(c['name'], c['length']) for c in inputs] == list(CASES.items()), 'input order/length differs')
    for c in inputs:
        _require(c == next(x for x in seed['candidate_inputs'] if x['name'] == c['name']),
                 'inputs differ from E36')
        _require(len(c['ids']) == c['length'] and c['token_sha256'] == hashlib.sha256(
            json.dumps(c['ids'], separators=(',', ':')).encode()).hexdigest(), 'token hash differs')
    checks = result['checks']
    _require(len(checks) == 4 and [c['name'] for c in checks] == list(CASES), 'missing/repeated checks')
    for c, inp in zip(checks, inputs):
        gold = {r['output_sha256'] for r in seed['timings'] if r['name'] == c['name']}
        _require(len(gold) == 1 and c['output_sha256'] in gold and _hash(c['output_sha256']),
                 'gold output hash differs')
        _require(c['length'] == inp['length'] and c['token_sha256'] == inp['token_sha256']
                 and all(c.get(k) is True for k in ('full_logits_exact', 'all_features_exact', 'state_exact')),
                 'prompt numerical check failed')
        _require(c['followup'] == [dict(step=i, exact=True) for i in range(8)], 'handoff check failed')
        _require(all(r['output_sha256'] == c['output_sha256'] for r in result['timings']
                     if r['name'] == c['name']), 'timing hash differs from numerical gate')
    summary = summarize_timings(result['timings'], inputs)
    _require(summary == result.get('summary'), 'saved timing summary differs')
    _require(result['scratch'] == dict(prepared_rows=512, storage_bytes=10682368,
        prepared_views=[64, 128, 512], storage_instances=1, fresh_allocations_per_prompt=36),
        'scratch allocation contract differs')
    _require(result.get('pool_calls') == 36*(4+4+24), 'pool path did not execute exactly as planned')
    profiles = result['profiles']
    _require(len(profiles) == 2 and [p['name'] for p in profiles] == ['code', 'mixed'],
             'missing/repeated profiles')
    for p in profiles:
        _require(p.get('exact') is True and p['root'] == 'e37.prompt.'+p['name']
                 and p['trace'] == p['name']+'-trace.json' and _hash(p['trace_sha256']),
                 'diagnostic scope/numerics differ')
        stages = p['exclusive_stage_intervals']
        _require(set(stages) == {'control', 'state_init', 'host_stage', 'embedding', 'hyper_connection',
            'forward_matrix_projection', 'gdn_other', 'moe', 'writeback', 'ple', 'attention_other',
            'features', 'head_and_mixer', 'commit', 'feature_concat', 'token_ready'}, 'missing stages')
        for value in stages.values():
            _require(value['segments'] > 0 and value['host_enqueue_ms'] >= 0
                     and value['gpu_interval_ms'] >= 0, 'invalid stage interval')
    return dict(**summary, prompt_cases=4, full_feature_rows=1216, followup_checks=32,
                timing_runs=48, profile_runs=4)


def export(raw):
    raw = Path(raw)
    worker = json.loads((raw/'worker-status.json').read_text())
    following = worker.get('status') == 'RUNNING' and isinstance(worker.get('job'), str) \
        and worker['job'] not in JOBS.values() and isinstance(worker.get('last_job'), str)
    _require((worker.get('status') == 'READY' or following) and worker.get('load_count') == 1,
             'worker residency differs')
    manifests, records, results = {}, [], {}
    identity = None
    for version, name in JOBS.items():
        folder = raw/'stages'/version
        manifest = json.loads((folder/'manifest.json').read_text())
        _require(bool(manifest), 'empty source manifest')
        for path, checksum in manifest.items():
            _require(not Path(path).is_absolute() and folder.resolve() in (folder/path).resolve().parents,
                     'invalid manifest path')
            _require(_hash(checksum) and digest(folder/path) == checksum, 'source hash differs: '+path)
        wrapper = raw/('e37_job_'+version.replace('-', '_')+'.py')
        _require(digest(wrapper) == manifest['scripts/e37_run.py'], 'wrapper differs')
        job_path = raw/'jobs/done'/name
        job = json.loads(job_path.read_text())
        _require(job.get('job') == name and job.get('status') == 'PASS'
                 and job.get('worker_pid') == worker.get('pid') and job.get('load_count') == 1,
                 'job identity/status differs')
        result = job['result']
        _require(result.get('source_sha256') == manifest and result.get('model_reloaded') is False,
                 'execution source/residency differs')
        objects = tuple(result[k] for k in ('target_object_id', 'draft_object_id'))
        _require(all(type(v) is int and v > 0 for v in objects), 'missing model object identity')
        if identity is None: identity = objects
        _require(objects == identity, 'model objects changed')
        manifests[version], results[version] = manifest, result
        records.append(dict(job=name, sha256=digest(job_path), status='PASS',
            manifest_sha256=digest(folder/'manifest.json'), wrapper_sha256=digest(wrapper)))
    _require(manifests['measure-v1'] == manifests['audit-v1'], 'audit source seal changed')
    seed_path = raw/'frozen'/SEED
    _require(digest(seed_path) == SEED_SHA, 'seed job hash differs')
    seed = json.loads(seed_path.read_text())['result']
    measure, audit = results['measure-v1'], results['audit-v1']
    _require(measure == json.loads((raw/'stages/measure-v1/result.json').read_text()), 'progress/final differs')
    for name in ('scripts/e36_prefill_candidate.py', 'scripts/e36_diagnose.py'):
        _require(manifests['measure-v1'][name] == seed['source_sha256'][name], 'E36 arithmetic changed')
    seal = json.loads((raw/'frozen/e33-online-seal.json').read_text())
    _require(measure['external_source_sha256'] == seal['external_source_sha256'], 'external seal differs')
    for name, checksum in seal['external_source_sha256'].items():
        _require(digest(raw/'external/tensorfold'/name) == checksum, 'external source changed')
    summary = summarize_measurement(measure, seed)
    for p in measure['profiles']:
        path = raw/'stages/measure-v1'/p['trace']
        _require(digest(path) == p['trace_sha256'], 'trace hash differs')
        _require(analyze_trace(json.loads(path.read_text()), p['root']) == p['cupti'], 'trace totals differ')
    _require(audit.get('status') == 'PASS_E37_AUDIT' and audit.get('measurement_sha256') == records[0]['sha256']
             and all(audit.get(k) is True for k in ('frozen_sources_verified', 'original_forward_restored',
                                                   'native_group_released')), 'post-return audit failed')
    _require(audit['baseline_bytes'] == measure['baseline_bytes']
             and 0 < audit['allocated_bytes'] <= audit['baseline_bytes'], 'temporary allocation not released')
    _require(len(audit['extensions_sha256']) == 6 and all(_hash(h) for h in audit['extensions_sha256'].values()),
             'extension fingerprints missing')
    output = dict(status='PASS_E37_PREFILL_EVIDENCE', summary=summary,
        measurement=_public(measure), audit=_public(audit), stage_sources=manifests, jobs=records,
        residency=dict(same_worker=True, same_model_objects=True, load_count=1, model_reloaded=False,
            final_status=worker['status'], other_job_running=bool(following),
            baseline_bytes=audit['baseline_bytes'], allocated_bytes=audit['allocated_bytes']),
        scope='prompt_only_token_API', default_promoted=False, task_quality_evaluated=False,
        exporter_sha256=digest(Path(__file__)))
    for p in output['measurement']['profiles']: p.pop('trace')
    _require(not any(s in json.dumps(output, allow_nan=False) for s in
                     ('/workspace/', '/Users/', 'target_object_id', 'draft_object_id', '"ids"')),
             'private path/object/token IDs in public output')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = export(args.raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], summary=result['summary']), indent=2))
