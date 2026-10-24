"""Validate sealed E36 evidence and export a bounded local numerical diagnosis."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark_infer_q.prefill_evidence import validate_local_control
from scripts.e36_candidate_evidence import summarize_candidate


EXECUTED = {
    'preflight-v1': '240-e36-preflight-v1.json',
    'diagnose-v2': '242-e36-diagnose-v2.json',
    'localize-v1': '244-e36-localize-v1.json',
    'candidate-v1': '246-e36-candidate-v1.json',
    'audit-v1': '248-e36-audit-v1.json',
}
CANCELLED = '241-e36-diagnose-v1.json'
CASE_LENGTHS = dict(historical4=4, english=64, chinese=64, code=512, mixed=512)
FLAGS = ('full_model_fp32_reference', 'task_quality_evaluated',
         'performance_evaluated', 'default_promoted')
_HASH = re.compile(r'[0-9a-fA-F]{64}')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _number(value, name, *, upper=None):
    _require(type(value) in (int, float), f'{name}: numeric measurement required')
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    _require(valid and (upper is None or value <= upper), f'{name}: out of finite range')


def _finite_tree(value):
    if isinstance(value, float):
        _require(math.isfinite(value), 'nonfinite evidence')
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _metrics(value, name):
    _require(isinstance(value, dict) and {'relative_l2', 'max_abs'} <= set(value)
             and set(value) <= {'relative_l2', 'max_abs', 'top1_agreement'},
             f'{name}: error metrics required')
    for key, number in value.items():
        _number(number, f'{name}.{key}', upper=1 if key == 'top1_agreement' else None)


def _unique(items, expected, key, description):
    _require(isinstance(items, list) and len(items) == len(expected), f'missing/repeated {description}')
    keys = [key(item) for item in items]
    _require(len(set(keys)) == len(keys) and set(keys) == set(expected), f'missing/repeated {description}')


def summarize_diagnosis(result):
    """Check the complete bounded experiment and return counts, never accuracy claims."""
    _finite_tree(result)
    _require(result.get('status') == 'PASS_E36_LOCAL_DIAGNOSIS', 'diagnosis did not pass')
    _require(all(result.get(field) is False for field in FLAGS), 'unsupported model/quality/performance promotion')
    _require(result.get('reference_cpu_tests') == dict(run=4, skipped=0, passed=True),
             'four unskipped reference equation tests required')
    cases = result.get('cases')
    _unique(cases, CASE_LENGTHS, lambda item: item['name'], 'diagnostic cases')
    for case in cases:
        name = case['name']
        _require(case.get('length') == CASE_LENGTHS[name] and _hash(case.get('token_sha256')),
                 f'{name}: invalid input identity')
        exact = case.get('serial_rows8')
        if case['length'] <= 64:
            _require(isinstance(exact, dict)
                     and set(exact) == {'residual_exact', 'picks_exact', 'last_logits_exact'}
                     and all(value is True for value in exact.values()), f'{name}: short serial control failed')
        else:
            _require(exact is None, f'{name}: long serial control was not executed')
        layers = case.get('layers')
        _unique(layers, range(48), lambda item: item['layer'], f'{name} layers')
        layers = sorted(layers, key=lambda item: item['layer'])
        for layer in layers:
            _metrics(layer['residual'], f'{name}.layer{layer["layer"]}')
            _require(type(layer['residual_exact']) is bool, 'residual exactness must be explicit')
            _number(layer['expert_set_agreement'], 'expert set agreement', upper=1)
        first = next((item['layer'] for item in layers if not item['residual_exact']), None)
        routed = next((item['layer'] for item in layers if item['expert_set_agreement'] < 1), None)
        _require(case.get('first_different_layer') == first
                 and case.get('first_route_difference') == routed, 'first divergence disagrees with layer evidence')
        _metrics(case['last_logits'], f'{name}.last_logits')
        _require(set(case['last_argmax']) == {'rows8', 'prefill'}
                 and all(type(value) is int and value >= 0 for value in case['last_argmax'].values()),
                 'invalid final argmax identifiers')
    controls = result.get('local_controls')
    expected = {(name, layer) for name in CASE_LENGTHS for layer in (0, 6)}
    _unique(controls, expected, lambda item: (item['name'], item['layer']), 'local controls')
    for item in controls:
        projection, recurrence = item['projection'], item['recurrence']
        for control in (projection['control'], recurrence['control']):
            validate_local_control(control)
        _require(projection['control']['component'] == 'mxfp8_gdn_projection'
                 and recurrence['control']['component'] == 'gdn_recurrent_final_state',
                 'projection and recurrence reference components differ')
        for field in ('serial_matches_rounded_reference', 'prefill_matches_rounded_reference'):
            _require(type(projection[field]) is bool, f'{field}: explicit boolean required')
        _metrics(projection['serial_prefill'], 'projection.serial_prefill')
        _require(all(_hash(projection[field]) for field in ('stored_weight_sha256', 'dense_weight_sha256')),
                 'projection weight fingerprints missing')
        _require(recurrence.get('common_scratch_exact') is True
                 and recurrence.get('checked_front_fields') == ['k', 'v', 'decay', 'beta'],
                 'only k/v/decay/beta front alignment was checked')
        _require(recurrence.get('raw_decode_y_exposed') is False
                 and recurrence.get('initial_state') == 'zero'
                 and recurrence.get('shared_front_and_back') is True,
                 'recurrence isolation boundaries differ')
        _require(recurrence.get('prefill_y_dtype') == 'torch.bfloat16'
                 and recurrence.get('gated_reference_boundary') == 'FP32 recurrence -> BF16 readout -> existing back',
                 'gated reference must disclose the BF16 readout and shared back')
        _require(_hash(recurrence.get('front_sha256'))
                 and recurrence['front_sha256'] == recurrence['control']['input_sha256']
                 and _hash(recurrence.get('projection_conv_initial_sha256')), 'recurrence input fingerprints differ')
        _require(type(recurrence.get('decode_prefill_gated_exact')) is bool, 'missing gated exactness')
        _number(recurrence['executed_eps'], 'executed_eps')
        _require(recurrence['executed_eps'] > 0, 'positive normalization epsilon required')
        for field in ('gated_output_vs_fp32', 'state_vs_fp64'):
            _require(set(recurrence[field]) == {'serial', 'prefill'}, f'{field}: both modes required')
            for mode, metrics in recurrence[field].items():
                _metrics(metrics, f'{field}.{mode}')
        for field in ('prefill_raw_y_vs_fp32', 'fp32_vs_fp64_state', 'fp32_vs_fp64_y'):
            _metrics(recurrence[field], field)
    return dict(case_count=5, layer_comparisons=240, local_control_pairs=10,
                short_serial_controls=3, long_serial_controls_not_run=2,
                full_model_fp32_reference=False, task_quality_evaluated=False,
                performance_evaluated=False, default_promoted=False)


def _manifest(raw, version):
    folder = raw/'stages'/version
    path = folder/'manifest.json'
    manifest = json.loads(path.read_text())
    _require(isinstance(manifest, dict) and manifest, f'{version}: empty manifest')
    for name, checksum in manifest.items():
        _require(isinstance(name, str) and not Path(name).is_absolute()
                 and folder.resolve() in (folder/name).resolve().parents, 'invalid source manifest path')
        _require(_hash(checksum) and digest(folder/name) == checksum, f'{version}: source hash differs: {name}')
    wrapper = raw/('e36_job_'+version.replace('-', '_')+'.py')
    _require(digest(wrapper) == manifest.get('scripts/e36_prefill_probe.py'), f'{version}: wrapper hash differs')
    return manifest, dict(manifest_sha256=digest(path), wrapper_sha256=digest(wrapper))


def summarize_localization(result, diagnosis):
    _require(result.get('status')=='PASS_E36_LOCALIZATION' and result.get('layer')==0,
             'layer-zero localization did not pass')
    _require(all(result.get(k) is False for k in FLAGS), 'localization exceeded bounded scope')
    _require(result.get('upstream_functions_unchanged') is True
             and result.get('single_layer_state_only') is True, 'private single-layer scope missing')
    _require(result.get('factor_bits')==dict(A=1,B=2,C=4),'factor mask differs')
    _unique(result.get('cases'),CASE_LENGTHS,lambda c:c['name'],'localization cases')
    components={'hc_readout','projection_P','gdn_gout','gdn_branch','moe_input','router_logits',
                'router_pick','router_weights','moe_slots_y','residual_after_writeback',
                'final_gdn_state','final_conv_state'}
    diagnostic={c['name']:c for c in diagnosis['cases']}
    aligned=[]
    for c in result['cases']:
        _require(c['length']==CASE_LENGTHS[c['name']] and
                 c['token_sha256']==diagnostic[c['name']]['token_sha256'],'localization input differs')
        _require(set(c['baseline_components'])==components,'missing localization baseline component')
        _require(all(_hash(v['sha256']) for v in c['baseline_components'].values()),'baseline fingerprint missing')
        _unique(c.get('variants'),range(8),lambda v:v['counterfactual_mask'],'factorial variants')
        for v in c['variants']:
            mask=v['counterfactual_mask']
            _require(v['counterfactual']==dict(A=bool(mask&1),B=bool(mask&2),C=bool(mask&4)),
                     'counterfactual bits differ')
            _require(set(v['component_vs_rows8'])==components and
                     set(v['actual_components_sha256'])==components,'missing localization component')
            for name,value in v['component_vs_rows8'].items():
                _require(all(type(value[k]) is bool for k in ('same_dtype','exact','values_equal')),
                         'component exactness must be explicit')
                for k in ('relative_l2','max_abs'):_number(value[k],k)
                _require(_hash(v['actual_components_sha256'][name]),'component fingerprint missing')
                _require(not value['exact'] or value['relative_l2']==value['max_abs']==0,
                         'exact component has nonzero error')
            _number(v['router_set_agreement'],'router set agreement',upper=1)
            _require(v['counters']['forced_decode_mm_calls']==(v['counters']['mxfp8_mm_calls'] if mask&1 else 0)
                     and v['counters']['replaced_prefill_chains']==int(bool(mask&2)), 'factor did not execute')
            _require(v['moe_slot_dtype']==('torch.float32' if mask&4 else 'torch.bfloat16'),
                     'MoE policy did not execute')
        original=next(v for v in c['variants'] if v['counterfactual_mask']==0)['component_vs_rows8']['residual_after_writeback']
        layer0=diagnostic[c['name']]['layers'][0]
        _require(original['exact']==layer0['residual_exact'] and
                 all(math.isclose(original[k],layer0['residual'][k],rel_tol=1e-12,abs_tol=1e-15)
                     for k in ('relative_l2','max_abs')), 'localization did not reproduce layer-zero baseline')
        full=next(v for v in c['variants'] if v['counterfactual_mask']==7)
        if all(v['exact'] for v in full['component_vs_rows8'].values()):aligned.append(c['name'])
    return dict(cases=5,variants=40,fully_aligned_mask7_cases=aligned)


def _input_metadata(inputs):
    _unique(inputs, CASE_LENGTHS, lambda item: item['name'], 'prefill inputs')
    for item in inputs:
        _require(item.get('length') == CASE_LENGTHS[item['name']], 'prefill input length differs')
        ids = item.get('ids')
        _require(isinstance(ids, list) and len(ids) == item['length']
                 and all(type(token) is int and token >= 0 for token in ids), 'invalid frozen token input')
        actual = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        _require(item.get('token_sha256') == actual, 'token fingerprint differs')
        if item['name'] != 'historical4':
            _require(_hash(item.get('text_sha256')), 'text fingerprint missing')
    _require(next(item['ids'] for item in inputs if item['name'] == 'historical4') == [100, 101, 102, 103],
             'historical input differs')
    return [{key: item[key] for key in ('name', 'length', 'token_sha256', 'text_sha256') if key in item}
            for item in inputs]


def _public(value):
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items()
                if key not in ('target_object_id', 'draft_object_id', 'source_root', 'ids')}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


def export(raw):
    raw = Path(raw)
    worker = json.loads((raw/'worker-status.json').read_text())
    # Other workspaces may immediately claim the shared queue after our audit.
    # Their RUNNING state is not evidence that our buffers were left in flight.
    following_job=(worker.get('status')=='RUNNING' and isinstance(worker.get('job'),str)
        and worker['job'] not in set(EXECUTED.values())|{CANCELLED}
        and isinstance(worker.get('last_job'),str))
    _require((worker.get('status') == 'READY' or following_job) and worker.get('load_count') == 1,
             'resident worker must be READY or serving a separate queued job with one model load')
    manifests, fingerprints = {}, {}
    for version in ('preflight-v1', 'diagnose-v1', 'diagnose-v2', 'localize-v1','candidate-v1', 'audit-v1'):
        manifests[version], fingerprints[version] = _manifest(raw, version)
    results, job_fingerprints = {}, []
    identity = None
    for version, name in EXECUTED.items():
        path = raw/'jobs/done'/name
        job = json.loads(path.read_text())
        _require(job.get('job') == name and job.get('status') == 'PASS'
                 and job.get('worker_pid') == worker.get('pid') and job.get('load_count') == 1,
                 f'{name}: worker, load count or job status differs')
        result = job['result']
        _require(result.get('source_sha256') == manifests[version], f'{name}: result source differs from stage')
        _require(result.get('model_reloaded') is False and result.get('tf32_disabled') is True,
                 f'{name}: residency/TF32 contract differs')
        objects = tuple(result[key] for key in ('target_object_id', 'draft_object_id'))
        _require(all(type(value) is int and value > 0 for value in objects), 'invalid model identities')
        if identity is None:
            identity = objects
        _require(objects == identity, 'model objects changed across jobs')
        results[version] = result
        job_fingerprints.append(dict(job=name, status='PASS', sha256=digest(path),
                                     stage=version, **fingerprints[version]))
    cancelled_path = raw/'jobs/cancelled'/CANCELLED
    cancelled = json.loads(cancelled_path.read_text())
    _require(not any((raw/'jobs'/state/CANCELLED).exists() for state in ('done', 'running', 'pending')),
             'cancelled diagnostic has a duplicate execution record')
    _require(cancelled.get('module') == 'e36_job_diagnose_v1'
             and cancelled.get('sha256') == fingerprints['diagnose-v1']['wrapper_sha256'],
             'cancelled diagnostic wrapper differs')
    preflight, diagnosis, localization, candidate, audit = (
        results[key] for key in ('preflight-v1', 'diagnose-v2', 'localize-v1','candidate-v1', 'audit-v1'))
    _require(preflight.get('status') == 'PASS_E36_PREFLIGHT', 'preflight did not pass')
    _require(preflight['inputs'] == diagnosis['inputs'], 'preflight and diagnosis inputs differ')
    _require(localization.get('status') == 'PASS_E36_LOCALIZATION'
             and localization.get('default_promoted') is False, 'localization did not pass its bounded scope')
    _require(localization.get('inputs') == preflight['inputs'], 'localization inputs differ from preflight')
    _finite_tree(localization)
    inputs = _input_metadata(preflight['inputs'])
    summary = summarize_diagnosis(diagnosis)
    local_summary=summarize_localization(localization,diagnosis)
    candidate_summary=summarize_candidate(candidate)
    _require(candidate['inputs']==preflight['inputs'],'candidate base inputs changed')
    _require(candidate['candidate_inputs'][:5]==preflight['inputs'],'candidate did not preserve base cases')
    cases = {case['name']: case for case in diagnosis['cases']}
    _require(all(cases[item['name']]['token_sha256'] == item['token_sha256'] for item in inputs),
             'diagnostic case fingerprint differs from frozen input')
    _require(audit.get('status') == 'PASS_E36_AUDIT'
             and audit.get('frozen_sources_verified') is True
             and audit.get('diagnose_job') == EXECUTED['diagnose-v2']
             and audit.get('candidate_job') == EXECUTED['candidate-v1'], 'final resource/source audit did not pass')
    baseline = diagnosis['allocated_before']
    _number(baseline, 'diagnostic baseline')
    _number(audit['allocated_bytes'], 'audit allocated bytes')
    _require(audit['baseline_bytes'] == baseline and audit['allocated_bytes'] <= baseline,
             'post-return allocation did not return to baseline')
    for field in ('original_forward_restored', 'native_group_released'):
        _require(audit.get(field) is True, f'{field} audit failed')
    _require(isinstance(audit.get('extensions_sha256'), dict) and audit['extensions_sha256']
             and all(_hash(value) for value in audit['extensions_sha256'].values()), 'extension fingerprints missing')
    _require(preflight.get('external_source_sha256') == diagnosis.get('external_source_sha256'),
             'external source identity changed')
    output = dict(status='PASS_E36_PREFILL_EVIDENCE', inputs=inputs, summary=summary,
                  localization_summary=local_summary,candidate_summary=candidate_summary,
                  diagnosis=_public(diagnosis), localization=_public(localization),
                  candidate=_public(candidate),audit=_public(audit),
                  residency=dict(same_worker=True, same_model_objects=True, load_count=1,
                                 model_reloaded=False, final_status=worker['status'],
                                 other_job_running=bool(following_job),
                                 baseline_bytes=baseline, allocated_bytes=audit['allocated_bytes']),
                  stage_sources=manifests, stage_fingerprints=fingerprints, jobs=job_fingerprints,
                  cancelled_not_executed=dict(job=CANCELLED, sha256=digest(cancelled_path), stage='diagnose-v1',
                                             **fingerprints['diagnose-v1']),
                  exporter_sha256=digest(Path(__file__)), **{field: False for field in FLAGS})
    output['diagnosis']['inputs'] = inputs
    output['localization']['inputs'] = inputs
    output['performance_evaluated']=candidate['performance_evaluated']
    output['performance_scope']='prompt_only_token_API'
    encoded = json.dumps(output, allow_nan=False)
    _require(not any(value in encoded for value in ('/workspace/', '/Users/', 'target_object_id', 'draft_object_id')),
             'private path or model identity in public output')
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
