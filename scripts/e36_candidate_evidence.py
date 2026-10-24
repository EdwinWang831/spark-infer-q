"""Validate the whole-target E36 gate before computing prompt-only throughput."""
import hashlib
import json
import math
import re


CASE_LENGTHS = dict(historical4=4, english=64, chinese=64, code=512,
                    mixed=512, new_json=128, new_zh=384)
MODES = ('rows8', 'candidate')
_HASH = re.compile(r'[0-9a-fA-F]{64}')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_tree(value):
    if isinstance(value, float):
        _require(math.isfinite(value), 'nonfinite candidate evidence')
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _number(value, field, *, positive=False, upper=None):
    _require(type(value) in (int, float), f'{field}: number required')
    try:
        valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    _require(valid and (upper is None or value <= upper), f'{field}: invalid finite range')


def _hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _metric(metrics, exact, field):
    _require(isinstance(metrics, dict) and set(metrics) == {'relative_l2', 'max_abs', 'top1_agreement'},
             f'{field}: complete metrics required')
    for key, value in metrics.items():
        _number(value, field+'.'+key, upper=1 if key == 'top1_agreement' else None)
    if exact:
        _require(metrics['relative_l2'] == metrics['max_abs'] == 0
                 and metrics['top1_agreement'] == 1, f'{field}: exact flag contradicts metrics')


def _case(case, inputs):
    name = case['name']
    _require(case.get('length') == CASE_LENGTHS[name]
             and case.get('token_sha256') == inputs[name]['token_sha256'], 'correctness input differs')
    for field in ('logits_exact', 'features_exact', 'state_exact'):
        _require(type(case.get(field)) is bool, f'{name}: explicit {field} required')
    _metric(case['logits'], case['logits_exact'], name+'.logits')
    _metric(case['features'], case['features_exact'], name+'.features')
    _require(isinstance(case.get('state_differences'), list)
             and case['state_exact'] == (not case['state_differences']), 'state exactness contradicts differences')
    tokens = case.get('tokens')
    _require(isinstance(tokens, dict) and set(tokens) == set(MODES)
             and all(type(token) is int and token >= 0 for token in tokens.values()), 'invalid correctness output token')
    followup = case.get('followup')
    _require(isinstance(followup, list) and len(followup) <= 8, 'invalid followup count')
    for step, row in enumerate(followup):
        _require(type(row.get('step')) is int and row['step'] == step
                 and type(row.get('exact')) is bool
                 and isinstance(row.get('state_differences'), list), 'missing/repeated followup step')
        _require(not row['exact'] or not row['state_differences'], 'exact followup has state differences')
    prompt_exact = all(case[field] for field in ('logits_exact', 'features_exact', 'state_exact'))
    if prompt_exact:
        _require(tokens['rows8'] == tokens['candidate'], 'exact prompt has different output tokens')
    return prompt_exact and len(followup) == 8 and all(row['exact'] for row in followup)


def summarize_candidate(result):
    """Return validated token/wall statistics, or a rejection without any TPS.

    These timings measure bounded prompt processing at the token API. They do
    not measure complete requests, decode throughput, task quality, or a full
    FP32 model reference. The input record is never modified.
    """
    _finite_tree(result)
    _require(result.get('status') in ('PASS_E36_CANDIDATE_PROMPT', 'REJECT_E36_CANDIDATE_NUMERICS'),
             'candidate result is not final')
    _require(all(result.get(field) is False for field in
                 ('full_model_fp32_reference', 'task_quality_evaluated', 'default_promoted')),
             'candidate exceeds bounded numerical scope')
    _require(result.get('expected_case_count') == 7
             and result.get('timing_scope') == 'prompt_only_token_API', 'candidate scope differs')
    frozen = result.get('candidate_inputs')
    _require(isinstance(frozen, list) and len(frozen) == 7
             and [row.get('name') for row in frozen] == list(CASE_LENGTHS), 'missing/repeated candidate inputs')
    inputs = {row['name']: row for row in frozen}
    for name, row in inputs.items():
        ids = row.get('ids')
        _require(row.get('length') == CASE_LENGTHS[name]
                 and isinstance(ids, list) and len(ids) == CASE_LENGTHS[name]
                 and all(type(token) is int and token >= 0 for token in ids), 'frozen prompt token count differs')
        checksum = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        _require(row.get('token_sha256') == checksum, 'frozen prompt hash differs')
    cases = result.get('cases')
    _require(isinstance(cases, list) and 1 <= len(cases) <= 7
             and [row.get('name') for row in cases] == list(CASE_LENGTHS)[:len(cases)],
             'missing/repeated correctness cases')
    passed = [_case(case, inputs) for case in cases]
    timings = result.get('timings')
    _require(isinstance(timings, list), 'timing records required')
    if result['status'] == 'REJECT_E36_CANDIDATE_NUMERICS':
        _require(result.get('performance_evaluated') is False and not timings,
                 'rejected candidate must not have performance measurements')
        stop = result.get('stop_case')
        _require(stop == cases[-1]['name'] and all(passed[:-1]) and not passed[-1],
                 'rejection must stop on a real failed gate')
        last = cases[-1]
        actual_false = (any(last[field] is False for field in ('logits_exact', 'features_exact', 'state_exact'))
                        or any(row['exact'] is False for row in last['followup']))
        _require(actual_false, 'rejection has no actual false numerical check')
        return dict(status=result['status'], performance_evaluated=False, stop_case=stop,
                    passed_cases=sum(passed), followup_checks=sum(len(case['followup']) for case in cases))
    _require(len(cases) == 7 and all(passed) and result.get('performance_evaluated') is True,
             'all seven correctness gates and 56 followups must pass before timing')
    expected = {(name, repetition, mode) for name in CASE_LENGTHS for repetition in range(4) for mode in MODES}
    _require(len(timings) == len(expected), 'missing/repeated timing rows')
    keyed = {}
    for row in timings:
        _require(type(row.get('repetition')) is int, 'integer timing repetition required')
        key = (row.get('name'), row['repetition'], row.get('mode'))
        _require(key in expected and key not in keyed, 'missing/repeated timing mode or repetition')
        _require(type(row.get('prompt_tokens')) is int and row['prompt_tokens'] == CASE_LENGTHS[key[0]],
                 'timed prompt token count differs')
        _number(row.get('prompt_ms'), 'prompt_ms', positive=True)
        _require(_hash(row.get('output_sha256')) and type(row.get('token')) is int
                 and row['token'] == cases[list(CASE_LENGTHS).index(key[0])]['tokens'][key[2]],
                 'timed output identity differs from correctness gate')
        keyed[key] = row
    for name in CASE_LENGTHS:
        for repetition in range(4):
            a, b = (keyed[name, repetition, mode] for mode in MODES)
            _require((a['output_sha256'], a['token']) == (b['output_sha256'], b['token']),
                     'paired timed outputs differ')
    modes = {}
    for mode in MODES:
        rows = [row for row in timings if row['mode'] == mode]
        tokens = sum(row['prompt_tokens'] for row in rows)
        wall = sum(row['prompt_ms'] for row in rows)
        modes[mode] = dict(prompt_tokens=tokens, prompt_ms=wall, prompt_tps=1000*tokens/wall)
    per_case = []
    for name, tokens in CASE_LENGTHS.items():
        means = {mode: sum(keyed[name, repetition, mode]['prompt_ms'] for repetition in range(4))/4 for mode in MODES}
        per_case.append(dict(name=name, prompt_tokens=tokens, mean_prompt_ms=means,
                             speedup=means['rows8']/means['candidate']))
    per_repetition = []
    for repetition in range(4):
        times = {mode: sum(keyed[name, repetition, mode]['prompt_ms'] for name in CASE_LENGTHS) for mode in MODES}
        per_repetition.append(dict(repetition=repetition, prompt_ms=times,
                                   speedup=times['rows8']/times['candidate']))
    return dict(status=result['status'], correctness_cases=7, followup_checks=56,
                performance_evaluated=True, timing_scope='prompt_only_token_API',
                modes=modes, per_case=per_case, per_repetition=per_repetition,
                prompt_speedup=modes['rows8']['prompt_ms']/modes['candidate']['prompt_ms'])
