"""Independently aggregate complete online traces; never fit or select a model."""
from collections import Counter
import argparse
import json
from pathlib import Path
import random
import statistics

from e27_online import CONTRACT, MODES, sha, summarize


def details(tasks):
    repeat = []
    for rep in range(CONTRACT['repetitions']):
        totals = {m: 0. for m in MODES}
        tokens = sum(t['completion_tokens'] - 1 for t in tasks)
        for task in tasks:
            ts = [v for v in task['timings'] if v['repetition'] == rep]
            assert len(ts) == 3 and {v['mode'] for v in ts} == set(MODES)
            for v in ts:
                totals[v['mode']] += v['decode_ms']
        rates = {m: tokens * 1000 / v for m, v in totals.items()}
        repeat.append(dict(repetition=rep, decode_tps=rates,
            relative_gain={m: rates['linear_live'] / rates[m] - 1 for m in MODES[1:]}))
    strata = {}
    for field in ('language', 'family'):
        strata[field] = {}
        for value in sorted({t[field] for t in tasks}):
            subset = [t for t in tasks if t[field] == value]
            tokens = sum((t['completion_tokens'] - 1) * CONTRACT['repetitions'] for t in subset)
            costs = {m: sum(v['decode_ms'] for t in subset for v in t['timings'] if v['mode'] == m) for m in MODES}
            rates = {m: tokens * 1000 / costs[m] for m in MODES}
            strata[field][value] = dict(prompts=len(subset), decode_tps=rates,
                relative_gain={m: rates['linear_live'] / rates[m] - 1 for m in MODES[1:]})
    trace = {}
    for mode in MODES:
        runs = [v for t in tasks for v in t['timings'] if v['mode'] == mode]
        steps = [s for v in runs for s in v['steps']]
        total = sum(v['decode_ms'] for v in runs)
        times = {key: sum(s[key] for s in steps) for key in ('draft_feature_budget_ms', 'verify_ms', 'commit_ms')}
        remainder = total - sum(times.values())
        trace[mode] = dict(decode_ms=total, verify_rounds=len(steps),
            emitted=sum(s['emitted'] for s in steps),
            emitted_per_verify=sum(s['emitted'] for s in steps)/len(steps),
            accepted=sum(s['accepted'] for s in steps), observed=sum(s['observed'] for s in steps),
            width_counts=dict(Counter(s['width'] for s in steps)), component_ms=times,
            other_loop_ms=remainder,
            per_width_verify_commit_mean_ms={str(w): statistics.mean(s['verify_ms'] + s['commit_ms'] for s in steps if s['width'] == w)
                                            for w in sorted({s['width'] for s in steps})})
    # Sensitivity: additionally resample paired repetitions within each sampled prompt.
    rng = random.Random(20261024)
    gains = {m: [] for m in MODES[1:]}
    for _ in range(4000):
        costs = {m: 0. for m in MODES}
        for task in rng.choices(tasks, k=len(tasks)):
            reps = rng.choices(range(CONTRACT['repetitions']), k=CONTRACT['repetitions'])
            for m in MODES:
                lookup = {v['repetition']: v['decode_ms'] for v in task['timings'] if v['mode'] == m}
                costs[m] += sum(lookup[r] for r in reps)
        for m in gains:
            gains[m].append(costs[m]/costs['linear_live'] - 1)
    sensitivity = {}
    for m, values in gains.items():
        values.sort()
        sensitivity[m] = [values[int(.025 * (len(values)-1))], values[int(.975 * (len(values)-1))]]
    return dict(per_repetition=repeat, strata_descriptive_only=strata, trace_totals=trace,
                hierarchical_prompt_repetition_bootstrap_gain_95=sensitivity)


def export(raw, out):
    online_path, seal_path = raw/'artifacts/e27-online.json', raw/'artifacts/e27-seal.json'
    online, seal = json.loads(online_path.read_text()), json.loads(seal_path.read_text())
    assert online['status'] == 'PASS_E27_ONLINE' and online['contract'] == seal['contract'] == CONTRACT
    assert online['source_seal_sha256'] == sha(seal_path)
    verified = {}
    for key in ('source_sha256', 'input_sha256'):
        for n, h in seal[key].items():
            if (raw/n).exists():
                assert sha(raw/n) == h, n
                verified[n] = h
    for n in ('e27_online.py', 'e27_prepare.py', 'spark_infer_q/linear_runtime.py', 'spark_infer_q/past_cost.py', 'e27_test_tasks.json', 'artifacts/e17-fit.json'):
        assert n in verified, n
    preflight_path = raw/'artifacts/e27-preflight.json'
    preflight = json.loads(preflight_path.read_text())
    assert online['preflight_sha256'] == sha(preflight_path)
    assert preflight['seal_sha256'] == sha(seal_path)
    assert preflight['model_sha256'] == online['model_sha256'] == sha(raw/'artifacts/e27-linear-model.json')
    meta = json.loads((raw/'artifacts/e27-linear-model.json').read_text())['metadata']
    old = json.loads((raw/'artifacts/e17-fit.json').read_text())
    assert meta == old['models'][old['selected_model']]['metadata']
    assert preflight['linear_changed_actions'] == 0 and preflight['linear_max_probability_error'] < 1e-12
    expected = {v['id']: v['prompt_sha256'] for v in preflight['preflight']['lengths']}
    tasks = json.loads((raw/'e27_test_tasks.json').read_text())['tasks']
    assert [t['id'] for t in online['tasks']] == [t['id'] for t in tasks]
    for t in online['tasks']:
        assert len(t['checks']) == 3 and {c['mode'] for c in t['checks']} == set(MODES)
        assert all(c['tokens_exact'] and c['committed_logits_exact'] and c['live_state_exact'] for c in t['checks'])
        assert t['prompt_sha256'] == expected[t['id']]
        assert all(sum(s['emitted'] for s in v['steps']) == t['completion_tokens']-1 for v in t['timings'])
    recomputed = summarize(online['tasks'])
    assert recomputed == online['summary']
    job = json.loads((raw/'jobs/done/091-e27-online.json').read_text())
    after = json.loads((raw/'worker-status.json').read_text())
    assert job['status'] == 'PASS' and job['result']['status'] == 'PASS_E27_ONLINE'
    assert after['pid'] == job['worker_pid'] == online['worker_pid'] == seal['resident_pid']
    assert after['load_count'] == job['load_count'] == online['load_count'] == 1
    assert job['result']['target_object_id'] == online['target_object_id'] == preflight['target_object_id']
    assert job['result']['draft_object_id'] == online['draft_object_id'] == preflight['draft_object_id']
    timeline = json.loads((raw/'resident-timeline.json').read_text())
    overlapping = [v for v in timeline if v['job'] != job['job'] and v.get('started_unix',v['finished_unix']) < job['finished_unix'] and v['finished_unix'] > job['started_unix']]
    assert not overlapping
    diagnostics = details(online['tasks'])
    # Validate causal forecasts by replaying only samples preceding each decision.
    from spark_infer_q.past_cost import PastVerificationCost
    errors = {}
    for mode in MODES:
        values = []
        for task in online['tasks']:
            for run in task['timings']:
                if run['mode'] != mode:
                    continue
                adapter = PastVerificationCost(preflight['costs'], history=CONTRACT['cost_history'], ridge=CONTRACT['cost_ridge'], relative_limit=CONTRACT['cost_relative_limit'])
                position = task['prompt_tokens']
                for step in run['steps']:
                    assert step['context_rows'] == position
                    baseline = preflight['costs']['long' if position >= 768 else 'short']['verify_commit_ms']
                    remaining = task['output_cap'] - (position-task['prompt_tokens']+1)
                    predicted = adapter.curve(position)[step['width']-1] if mode == 'linear_live' and remaining > 1 else baseline[step['width']-1]
                    assert abs(predicted-step['forecast_verify_commit_ms']) < 1e-10
                    actual = step['verify_ms'] + step['commit_ms']
                    values.append(abs(actual-step['forecast_verify_commit_ms']))
                    adapter.observe(position, step['width'], actual)
                    position += step['emitted']
        errors[mode] = dict(observed_step_count=len(values), verify_commit_mae_ms=sum(values)/len(values))
    backend_costs=[{m:sum(v['decode_ms'] for v in t['timings'] if v['mode']==m) for m in ('linear_original','linear_fast')} for t in online['tasks']]
    def backend_gain(indices):
        return sum(backend_costs[i]['linear_original'] for i in indices)/sum(backend_costs[i]['linear_fast'] for i in indices)-1
    rng=random.Random(20261027)
    draws=sorted(backend_gain(rng.choices(range(12),k=12)) for _ in range(4000))
    # The two frozen-cost backends must visit the same verification trajectory.
    for task in online['tasks']:
        for rep in range(CONTRACT['repetitions']):
            paired = {v['mode']: v for v in task['timings'] if v['repetition'] == rep}
            keys = ('context_rows', 'width', 'emitted', 'accepted', 'observed')
            assert [[s[k] for k in keys] for s in paired['linear_original']['steps']] == [[s[k] for k in keys] for s in paired['linear_fast']['steps']]
    rng = random.Random(20261028)
    nested = []
    for _ in range(4000):
        costs = {m: 0. for m in ('linear_original', 'linear_fast')}
        for task in rng.choices(online['tasks'], k=12):
            reps = rng.choices(range(CONTRACT['repetitions']), k=CONTRACT['repetitions'])
            for mode in costs:
                lookup = {v['repetition']: v['decode_ms'] for v in task['timings'] if v['mode'] == mode}
                costs[mode] += sum(lookup[r] for r in reps)
        nested.append(costs['linear_original']/costs['linear_fast']-1)
    nested.sort()
    repeat_gains = [v['decode_tps']['linear_fast']/v['decode_tps']['linear_original']-1 for v in diagnostics['per_repetition']]
    backend=dict(relative_gain=backend_gain(range(12)),prompt_bootstrap_gain_95=[draws[99],draws[3899]],
        hierarchical_prompt_repetition_bootstrap_gain_95=[nested[99],nested[3899]],
        per_repetition_relative_gain=repeat_gains, same_verification_trajectories=True)
    report = dict(status='PASS_E27_ONLINE', contract=CONTRACT, summary=recomputed, diagnostics=diagnostics,fast_backend_versus_original=backend,
        forecast_error_descriptive_only=errors, preflight={k:v for k,v in preflight.items() if k not in ('target_object_id','draft_object_id')},
        raw_sha256=sha(online_path), source_seal_sha256=sha(seal_path), local_verified_source_input_sha256=verified,
        resident_reuse=dict(same_pid=True,load_count=1,same_target_draft_objects=True,overlapping_worker_jobs=0),
        source_verification_scope='worker checked complete E24/E27 seals; exporter verifies archived subset and complete timings',
        exporter_sha256=sha(Path(__file__)),started_unix=online['started_unix'],finished_unix=online['finished_unix'],tasks=online['tasks'])
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    for prefix in ('/Users/','/workspace/','/home/kai/','/checkpoint-store/'):
        assert prefix not in text
    out.mkdir(parents=True,exist_ok=True)
    (out/'2026-10-23-linear-cost-comparison.json').write_text(text)
    return report


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    r=export(a.raw,a.out)
    print(json.dumps(r['summary'],indent=2))
