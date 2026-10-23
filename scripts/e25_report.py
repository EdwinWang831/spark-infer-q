"""Independently aggregate complete online traces; never fit or select a model."""
from collections import Counter
import argparse
import json
from pathlib import Path
import random
import statistics

from e25_online import CONTRACT, MODES, sha, summarize


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
            relative_gain={m: rates['expanded'] / rates[m] - 1 for m in MODES[1:]}))
    strata = {}
    for field in ('language', 'family'):
        strata[field] = {}
        for value in sorted({t[field] for t in tasks}):
            subset = [t for t in tasks if t[field] == value]
            tokens = sum((t['completion_tokens'] - 1) * CONTRACT['repetitions'] for t in subset)
            costs = {m: sum(v['decode_ms'] for t in subset for v in t['timings'] if v['mode'] == m) for m in MODES}
            rates = {m: tokens * 1000 / costs[m] for m in MODES}
            strata[field][value] = dict(prompts=len(subset), decode_tps=rates,
                relative_gain={m: rates['expanded'] / rates[m] - 1 for m in MODES[1:]})
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
            gains[m].append(costs[m]/costs['expanded'] - 1)
    sensitivity = {}
    for m, values in gains.items():
        values.sort()
        sensitivity[m] = [values[int(.025 * (len(values)-1))], values[int(.975 * (len(values)-1))]]
    return dict(per_repetition=repeat, strata_descriptive_only=strata, trace_totals=trace,
                hierarchical_prompt_repetition_bootstrap_gain_95=sensitivity)


def export(raw, out):
    online_path, seal_path = raw/'artifacts/e25-online.json', raw/'artifacts/e25-online-seal.json'
    online, seal = json.loads(online_path.read_text()), json.loads(seal_path.read_text())
    assert online['status'] == 'PASS_E25_ONLINE' and online['contract'] == seal['contract'] == CONTRACT
    assert online['source_seal_sha256'] == sha(seal_path)
    verified = {}
    for key in ('source_sha256', 'input_sha256'):
        for n, h in seal[key].items():
            if (raw/n).exists():
                assert sha(raw/n) == h, n
                verified[n] = h
    mandatory = ['e25_online.py', 'e25_prepare.py', 'artifacts/e24-fit.json', 'artifacts/e24-evaluate.json',
        'artifacts/e24-expanded-model.json', 'artifacts/e24-frozen_reference-model.json',
        'artifacts/e24-selected-model.json', 'artifacts/e17-fit.json', 'e24_test_tasks.json',
        'artifacts/e24-preflight.json'] + [f'artifacts/e24-test-{i:02d}.json' for i in range(3)]
    assert all(n in verified for n in mandatory)
    fit = json.loads((raw/'artifacts/e24-fit.json').read_text())
    assert online['fit_sha256'] == sha(raw/'artifacts/e24-fit.json')
    assert online['evaluation_sha256'] == sha(raw/'artifacts/e24-evaluate.json')
    assert online['model_sha256'] == fit['model_sha256']
    assert [t['id'] for t in online['tasks']] == seal['task_ids']
    expected = {t['id']: t for i in range(3) for t in json.loads((raw/f'artifacts/e24-test-{i:02d}.json').read_text())['tasks']}
    for t in online['tasks']:
        assert len(t['checks']) == 3 and {c['mode'] for c in t['checks']} == set(MODES)
        assert all(c['tokens_exact'] and c['committed_logits_exact'] and c['live_state_exact'] for c in t['checks'])
        assert t['token_sha256'] == expected[t['id']]['token_sha256']
        assert t['prompt_sha256'] == expected[t['id']]['prompt_sha256']
        assert all(sum(s['emitted'] for s in v['steps']) == t['completion_tokens'] - 1 for v in t['timings'])
    recomputed = summarize(online['tasks'])
    assert recomputed == online['summary']
    before = seal['worker_before']
    after = json.loads((raw/'worker-status.json').read_text())
    job = json.loads((raw/'jobs/done/087-e25-online.json').read_text())
    assert before['pid'] == after['pid'] == job['worker_pid'] == online['worker_pid']
    assert before['load_count'] == after['load_count'] == job['load_count'] == 1
    assert after['status'] == 'READY' and job['status'] == 'PASS'
    assert after['failed_jobs'] == before['failed_jobs']
    report = dict(status='PASS_E25_ONLINE', contract=CONTRACT, summary=recomputed,
        diagnostics=details(online['tasks']), started_unix=online['started_unix'], finished_unix=online['finished_unix'],
        raw_sha256=sha(online_path), source_seal_sha256=sha(seal_path),
        local_verified_source_input_sha256=verified,
        source_verification_scope='worker checked complete E24/E25 seals before/after; local exporter checks archived subset and complete timing/output records',
        resident_reuse=dict(load_count=1, same_pid=True, same_target_draft_objects=True, no_added_failed_jobs=True),
        exporter_sha256=sha(Path(__file__)), tasks=online['tasks'])
    assert job['result']['target_object_id'] == online['target_object_id']
    assert job['result']['draft_object_id'] == online['draft_object_id']
    evaluation = json.loads((raw/'artifacts/e24-evaluate.json').read_text())
    assert evaluation['target_object_id'] == online['target_object_id']
    assert evaluation['draft_object_id'] == online['draft_object_id']
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    for prefix in ('/Users/', '/workspace/', '/home/kai/', '/checkpoint-store/'):
        assert prefix not in text
    out.mkdir(parents=True, exist_ok=True)
    (out/'2026-10-23-online-comparison.json').write_text(text)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    report = export(args.raw, args.out)
    print(json.dumps(report['summary'], indent=2))
