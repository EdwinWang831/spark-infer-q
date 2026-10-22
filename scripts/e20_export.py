"""Audit paired collector records and export measured collection-only speed."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from e20_validate import CONTRACT, compatible_record
from spark_infer_q.collector import selected_positions
from spark_infer_q.confidence_head import cumulative_labels


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_records(path, raw):
    assert sha(path) == raw['dataset_sha256']
    records = [json.loads(line) for line in path.read_text().splitlines()]
    indexed = {}
    for record in records:
        key = record['id'], record['trajectory_mode']
        assert key not in indexed
        indexed[key] = record
        assert record['split'] == 'train'
        assert record['selected_positions'] == selected_positions(record['completion_tokens']-1)
        assert [r['position'] for r in record['rows']] == record['selected_positions']
        for row in record['rows']:
            assert len(row['x']) == 137 and all(math.isfinite(v) for v in row['x'])
            assert len(row['y']) == 7 and row['y'] == cumulative_labels(row['emitted'])
            assert len(row['emitted']) == 8
            assert all(type(v) is int and 1 <= v <= min(width, row['remaining'])
                       for width, v in enumerate(row['emitted'], 1))
            assert row['remaining'] == record['output_cap']-(row['position']+1)
            assert row['max_rows'] == min(8, row['remaining'])
            assert row['context_rows'] == record['prompt_tokens']+row['position']
            known = [math.log1p(row['context_rows']), math.log1p(row['position']+1),
                     math.log1p(row['remaining']), float(record['thinking'])]
            assert all(abs(a-b) < 1e-13 for a, b in zip(row['x'][21:25], known))
    assert len(records) == len(raw['tasks'])*2
    for task in raw['tasks']:
        a, b = [indexed[task['id'], mode] for mode in CONTRACT['modes']]
        compatible_record(a, b)
        assert a['token_sha256'] == task['token_sha256']
        assert len(a['rows']) == task['sampled_positions']
    return dict(status='PASS_PAIRED_RECORD_REPLAY', records=len(records),
                sampled_rows=sum(len(r['rows']) for r in records),
                limits='CPU replay checks persisted row identity/contracts; full target features/logits/state are checked by the GPU run')


def summary(raw):
    assert raw['contract'] == CONTRACT and len(raw['tasks']) == CONTRACT['tasks']
    tasks = []
    for index, task in enumerate(raw['tasks']):
        assert task['source_split'] == 'train'
        assert all(task[k] for k in ('exact_tokens', 'exact_features', 'exact_logits', 'exact_state', 'exact_training_rows'))
        assert task['committed_feature_rows'] == task['prompt_tokens']+task['completion_tokens']-1
        medians = {phase: {} for phase in ('collection_ms', 'trajectory_ms', 'sampled_prefix_and_calibration_ms')}
        for mode in CONTRACT['modes']:
            runs = [r for r in task['timings'] if r['mode'] == mode]
            assert sorted(r['repetition'] for r in runs) == list(range(CONTRACT['repetitions']))
            assert all(r['exact'] and 0 < r['trajectory_ms'] <= r['collection_ms'] for r in runs)
            for phase in medians:
                medians[phase][mode] = statistics.median(r[phase] for r in runs)
        for rep in range(CONTRACT['repetitions']):
            order = CONTRACT['modes'][:]
            if (index+rep) % 2:
                order.reverse()
            actual = sorted((r for r in task['timings'] if r['repetition'] == rep), key=lambda r: r['order'])
            assert [r['mode'] for r in actual] == order
        tasks.append(dict(id=task['id'], group=task['group'], language=task['language'], family=task['family'],
                          prompt_tokens=task['prompt_tokens'], completion_tokens=task['completion_tokens'],
                          eos=task['eos'], medians_ms=medians,
                          collection_speedup=medians['collection_ms']['serial']/medians['collection_ms']['speculative']))

    def totals(subset):
        times = {phase: {mode: sum(t['medians_ms'][phase][mode] for t in subset) for mode in CONTRACT['modes']}
                 for phase in ('collection_ms', 'trajectory_ms', 'sampled_prefix_and_calibration_ms')}
        ratio = times['collection_ms']['serial']/times['collection_ms']['speculative']
        return dict(tasks=len(subset), sum_task_median_ms=times, collection_speedup=ratio,
                    collection_time_reduction=1-1/ratio,
                    trajectory_speedup=times['trajectory_ms']['serial']/times['trajectory_ms']['speculative'])

    overall = totals(tasks)
    for phase, modes in raw['summary']['sum_task_median_ms'].items():
        assert all(abs(value-overall['sum_task_median_ms'][phase][mode]) < 1e-8 for mode, value in modes.items())
    assert abs(overall['collection_speedup']-raw['summary']['full_collection_speedup']) < 1e-12
    groups = defaultdict(list)
    for task in tasks:
        groups[task['group']].append(task)
    rng = random.Random(20261022)
    keys = sorted(groups)
    draws = sorted(totals([t for key in rng.choices(keys, k=len(keys)) for t in groups[key]])['collection_time_reduction']
                   for _ in range(4000))
    overall['group_bootstrap_time_reduction_95'] = [draws[99], draws[3899]]
    strata = {}
    for label, condition in [*[(lang, lambda t, lang=lang: t['language'] == lang) for lang in ('en', 'zh', 'mixed')],
                             ('long_input', lambda t: t['prompt_tokens'] >= 768),
                             ('natural_eos', lambda t: t['eos']), ('cap_reached', lambda t: not t['eos'])]:
        subset = [t for t in tasks if condition(t)]
        strata[label] = totals(subset) if subset else dict(tasks=0)
    repeats = []
    for repetition in range(CONTRACT['repetitions']):
        values = {mode: sum(t['collection_ms'] for task in raw['tasks'] for t in task['timings']
                            if t['mode'] == mode and t['repetition'] == repetition) for mode in CONTRACT['modes']}
        repeats.append(dict(repetition=repetition, total_ms=values, speedup=values['serial']/values['speculative']))
    return dict(overall=overall, strata=strata, per_task=tasks, repeats=repeats,
                formal_runs=len(tasks)*2*CONTRACT['repetitions'], paired_trace_comparisons=len(tasks),
                compared_committed_logit_rows=sum(t['committed_logit_rows'] for t in raw['tasks']),
                compared_committed_feature_rows=sum(t['committed_feature_rows'] for t in raw['tasks']))


def export(root, out):
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status='IN_PROGRESS', raw_sha256={}, exporter_sha256=sha(Path(__file__)),
        limits=['eight training-only tasks, three warmed paired repetitions; bounded greedy single-request collector',
                'collection timing includes prompt/trajectory/features/labels/calibration checks; excludes model setup, final JSON write and cross-mode oracle comparisons',
                'not measured serving decode TPS, predictor quality, multi-request GPU batching, random sampling or 4K+'])
    path = root/'e20-collector-validation.json'
    if path.exists():
        raw = json.loads(path.read_text())
        report['raw_sha256'][path.name] = sha(path)
        report.update(runtime_status=raw['status'], source_sha256=raw['source_sha256'],
                      task_manifest_sha256=raw['task_manifest_sha256'], tasks_started=len(raw['tasks']))
        if raw['status'] in ('PASS_E20_FAST_COLLECTOR', 'COMPLETE_E20_NO_SPEEDUP'):
            report['summary'] = summary(raw)
            data = root/raw['dataset']
            report['row_audit'] = audit_records(data, raw)
            report['raw_sha256'][data.name] = sha(data)
            report['status'] = 'VALIDATED_PENDING_ENTRY_SMOKE' if raw['gate_pass'] else 'COMPLETE_E20_NO_SPEEDUP'
            smoke_path = root/'e20-fast-smoke-00.json'
            if smoke_path.exists():
                smoke = json.loads(smoke_path.read_text())
                assert smoke['validation_sha256'] == sha(path)
                report['entry_smoke'] = dict(status=smoke['status'], prompts=len(smoke['tasks']))
                report['raw_sha256'][smoke_path.name] = sha(smoke_path)
                if smoke['status'] == 'PASS_E20_FAST_COLLECTION':
                    source = root/smoke['dataset']
                    assert sha(source) == smoke['dataset_sha256']
                    records = [json.loads(line) for line in source.read_text().splitlines()]
                    assert len(records) == len(smoke['tasks']) == 2
                    assert all(r['trajectory_mode'] == 'speculative' for r in records)
                    assert sum(len(r['rows']) for r in records) == smoke['positions']
                    report['status'] = 'PASS_E20_FAST_COLLECTION_VALIDATED'
                    report['raw_sha256'][source.name] = sha(source)
        elif raw['status'] == 'FAIL':
            report.update(status='FAIL', error=raw['error'])
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    for prefix in ('/Users/', '/workspace/', '/home/kai/', '/checkpoint-store/'):
        assert prefix not in payload
    (out/'2026-10-22-fast-collector.json').write_text(payload)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    result = export(args.raw, args.out)
    print(json.dumps(dict(status=result['status'], summary=result.get('summary', {}).get('overall'))))
