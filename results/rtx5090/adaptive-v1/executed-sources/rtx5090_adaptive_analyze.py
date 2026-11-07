"""Select on old calibration only, then summarize fresh full-generation results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

from rtx5090_adaptive_policy import Controller
from rtx5090_fixed_analyze import paired_reduction, totals


def prepare(measured, quality, manifest):
    expected = {task['id'] for task in manifest['tasks']}
    if len(measured['tasks']) != len(expected) or {task['id'] for task in measured['tasks']} != expected:
        raise ValueError('incomplete or duplicated tasks')
    grades = {row['id']: row['grades'] for row in quality['tasks']}
    rows = []
    for row in measured['tasks']:
        if set(row['candidates']) != set(measured['modes']) or not row['fixed16_bridge_exact']:
            raise ValueError('candidate coverage or bridge failure')
        metrics = {}
        for name, candidate in row['candidates'].items():
            runs = [r for r in row['timings'] if r['width'] == name]
            repetitions = measured['protocol']['repetitions']
            if len(runs) != repetitions or {r['repetition'] for r in runs} != set(range(repetitions)) or not all(r['repeat_exact'] for r in runs):
                raise ValueError('missing or unstable repeats')
            config = measured['modes'][name]
            if config['kind'] != 'official':
                replay = Controller(config)
                for width, progress in zip(candidate['action_trace'], candidate['progress_per_round'], strict=True):
                    if replay.choose() != width:
                        raise ValueError('recorded action is not reproducible from causal history')
                    replay.observe(width, progress)
            metrics[name] = {
                'wall_s': statistics.median(r['request_wall_s'] for r in runs),
                'decode_s': statistics.median(r['decode_s'] for r in runs),
                'controller_s': statistics.median(r['controller_wall_ns'] for r in runs) / 1e9,
                'output_tokens': len(candidate['token_ids']), 'decode_tokens': len(candidate['token_ids']) - 1,
                'exact_hf_match': candidate['exact_hf_match'], 'hit_output_cap': candidate['hit_output_cap'],
                'quality': grades[row['id']][name], 'progress_per_round': candidate['progress_per_round'],
                'actions': candidate['action_trace'], 'token_ids': candidate['token_ids'],
            }
        rows.append({'id': row['id'], 'dataset': row['dataset'], 'metrics': metrics})
    return rows


def summarize(rows, modes):
    result = {'totals': {}, 'comparisons_vs_official16': {}, 'by_dataset': {}, 'same_output_subsets': {}}
    for name in modes:
        result['totals'][name] = totals(rows, lambda task: name)
        action_counts = Counter(width for row in rows for width in row['metrics'][name]['actions'])
        controller = sum(row['metrics'][name]['controller_s'] for row in rows)
        result['totals'][name].update(action_counts=dict(action_counts), controller_decide_observe_s=controller,
                                     controller_fraction_of_wall=controller/result['totals'][name]['sum_median_request_s'])
        if name != 'official16':
            result['comparisons_vs_official16'][name] = paired_reduction(rows, lambda t: 'official16', lambda t: name)
            same = [row for row in rows if row['metrics'][name]['token_ids'] == row['metrics']['official16']['token_ids']]
            result['same_output_subsets'][name] = {
                'tasks': len(same), 'ids': [row['id'] for row in same],
                'comparison': paired_reduction(same, lambda t: 'official16', lambda t: name) if same else None}
    for dataset in sorted({row['dataset'] for row in rows}):
        group = [row for row in rows if row['dataset'] == dataset]
        result['by_dataset'][dataset] = {name: totals(group, lambda t: name) for name in modes}
        if dataset == 'squad':
            for name in modes:
                result['by_dataset'][dataset][name]['mean_f1'] = statistics.mean(row['metrics'][name]['quality']['f1'] for row in group)
    result['per_task'] = [{'id': row['id'], 'dataset': row['dataset'],
        'modes': {name: {key: value for key, value in metric.items()
                        if key not in ('token_ids', 'progress_per_round', 'actions')}
                  for name, metric in row['metrics'].items()}} for row in rows]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--select-for-test', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve prior evidence')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    measured, quality, manifest = (json.loads(p.read_text()) for p in (args.run, args.quality, args.manifest))
    if measured['status'] != 'COMPLETE_MEASURED' or quality['status'] != 'COMPLETE' or quality['run_sha256'] != sha(args.run) or measured['manifest_sha256'] != sha(args.manifest) or quality['manifest_sha256'] != sha(args.manifest):
        raise ValueError('incomplete run or mismatched evidence')
    rows = prepare(measured, quality, manifest)
    result = summarize(rows, measured['modes'])
    result['created_unix'] = time.time()
    result['source_hashes'] = {p.name: sha(p) for p in (args.run, args.quality, args.manifest, Path(__file__),
                               Path(__file__).with_name('rtx5090_fixed_analyze.py'),
                               Path(__file__).with_name('rtx5090_adaptive_policy.py'))}
    result['timed_runs'] = sum(len(row['timings']) for row in measured['tasks'])
    result['discarded_timings'] = len(measured['discarded_timings'])
    result['limitations'] = measured['protocol']['limitations'] + [
        'generation wall includes prefill/controller/draft/verify/commit; excludes tokenization, loading, transport and grading',
        'CPU controller sub-timing covers choose/observe only; complete generation wall includes all adapter work',
        'same-output subsets are retrospective secondary diagnostics; primary result retains all requests',
        'bootstrap is conditional on a small public-task sample; no quality noninferiority or broad generalization claim',
        'NVML process sampling cannot exclude sub-sample activity or shared CPU interference']
    if args.select_for_test:
        if manifest['protocol']['phase'] != 'calibration' or any(t['split'] != 'calibration' for t in manifest['tasks']):
            raise ValueError('selection requires calibration-only data')
        candidates = manifest['protocol']['selectable_modes']
        selected = min(candidates, key=lambda name: (result['totals'][name]['sum_median_request_s'], name))
        result.update(selected_name=selected, selected_config=measured['modes'][selected],
                      test_manifest_sha256=sha(args.select_for_test),
                      selection_metric=manifest['protocol']['selection_metric'])
    elif 'candidate' in measured['modes']:
        result['candidate_vs_simple_baselines'] = {
            name: paired_reduction(rows, lambda t: name, lambda t: 'candidate') for name in ('lagged', 'ewma')}
        result['frozen_selection'] = measured['selection']['selected_name']
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key in (
        'selected_name', 'selected_config', 'totals', 'comparisons_vs_official16', 'candidate_vs_simple_baselines')}, indent=2))


if __name__ == '__main__':
    main()
