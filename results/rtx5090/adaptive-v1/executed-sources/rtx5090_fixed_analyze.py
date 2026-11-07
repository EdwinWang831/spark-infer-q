"""Select on calibration; summarize paired held-out cost, quality and optimistic width choice."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics


def prepare_tasks(measured, quality):
    grades = {task['id']: task['grades'] for task in quality['tasks']}
    tasks = []
    for row in measured['tasks']:
        metrics = {}
        for width, candidate in row['candidates'].items():
            runs = [run for run in row['timings'] if str(run['width']) == width]
            if len(runs) != measured['protocol']['repetitions'] or not all(run['repeat_exact'] for run in runs):
                raise ValueError('incomplete or unstable timing rows')
            metrics[width] = {
                'wall_s': statistics.median(run['request_wall_s'] for run in runs),
                'decode_s': statistics.median(run['decode_s'] for run in runs),
                'output_tokens': len(candidate['token_ids']), 'decode_tokens': len(candidate['token_ids'])-1,
                'exact_hf_match': candidate['exact_hf_match'], 'hit_output_cap': candidate['hit_output_cap'],
                'quality': grades[row['id']][width], 'repeats': runs,
                'progress_per_round': candidate['progress_per_round'],
            }
        tasks.append({'id': row['id'], 'dataset': row['dataset'], 'split': row['split'], 'metrics': metrics,
                      'all_width_outputs_equal': all(c['token_ids'] == row['candidates']['1']['token_ids']
                                                     for c in row['candidates'].values())})
    return tasks


def choose_fixed(calibration, widths):
    return min(widths, key=lambda width: (sum(t['metrics'][width]['wall_s'] for t in calibration), int(width)))


def totals(tasks, choose):
    selected = [task['metrics'][choose(task)] for task in tasks]
    wall = sum(row['wall_s'] for row in selected)
    decode = sum(row['decode_s'] for row in selected)
    tokens = sum(row['decode_tokens'] for row in selected)
    progress = [n for row in selected for n in row['progress_per_round']]
    return {'tasks': len(tasks), 'sum_median_request_s': wall, 'sum_median_decode_s': decode,
            'output_tokens': sum(row['output_tokens'] for row in selected),
            'decode_tokens_per_s': tokens/decode if decode else None,
            'mean_committed_progress': statistics.mean(progress) if progress else None,
            'hf_exact_count': sum(row['exact_hf_match'] for row in selected),
            'output_cap_count': sum(row['hit_output_cap'] for row in selected),
            'strict_format_pass_count': sum(row['quality'].get('strict_pass', row['quality']['pass']) for row in selected),
            'quality_pass_count': sum(row['quality']['pass'] for row in selected)}


def paired_reduction(tasks, baseline, candidate, draws=4000):
    def reduction(rows):
        base = sum(t['metrics'][baseline(t)]['wall_s'] for t in rows)
        other = sum(t['metrics'][candidate(t)]['wall_s'] for t in rows)
        return 1-other/base
    groups = defaultdict(list)
    for task in tasks:
        groups[task['dataset']].append(task)
    rng = random.Random(20261106)
    estimates = []
    for _ in range(draws):
        sample = [rng.choice(group) for group in groups.values() for _ in group]
        estimates.append(reduction(sample))
    estimates.sort()
    return {'wall_time_reduction': reduction(tasks),
            'stratified_prompt_bootstrap_95': [estimates[int(.025*(draws-1))], estimates[int(.975*(draws-1))]],
            'bootstrap_draws': draws, 'independent_tasks': len(tasks)}


def analyze(measured, quality, draws=4000):
    tasks = prepare_tasks(measured, quality)
    calibration = [t for t in tasks if t['split'] == 'calibration']
    test = [t for t in tasks if t['split'] == 'test']
    widths = [str(width) for width in measured['protocol']['widths']]
    selected = choose_fixed(calibration, widths)
    speculative = choose_fixed(calibration, [width for width in widths if width != '1'])
    oracle = lambda task: min(widths, key=lambda w: (task['metrics'][w]['wall_s'], int(w)))
    result = {
        'calibration_selected_width': selected, 'calibration_selected_speculative_width': speculative,
        'calibration': {w: totals(calibration, lambda t, w=w: w) for w in widths},
        'test': {w: totals(test, lambda t, w=w: w) for w in widths},
        'test_selected_vs_target': paired_reduction(test, lambda t: '1', lambda t: selected, draws),
        'test_each_width_vs_target': {w: paired_reduction(test, lambda t: '1', lambda t, w=w: w, draws)
                                      for w in widths if w != '1'},
        'test_oracle_vs_selected': paired_reduction(test, lambda t: selected, oracle, draws),
        'test_oracle_totals': totals(test, oracle),
        'test_oracle_choice_counts': dict(Counter(oracle(t) for t in test)),
        'by_dataset': {},
    }
    # The held-out repetition is never used to choose its own width.
    base_seconds, chosen_seconds = 0.0, 0.0
    choices = []
    for task in test:
        for held_out in range(measured['protocol']['repetitions']):
            choice = min(widths, key=lambda w: (
                statistics.mean(row['request_wall_s'] for row in task['metrics'][w]['repeats']
                                if row['repetition'] != held_out), int(w)))
            observed = {w: next(row['request_wall_s'] for row in task['metrics'][w]['repeats']
                               if row['repetition'] == held_out) for w in widths}
            base_seconds += observed[selected]
            chosen_seconds += observed[choice]
            choices.append({'id': task['id'], 'held_out_repetition': held_out, 'width': choice})
    result['cross_repeat_oracle'] = {
        'wall_time_reduction': 1-chosen_seconds/base_seconds,
        'baseline_s': base_seconds, 'chosen_s': chosen_seconds, 'choices': choices,
        'note': 'same requests used for retrospective width selection; not unseen-request policy evaluation',
    }
    same = [task for task in test if task['all_width_outputs_equal']]
    result['same_output_subset'] = {
        'tasks': len(same), 'ids': [t['id'] for t in same],
        'oracle_vs_selected': paired_reduction(same, lambda t: selected, oracle, draws) if same else None,
    }
    for dataset in sorted({task['dataset'] for task in test}):
        roots = [task for task in test if task['dataset'] == dataset]
        result['by_dataset'][dataset] = {w: totals(roots, lambda t, w=w: w) for w in widths}
        if dataset == 'squad':
            for width in widths:
                result['by_dataset'][dataset][width]['mean_f1'] = statistics.mean(
                    task['metrics'][width]['quality']['f1'] for task in roots)
    result['per_task'] = [
        {'id': task['id'], 'dataset': task['dataset'], 'split': task['split'],
         'best_width_descriptive': oracle(task), 'all_width_outputs_equal': task['all_width_outputs_equal'],
         'widths': {w: {k: v for k, v in m.items() if k not in ('repeats', 'progress_per_round')}
                    for w, m in task['metrics'].items()}} for task in tasks]
    result['phase_progress'] = {}
    for width in widths:
        buckets = [[], [], []]
        for task in test:
            metric = task['metrics'][width]
            completed = 0
            for progress in metric['progress_per_round']:
                phase = min(2, completed*3//max(1, metric['decode_tokens']))
                buckets[phase].append(progress)
                completed += progress
        result['phase_progress'][width] = [
            {'rounds': len(values), 'mean_committed_progress': statistics.mean(values) if values else None}
            for values in buckets]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists')
    measured, quality = json.loads(args.run.read_text()), json.loads(args.quality.read_text())
    if measured['status'] != 'COMPLETE_MEASURED' or quality['run_sha256'] != hashlib.sha256(args.run.read_bytes()).hexdigest():
        raise ValueError('incomplete run or mismatched quality evidence')
    result = analyze(measured, quality)
    result['sources'] = {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (args.run, args.quality, Path(__file__))}
    result['limitations'] = measured['protocol']['limitations'] + [
        'different BF16 widths can produce different output lengths and answers; wall differences are not exact-output speedups',
        'oracle uses future measured outcomes; neither its gain nor its bootstrap interval establishes a deployable dynamic method',
        'bootstrap interval conditions on this small sample and chosen calibration width; no quality noninferiority claim',
    ]
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('calibration_selected_width', 'calibration_selected_speculative_width',
                                           'test', 'test_oracle_choice_counts', 'test_oracle_vs_selected')}, indent=2))


if __name__ == '__main__':
    main()
