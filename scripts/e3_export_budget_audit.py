#!/usr/bin/env python3
"""Validate and export a completed budget audit without private paths or logs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from spark_infer_q.budget_audit import MODES, hindsight_paths, mode_order


def export(source, output, repo):
    report = json.loads(source.read_text())
    assert report['status'] == 'PASS_BOUNDED_BUDGET_AUDIT'
    for name, expected in report['harness_sha256'].items():
        path = repo/name if '/' in name else repo/'scripts'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    repetitions = report['measurement_contract']['repetitions']
    assert len(report['tasks']) == 8
    histogram = Counter()
    replay_histogram = Counter()
    replay_best_fixed, replay_fixed3, replay_oracle = 0., 0., 0.
    replay_positions = 0
    for task_index, task in enumerate(report['tasks']):
        assert task['stopped_by_eos']
        assert len(task['correctness']) == 4
        assert all(c['tokens_equal'] and c['kept_logits_bit_exact'] and c['live_state_bit_exact'] for c in task['correctness'])
        assert len(task['timings']) == repetitions*len(MODES)
        for rep in range(repetitions):
            runs = sorted((r for r in task['timings'] if r['repetition'] == rep), key=lambda r:r['order'])
            assert tuple(r['mode'] for r in runs) == mode_order(rep,task_index)
            assert all(r['tokens_equal'] and r['live_state_bit_exact'] for r in runs)
        for row in task['timings']:
            if row['mode'] == 'dynamic': histogram.update(s['width'] for s in row['steps'])
        replay = task.get('counterfactual_replay')
        if replay:
            assert replay['all_kept_logits_and_live_states_exact']
            assert len(replay['positions']) == len(task['token_ids'])-1
            assert replay['paths'] == hindsight_paths(replay['positions'])
            paths = replay['paths']
            replay_best_fixed += paths['fixed'][str(paths['best_fixed_width'])]['replayed_cost_ms']
            replay_fixed3 += paths['fixed']['4']['replayed_cost_ms']
            replay_oracle += paths['hindsight']['replayed_cost_ms']
            replay_histogram.update(a['width'] for a in paths['hindsight']['path'])
            replay_positions += len(replay['positions'])
    summary = report['summary']
    summary['dynamic_width_counts'] = dict(sorted(histogram.items()))
    best = min(('fixed1','fixed2','fixed3'), key=lambda mode:summary[mode]['sum_medians_ms'])
    summary['best_aggregate_fixed_mode'] = best
    summary['dynamic_ratio_over_best_fixed'] = summary[best]['sum_medians_ms']/summary['dynamic']['sum_medians_ms']
    summary['maximum_live_position'] = max(t['prompt_tokens']+len(t['token_ids'])-1 for t in report['tasks'])
    report['replay_summary'] = {'tasks':sum('counterfactual_replay' in t for t in report['tasks']),
        'serial_positions':replay_positions,'timed_actions':replay_positions*4*3,
        'live_state_checks':replay_positions*4,
        'sum_per_task_best_fixed_replayed_ms':replay_best_fixed,
        'sum_fixed3_replayed_ms':replay_fixed3,'sum_hindsight_replayed_ms':replay_oracle,
        'best_fixed_over_hindsight':replay_best_fixed/replay_oracle,
        'fixed3_over_hindsight':replay_fixed3/replay_oracle,
        'hindsight_width_counts':dict(sorted(replay_histogram.items())),
        'scope':'future-aware sum of independent action medians; neither an online policy nor measured end-to-end gain'}
    text = json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for prefix in ('/home/kai/','/Users/','/workspace/','/checkpoint-store/'):
        assert prefix not in text, prefix
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(text)
    return summary,report['replay_summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    summary,replay = export(args.source,args.output,args.repo)
    print(json.dumps({'summary':summary,'replay_summary':replay},indent=2))
