"""Validate and export the frozen online trial, including negative outcomes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics

from e14_tasks import CONTRACT, TASKS


def analyze(raw):
    assert raw['status'] == 'PASS_E14_ONLINE' and raw['contract'] == CONTRACT
    assert [t['id'] for t in raw['tasks']] == [t['id'] for t in TASKS]
    modes = CONTRACT['modes']
    for ti, task in enumerate(raw['tasks']):
        assert task['selector_path_margin_exact']
        assert {c['mode'] for c in task['checks']} == set(modes)
        assert all(c['all_committed_logits_exact'] and c['live_state_exact'] for c in task['checks'])
        for rep in range(CONTRACT['repetitions']):
            order = modes[:]
            if rep % 2: order.reverse()
            shift = (rep+ti) % len(order); order = order[shift:]+order[:shift]
            runs = sorted((r for r in task['timings'] if r['repetition'] == rep), key=lambda r:r['order'])
            assert [r['mode'] for r in runs] == order
            for run in runs:
                assert run['tokens_and_state_exact'] and run['decode_ms'] > 0
                assert sum(s['emitted'] for s in run['steps']) == task['completion_tokens']-1
                assert all(1 <= s['verified_rows'] <= 6 for s in run['steps'])
    def summarize(tasks, mode):
        medians = [statistics.median(r['decode_ms'] for r in t['timings'] if r['mode'] == mode) for t in tasks]
        steps = [s for t in tasks for r in t['timings'] if r['mode'] == mode for s in r['steps']]
        calls = sum(bool(s['cumulative_rates']) if mode != 'native_cached' else s['width'] > 1 for s in steps)
        return dict(sum_task_median_decode_ms=sum(medians), weighted_decode_tps=1000*sum(t['completion_tokens']-1 for t in tasks)/sum(medians),
                    steps=len(steps), draft_calls=calls, width_counts=dict(Counter(s['width'] for s in steps)),
                    verified_rows=sum(s['verified_rows'] for s in steps), committed_rows=sum(s['emitted'] for s in steps),
                    accepted=sum(s['accepted'] for s in steps), observed=sum(s['observed'] for s in steps),
                    phase_ms={k:sum(s[k] for s in steps) for k in ('draft_and_budget_ms','verify_ms','commit_ms')})
    summary = {scope:{m:summarize(tasks,m) for m in modes} for scope,tasks in
               [('all',raw['tasks']), ('short',raw['tasks'][:6]), ('long',raw['tasks'][6:])]}
    per_task = []
    for t in raw['tasks']:
        stats = {m:summarize([t],m) for m in modes}
        # Fixed calibration/costs make both post-draft arms deterministic. Only
        # compare widths at identical committed-prefix positions, never align
        # arbitrary step indices after their paths have diverged.
        paths = {m:{s['context_rows']:s['width'] for r in t['timings'] if r['mode']==m and r['repetition']==0
                    for s in r['steps']} for m in ('depth_budget','margin_budget')}
        common = paths['depth_budget'].keys() & paths['margin_budget'].keys()
        all_repeats_equal = True
        for rep in range(CONTRACT['repetitions']):
            sequences = {m:[(s['context_rows'],s['width']) for r in t['timings'] if r['mode']==m and r['repetition']==rep
                            for s in r['steps']] for m in ('depth_budget','margin_budget')}
            all_repeats_equal &= sequences['depth_budget'] == sequences['margin_budget']
        per_task.append(dict(id=t['id'], prompt_tokens=t['prompt_tokens'], completion_tokens=t['completion_tokens'],
                             median_decode_ms={m:stats[m]['sum_task_median_decode_ms'] for m in modes},
                             weighted_decode_tps={m:stats[m]['weighted_decode_tps'] for m in modes},
                             shared_prefix_positions=len(common),
                             different_width_at_shared_prefix=sum(paths['depth_budget'][p]!=paths['margin_budget'][p] for p in common),
                             post_draft_width_sequences_equal_all_repetitions=all_repeats_equal,
                             margin_vs_depth=stats['depth_budget']['sum_task_median_decode_ms']/stats['margin_budget']['sum_task_median_decode_ms']))
    comparisons = {}
    for baseline in ('depth_budget','native_cached'):
        rng = random.Random(20261021)
        def reduction(chosen):
            return 1-sum(t['median_decode_ms']['margin_budget'] for t in chosen)/sum(t['median_decode_ms'][baseline] for t in chosen)
        draws = sorted(reduction(rng.choices(per_task,k=len(per_task))) for _ in range(4000))
        repeats=[]
        for rep in range(CONTRACT['repetitions']):
            costs={m:sum(r['decode_ms'] for t in raw['tasks'] for r in t['timings'] if r['mode']==m and r['repetition']==rep) for m in modes}
            repeats.append(costs[baseline]/costs['margin_budget'])
        comparisons[baseline] = dict(factor=summary['all'][baseline]['sum_task_median_decode_ms']/summary['all']['margin_budget']['sum_task_median_decode_ms'],
                                     time_reduction=reduction(per_task), task_bootstrap_95=[draws[99],draws[3899]], repeat_factors=repeats)
    conditional = {}
    once = [s for t in raw['tasks'] for r in t['timings'] if r['mode']=='margin_budget' and r['repetition']==0 for s in r['steps']]
    for depth in range(1,6):
        observed = [s for s in once if s['observed']>=depth]
        model = raw['calibration']['model'][str(depth)]
        predicted = [model['buckets'][int(s['edge_margins'][depth-1]>model['threshold'])]['probability'] for s in observed]
        conditional[str(depth)] = dict(trials=len(observed), accepted=sum(s['accepted']>=depth for s in observed),
            observed_rate=sum(s['accepted']>=depth for s in observed)/len(observed) if observed else None,
            mean_margin_prediction=statistics.mean(predicted) if predicted else None,
            train_depth_prior=model['baseline'], train_labels=model['n'])
    return dict(status='PASS_E14_ANALYSIS', contract=CONTRACT, summary=summary, per_task=per_task, comparisons=comparisons,
                conditional_acceptance_once_per_task=conditional,
                formal_runs=len(TASKS)*len(modes)*CONTRACT['repetitions'], oracle_runs=len(TASKS)*len(modes),
                oracle_committed_logit_rows=sum(c['committed_logit_rows'] for t in raw['tasks'] for c in t['checks']),
                calibration=raw['calibration'], source_sha256=raw['source_sha256'], shared_weights=raw['shared_weights'],
                verify_commit_curve_ms=raw['verify_commit_curve_ms'], frozen_draft_cost_ms=raw['frozen_draft_cost_ms'],
                resource=raw['resource'], observations=raw['tasks'],
                limits=['bounded greedy decode excludes prompt, state clones and oracle; no complete-request or native TensorFold comparison',
                        'native reference capped to depth5 and keeps runtime feedback/metadata; isolates neither signal nor decision timing',
                        'confidence contribution is margin_budget versus depth_budget; both use fixed E13 train-only calibration and identical frozen costs',
                        'draft_and_budget_ms includes full drafting, signal extraction/transfer and decision; not standalone signal or GPU kernel cost',
                        'three long tasks have synthetic repeated backgrounds; task bootstrap describes this fixed suite',
                        'post-draft width1 still paid for its draft; no pre-draft confidence signal',
                        'no depth6/7, online expert cost model, random sampling or GPU batching tested'])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('input',type=Path); parser.add_argument('output',type=Path)
    parser.add_argument('--cost-probe',type=Path)
    args=parser.parse_args(); raw=json.loads(args.input.read_text()); result=analyze(raw)
    if args.cost_probe:
        cost=json.loads(args.cost_probe.read_text())
        assert cost['status']=='PASS_E14_COST_PROBE'
        assert len(cost['frozen_curve_choices']['corners'])==32
        assert [c['context_rows'] for c in cost['conditions']]==[384,1024,1792]
        for c in cost['conditions']:
            assert c['logits_and_committed_state_exact'] and len(c['choices']['corners'])==32
            assert all(len(v)==4 for v in c['verify_commit_samples_ms'].values())
        result['cost_diagnostic']={k:cost[k] for k in ('status','source_sha256','scope','frozen_curve_choices','conditions')}
        result['cost_diagnostic']['raw_sha256']=hashlib.sha256(args.cost_probe.read_bytes()).hexdigest()
    result.update(raw_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    content=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for prefix in ('/home/kai/','/Users/','/workspace/','/checkpoint-store/'): assert prefix not in content
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(content)
    print(json.dumps(dict(status=result['status'],comparisons=result['comparisons'],tps={m:s['weighted_decode_tps'] for m,s in result['summary']['all'].items()})))


if __name__=='__main__': main()
