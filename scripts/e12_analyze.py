"""Prompt-held-out cost prediction; current routes are explicitly offline-only."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark_infer_q.moe_cost import fit_step_cost, predict_ms, union_sizes, prefix_unions, routed_bytes
from e12_tasks import CONTRACT, TASKS


def summary(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values), count=len(values))


def flatten(raw):
    if raw['status'] != 'PASS_OBSERVATION' or raw['contract'] != CONTRACT or raw['tasks'] != TASKS:
        raise ValueError('incomplete or unfrozen observation')
    tasks = {t['id']: t for t in TASKS}
    positions = defaultdict(list)
    rows = []
    seen = set()
    for sample in raw['samples']:
        key = sample['task'], sample['position']
        if key in seen:
            raise ValueError('duplicate position')
        seen.add(key)
        if sample['task'] not in tasks or sample['split'] != tasks[sample['task']]['split']:
            raise ValueError('split mismatch')
        positions[sample['task']].append(sample)
        if [a['width'] for a in sample['actions']] != CONTRACT['widths']:
            raise ValueError('missing width')
        full_routes = sample['actions'][-1]['routes']
        for action in sample['actions']:
            width, routes, timings = action['width'], action['routes'], action['timings']
            if routes != full_routes[:width]:
                raise ValueError('route prefix differs across verification widths')
            if (len(routes) != width or any(len(row) != 48 for row in routes)
                    or any(len(layer) != 10 or len(set(layer)) != 10
                           or any(type(e) is not int or not 0 <= e < 512 for e in layer)
                           for row in routes for layer in row)):
                raise ValueError('invalid routes')
            unions = union_sizes(routes)
            if (unions != action['union_per_layer'] or sum(unions) != action['union_total']
                    or prefix_unions(routes) != action['prefix_union_totals']
                    or routed_bytes(sum(unions)) != action['selected_pack_footprint_bytes']):
                raise ValueError('route accounting mismatch')
            if (sorted(t['repetition'] for t in timings) != list(range(CONTRACT['repetitions']))
                    or len(action['correctness']) != 4 or not all(v is True for v in action['correctness'].values())
                    or not 1 <= action['emitted_tokens'] <= width):
                raise ValueError('missing repetition or failed oracle')
            numbers = [sample['draft_ms'], action['observation_wall_ms']]
            numbers += [t[k] for t in timings for k in ('verify_ms', 'commit_ms', 'stream_elapsed_ms')]
            if any(not math.isfinite(n) or n <= 0 for n in numbers):
                raise ValueError('invalid measurement')
            costs = [t['verify_ms']+t['commit_ms'] for t in timings]
            rows.append(dict(task=sample['task'], split=sample['split'], position=sample['position'],
                width=width, context=sample['context_rows'], experts=action['union_total'],
                previous=sample['previous_behavior_union'], ms=statistics.median(costs),
                draft_ms=sample['draft_ms'], emitted=action['emitted_tokens'],
                verify_ms=statistics.median(t['verify_ms'] for t in timings),
                cost_relative_span=(max(costs)-min(costs))/statistics.median(costs),
                observation_ms=action['observation_wall_ms']))
    if set(positions) != set(tasks):
        raise ValueError('missing prompt')
    for samples in positions.values():
        samples.sort(key=lambda s: s['position'])
        previous = None
        for index, sample in enumerate(samples):
            if sample['position'] != index or sample['previous_behavior_union'] != previous:
                raise ValueError('future/counterfactual route leaked into history')
            if index and sample['context_rows'] != (samples[index-1]['context_rows']+
                                                    samples[index-1]['actions'][3]['emitted_tokens']):
                raise ValueError('behavior state position mismatch')
            previous = sample['actions'][3]['union_total']
        if (len(samples) > CONTRACT['positions_per_prompt'] or
                (len(samples) < CONTRACT['positions_per_prompt'] and not samples[-1]['actions'][3]['stopped_by_eos'])):
            raise ValueError('incomplete behavior trajectory')
    return rows


def fit_models(train):
    models = {name: {} for name in ('M0', 'M1', 'M2', 'M3')}
    for width in CONTRACT['widths']:
        selected = [r for r in train if r['width'] == width]
        models['M0'][str(width)] = dict(median_ms=statistics.median(r['ms'] for r in selected))
        for name, feature in (('M1', None), ('M2', 'experts'), ('M3', 'previous')):
            data = [dict(rows=r['context'], experts=r[feature] if feature else 0., ms=r['ms']) for r in selected]
            try:
                # Top-10 gives a constant union of 480 at width1. Its
                # coefficient cannot be identified separately from the intercept.
                constant = bool(feature) and len({d['experts'] for d in data}) == 1
                models[name][str(width)] = fit_step_cost(data, experts=bool(feature) and not constant)
                if constant:
                    models[name][str(width)]['fallback'] = 'M1: constant expert feature in training'
            except ValueError as exc:
                models[name][str(width)] = dict(error=str(exc))
    return models


def predict(name, model, row):
    cost = model[str(row['width'])]
    if 'error' in cost:
        raise ValueError(cost['error'])
    if name == 'M0':
        value = cost['median_ms']
    else:
        feature = row['experts'] if name == 'M2' else row['previous'] if name == 'M3' else 0.
        value = predict_ms(cost, row['context'], feature)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('nonpositive predicted time')
    return value


def paired_reduction(base, candidate):
    """Bootstrap paired prompt clusters; never count repeats as independent."""
    tasks = sorted(base)
    if tasks != sorted(candidate):
        raise ValueError('unpaired predictions')
    def reduction(chosen):
        b = sum(sum(base[t]) for t in chosen)
        c = sum(sum(candidate[t]) for t in chosen)
        return 1-c/b if b else None
    point = reduction(tasks)
    if point is None:
        return dict(relative_mae_reduction=None, reason='zero baseline error')
    generator = random.Random(20261021)
    draws = [reduction(generator.choices(tasks, k=len(tasks))) for _ in range(2000)]
    draws = sorted(d for d in draws if d is not None)
    return dict(relative_mae_reduction=point,
                prompt_bootstrap_95=[draws[int(.025*len(draws))], draws[min(len(draws)-1, int(.975*len(draws)))]],
                prompts=len(tasks), bootstrap_draws=len(draws))


def analyze(raw):
    rows = flatten(raw)
    # Equal coverage for all models; M3 has no prior-request route at position 0.
    eligible = [r for r in rows if r['previous'] is not None]
    train = [r for r in eligible if r['split'] == 'train']
    test = [r for r in eligible if r['split'] == 'test']
    if not train or not test:
        raise ValueError('missing eligible train/test data')
    models = fit_models(train)
    result = dict(status='PASS_OFFLINE_ANALYSIS', contract=CONTRACT, models=models,
        positions=len(raw['samples']), actions=len(rows), timed_verify_calls=len(rows)*CONTRACT['repetitions'],
        train_actions=len(train), test_actions=len(test), excluded_first_step_actions=len(rows)-len(eligible),
        evaluations={}, comparisons={}, observations=rows,
        model_contract=dict(M0='one median per width', M1='per-width intercept plus context',
            M2='M1 plus current true expert union, offline only',
            M3='M1 plus previous executed width4 union',
            coefficient_mapping='within each width, inherited per_row_ms is the context slope, not a verification-width slope'),
        measurements=dict(
            repeated_cost_relative_span=summary([r['cost_relative_span'] for r in rows]),
            observation_over_original_verify=summary([r['observation_ms']/r['verify_ms'] for r in rows])),
        limits=['same-prefix costs and fixed width4 behavior, not online decode speedup',
                'M2 uses current target routes and is offline-only',
                'M3 history reflects executed width4 only; changed-policy feedback untested',
                'six held-out prompts; broader workload and context claims require further evidence',
                'selected pack footprint is not measured DRAM traffic'])
    losses, predictions = {}, {}
    for name, model in models.items():
        try:
            predicted = [predict(name, model, row) for row in test]
        except ValueError as exc:
            result['evaluations'][name] = dict(status='INVALID_MODEL', error=str(exc))
            continue
        errors = [abs(p-r['ms']) for p, r in zip(predicted, test)]
        by_task = defaultdict(list)
        for row, error in zip(test, errors):
            by_task[row['task']].append(error)
        losses[name], predictions[name] = by_task, predicted
        result['evaluations'][name] = dict(status='PASS_PREDICTION', mae_ms=statistics.mean(errors),
            median_ae_ms=statistics.median(errors),
            per_width={str(w): summary([e for e, r in zip(errors, test) if r['width'] == w]) for w in CONTRACT['widths']},
            per_prompt={t: summary(es) for t, es in by_task.items()})
    for candidate in ('M2', 'M3'):
        for base in ('M0', 'M1'):
            if candidate in losses and base in losses:
                result['comparisons'][candidate+'_vs_'+base] = paired_reduction(losses[base], losses[candidate])
    result['budget_diagnostic'] = budget_diagnostic(test, predictions, raw['frozen_conditional_prior'])
    m2 = result['comparisons'].get('M2_vs_M1', {})
    m3 = result['comparisons'].get('M3_vs_M1', {})
    gate = CONTRACT['model_gate_relative_mae_reduction']
    result['decision'] = dict(
        offline_expert_point_gate=(m2.get('relative_mae_reduction') or 0) >= gate,
        historical_point_gate=(m3.get('relative_mae_reduction') or 0) >= gate,
        default_changed=False, online_benefit='NOT_TESTED',
        interpretation='point gates are diagnostics; inspect prompt intervals, noise and budget opportunity before online work')
    return result


def budget_diagnostic(test, predictions, conditional):
    if len(conditional) != 7:
        raise ValueError('expected frozen seven-depth prior')
    expected, product = [1.], 1.
    for probability in conditional:
        if not 0 <= probability <= 1:
            raise ValueError('invalid prior')
        product *= probability
        expected.append(expected[-1]+product)
    groups = defaultdict(list)
    for index, row in enumerate(test):
        groups[(row['task'], row['position'])].append(index)
    totals = {name: [] for name in list(predictions)+['true_cost_diagnostic']}
    disagreements = defaultdict(int)
    details = []
    for key, indices in groups.items():
        actual = {test[i]['width']: test[i]['ms']+(test[i]['draft_ms'] if test[i]['width'] > 1 else 0.) for i in indices}
        selected = {}
        for name in totals:
            costs = actual if name == 'true_cost_diagnostic' else {
                test[i]['width']: predictions[name][i]+(test[i]['draft_ms'] if test[i]['width'] > 1 else 0.) for i in indices}
            width = max(CONTRACT['widths'], key=lambda w: (expected[w-1]/costs[w], -w))
            selected[name] = width
            totals[name].append(expected[width-1]/actual[width])
        for name in predictions:
            disagreements[name] += selected[name] != selected.get('M1')
        details.append(dict(task=key[0], position=key[1], selected_widths=selected))
    return dict(scope='same-prefix expected tokens per measured ms with fixed acceptance prior; not end-to-end or a rigorous upper bound',
        expected_tokens_by_width=expected, mean_expected_tokens_per_ms={n: statistics.mean(v) for n, v in totals.items()},
        width_disagreements_vs_M1=dict(disagreements), positions=details)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text()))
    result['raw_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result['analysis_sources_sha256'] = {
        str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__).resolve(), Path(__file__).resolve().with_name('e12_tasks.py'),
                  Path(__file__).resolve().parents[1]/'spark_infer_q/moe_cost.py')}
    raw = json.loads(args.input.read_text())
    result['collection_sources_sha256'] = raw['harness_sha256']
    result['target'] = raw['target']
    result['head'] = raw['head']
    result['resource'] = raw['resource']
    result['elapsed_seconds'] = raw['finished_unix']-raw['started_unix']
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(decision=result['decision'], comparisons=result['comparisons'])))


if __name__ == '__main__':
    main()
