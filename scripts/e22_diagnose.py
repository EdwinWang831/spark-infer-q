"""Post-test diagnosis only: frozen E17/E19 predictions, fees and decisions.

No fitting, candidate selection, promotion gate or target calls. Counterfactuals
reuse uniform sampled prefixes; their rates are not online decode throughput.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e17_train import expand, load_collection
from spark_infer_q.confidence_head import choose_rows
from spark_infer_q.local_policy import ProgressModel
from spark_infer_q.mlp_head import MLPProgressModel


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_rows(rows, predictions, costs, fee):
    """Change only the charged prediction fee; reconsider each action."""
    result = []
    for row, rates in zip(rows, predictions):
        c = costs['long' if row['context_rows'] >= 768 else 'short']
        paid = row['max_rows'] > 1
        curve = [v+(c['native_draft_ms']+c['feature_increment_ms']+fee if paid else 0.)
                 for v in c['verify_commit_ms']]
        width = choose_rows(rates, curve, row['max_rows'])
        result.append(dict(group=row['group'], width=width,
                           tokens=row['emitted'][width-1], ms=curve[width-1],
                           curve=curve, emitted=row['emitted'], limit=row['max_rows']))
    assert len(result) == len(rows) == len(predictions)
    return result


def rate(records):
    return sum(r['tokens'] for r in records)/sum(r['ms'] for r in records)


def comparison(candidate, baseline, samples=2000):
    assert len(candidate) == len(baseline) and candidate
    groups = defaultdict(lambda: [0., 0., 0., 0.])
    for a, b in zip(candidate, baseline):
        assert a['group'] == b['group']
        v = groups[a['group']]
        for i, n in enumerate((a['tokens'], a['ms'], b['tokens'], b['ms'])):
            v[i] += n
    def gain(keys):
        totals = [sum(groups[k][i] for k in keys) for i in range(4)]
        return (totals[0]/totals[1])/(totals[2]/totals[3])-1
    keys = sorted(groups)
    rng = random.Random(20261022)
    draws = sorted(gain(rng.choices(keys, k=len(keys))) for _ in range(samples))
    return dict(relative_gain=gain(keys), groups=len(keys),
                group_bootstrap_gain_95=[draws[int(.025*(samples-1))], draws[int(.975*(samples-1))]])


def quality(rows, predictions):
    """Cumulative Brier/ECE and properly masked conditional log loss."""
    count = len(rows)
    brier, ece, nll, support = [], [], [], []
    for d in range(7):
        brier.append(sum((p[d]-r['y'][d])**2 for r, p in zip(rows, predictions))/count)
        bins = defaultdict(list)
        conditional = []
        for r, p in zip(rows, predictions):
            bins[min(9, int(p[d]*10))].append((p[d], r['y'][d]))
            if d == 0 or r['y'][d-1]:
                q = p[d] if d == 0 else p[d]/max(1e-300, p[d-1])
                q = max(1e-12, min(1-1e-12, q))
                conditional.append(-(r['y'][d]*math.log(q)+(1-r['y'][d])*math.log1p(-q)))
        ece.append(sum(abs(sum(p-y for p, y in pairs)) for pairs in bins.values())/count)
        support.append(len(conditional))
        nll.append(statistics.mean(conditional) if conditional else None)
    return dict(rows=count, cumulative_brier=brier, mean_cumulative_brier=statistics.mean(brier),
                cumulative_ece_10bins=ece, conditional_nll=nll, conditional_support=support)


def oracle_diagnosis(rows, policies):
    """Aggregate-ratio oracle, then additive opportunity loss, not local ratios.

    A row's regret is max_w(e_w-q*c_w)-(e_chosen-q*c_chosen), where q is the
    aggregate oracle optimum. Ties are broken toward smaller widths. All arms
    must share a curve, including a common prediction fee.
    """
    first = next(iter(policies.values()))
    assert len(rows) == len(first)
    for arm in policies.values():
        assert len(arm) == len(first)
        assert all(a['curve'] == b['curve'] and a['emitted'] == b['emitted']
                   and a['group'] == b['group'] for a, b in zip(arm, first))
    q = 0.
    for _ in range(100):
        widths = [max(range(p['limit']), key=lambda i: p['emitted'][i]-q*p['curve'][i]) for p in first]
        new = sum(p['emitted'][w] for p, w in zip(first, widths))/sum(p['curve'][w] for p, w in zip(first, widths))
        if abs(new-q) < 1e-13:
            q = new
            break
        q = new
    else:
        raise RuntimeError('oracle did not converge')
    result = dict(oracle_tokens_per_ms=q, arms={})
    for name, arm in policies.items():
        by_direction = Counter()
        losses = Counter()
        strata = defaultdict(lambda: dict(positions=0, regret_token_equivalents=0.))
        prompts = defaultdict(lambda: dict(positions=0, regret_token_equivalents=0.))
        widths_hist = Counter()
        exact = 0
        total_regret = 0.
        for row, p in zip(rows, arm):
            values = [e-q*c for e, c in zip(p['emitted'][:p['limit']], p['curve'][:p['limit']])]
            best = max(range(len(values)), key=values.__getitem__)+1
            regret = max(0., max(values)-values[p['width']-1])
            direction = 'equal' if best == p['width'] else 'over' if p['width'] > best else 'under'
            by_direction[direction] += 1
            losses[direction] += regret
            total_regret += regret
            exact += regret < 1e-10
            widths_hist[p['width']] += 1
            ctx = row['context_rows']
            context = '<128' if ctx < 128 else '128..383' if ctx < 384 else '384..767' if ctx < 768 else '>=768'
            accepted = row['emitted'][-1]-1
            difficulty = 'accept0' if accepted == 0 else 'accept1..2' if accepted <= 2 else 'accept3..6' if accepted <= 6 else 'accept7'
            for key in ('language:'+row['language'], 'context:'+context, 'outcome:'+difficulty):
                strata[key]['positions'] += 1
                strata[key]['regret_token_equivalents'] += regret
            prompts[row['group']]['positions'] += 1
            prompts[row['group']]['regret_token_equivalents'] += regret
        # Dinkelbach residuals sum to zero, so regret is the exact aggregate gap.
        assert abs(total_regret-(q*sum(p['ms'] for p in arm)-sum(p['tokens'] for p in arm))) < 1e-7
        result['arms'][name] = dict(policy_tokens_per_ms=rate(arm), oracle_optimal_positions=exact,
            direction_counts=dict(by_direction), direction_regret=dict(losses), width_counts=dict(widths_hist),
            regret_token_equivalents=total_regret, strata=dict(strata),
            worst_groups=[dict(group=k, **v) for k, v in sorted(prompts.items(), key=lambda kv: -kv[1]['regret_token_equivalents'])[:10]])
    return result


def cost_diagnosis(online, costs):
    """Observed online verify/commit residuals; descriptive, with no refitting."""
    cells = defaultdict(list)
    crossed = Counter()
    for task in online['tasks']:
        # The sealed online code freezes the bucket at request start.
        bucket = 'long' if task['prompt_tokens'] >= 768 else 'short'
        c = costs[bucket]
        for trial in task['timings']:
            for step in trial['steps']:
                observed = step['verify_ms']+step['commit_ms']
                predicted = c['verify_commit_ms'][step['width']-1]
                cells['mode:'+trial['mode']].append((predicted, observed))
                if trial['mode'] == 'selected':
                    cells['selected/width:'+str(step['width'])].append((predicted, observed))
                    cells['selected/advance:'+str(step['emitted'])].append((predicted, observed))
                    crossed['steps'] += 1
                    crossed['different_context_bucket'] += (step['context_rows'] >= 768) != (bucket == 'long')
    def summarize(pairs):
        errors = sorted(abs(p-o) for p, o in pairs)
        return dict(samples=len(pairs), mean_observed_ms=statistics.mean(o for p,o in pairs),
            mae_ms=statistics.mean(errors), p90_absolute_error_ms=errors[int(.9*(len(errors)-1))],
            mape=statistics.mean(abs(p-o)/o for p,o in pairs),
            aggregate_signed_error=sum(p-o for p,o in pairs)/sum(o for p,o in pairs))
    return dict(scope='held-out online steps, three repetitions; correlated observations, no fitting',
                bucket_policy='request-start online versus dynamic context offline',
                selected_bucket_counts=dict(crossed), cells={k:summarize(v) for k,v in cells.items()})


def run(root, out):
    a = root/'artifacts'
    inputs = {}
    def read(name):
        p = a/name
        inputs['artifacts/'+name] = sha(p)
        return json.loads(p.read_text())
    transfer = json.loads((root/'transfer-manifest.json').read_text())
    for name, digest in transfer['files'].items():
        assert sha(root/name) == digest, name
    helpers = ['e17_train.py', 'e17_run_contract.py', 'spark_infer_q/local_policy.py',
               'spark_infer_q/mlp_head.py', 'spark_infer_q/confidence_head.py']
    local_root = Path(__file__).resolve().parents[1]
    helper_hashes = {}
    for name in helpers:
        local = local_root/('scripts/'+name if '/' not in name else name)
        helper_hashes[name] = sha(local)
        assert helper_hashes[name] == sha(root/name), name
    fit17, fit19 = read('e17-fit.json'), read('e19-fit.json')
    eval17, eval19 = read('e17-evaluation.json'), read('e19-evaluate.json')
    assert fit17['selected_model'] == fit19['selected_model'] == 'unweighted_f1'
    assert fit17['dataset_sha256'] == fit19['dataset_sha256']
    assert eval19['fit_sha256'] == sha(a/'e19-fit.json')
    for fit, prefix in ((fit17, 'e17'), (fit19, 'e19')):
        assert fit['selected_model_sha256'] == sha(a/f'{prefix}-selected-model.json')
        selected = read(f'{prefix}-selected-model.json')
        assert selected['metadata'] == fit['models'][fit['selected_model']]['metadata']
    records, _ = load_collection(root, 'test')
    tests = expand(records, 'external_test')
    fresh, _ = load_collection(root, 'learning')
    new_rows = expand(fresh, 'external')
    old = []
    for index in range(12):
        meta = read(f'e16-collection-{index:02d}.json')
        p = a/meta['dataset']
        assert sha(p) == meta['dataset_sha256']
        old.extend(r for r in map(json.loads, p.read_text().splitlines()) if r['split'] == 'train')
    subsets = dict(train=expand(old, 'e16_train')+[r for r in new_rows if r['split'] == 'train'],
                   calibration=[r for r in new_rows if r['split'] == 'calibration'], test=tests)
    assert len(subsets['train']) == fit17['training_rows'] == fit19['training_rows']
    assert len(subsets['calibration']) == fit17['calibration_rows']
    for left, right in (('train', 'test'), ('calibration', 'test'), ('train', 'calibration')):
        assert not {r['group'] for r in subsets[left]} & {r['group'] for r in subsets[right]}
    semantic_positions = 0
    for split_rows in subsets.values():
        for row in split_rows:
            assert row['y'] == [int(row['emitted'][-1] > d+1) for d in range(7)]
            assert len(row['x']) == 137 and len(row['emitted']) == 8
            assert 1 <= row['max_rows'] <= 8
            expected = [math.log1p(row['context_rows']), math.log1p(row['position']+1), math.log1p(row['remaining'])]
            assert all(abs(x-y) < 1e-13 for x,y in zip(row['x'][21:24], expected))
            semantic_positions += 1
    models = dict(linear=ProgressModel(fit17['models']['unweighted_f1']['metadata']),
                  mlp=MLPProgressModel(fit19['models']['unweighted_f1']['metadata']))
    fees = dict(linear=fit19['control_timings']['e17_selected']['cpu_predict_ms'],
                mlp=fit19['models']['unweighted_f1']['cpu_predict_ms'])
    predictions = {}
    quality_result = {}
    for split, rows in subsets.items():
        quality_result[split] = {}
        for name, model in models.items():
            ps = [model.predict(r['x']) for r in rows]
            quality_result[split][name] = quality(rows, ps)
            if split == 'test':
                predictions[name] = ps
        print(json.dumps(dict(split=split, rows=len(rows), quality=quality_result[split])), flush=True)
    arms = {name+'_fee_'+fee: policy_rows(tests, predictions[name], fit17['costs'], value)
            for name in models for fee, value in {**fees, 'zero': 0.}.items()}
    # Reproduce the sealed evaluation before using its counterfactuals.
    assert abs(rate(arms['mlp_fee_mlp'])-eval19['test']['policy']) < 1e-12
    sealed_linear = policy_rows(tests, predictions['linear'], fit17['costs'], fit17['models']['unweighted_f1']['cpu_predict_ms'])
    assert abs(rate(sealed_linear)-eval17['test']['policy']) < 1e-12
    reference = arms['linear_fee_linear']
    base = rate(reference)
    comps = {k: comparison(v, reference) for k, v in arms.items()}
    equal_gap = rate(arms['mlp_fee_linear'])/base-1
    fee_component = (rate(arms['mlp_fee_mlp'])-rate(arms['mlp_fee_linear']))/base
    assert abs(comps['mlp_fee_mlp']['relative_gain']-equal_gap-fee_component) < 1e-12
    oracle = oracle_diagnosis(tests, dict(linear=reference, mlp=arms['mlp_fee_linear']))
    report = dict(status='PASS_POSTHOC_DIAGNOSIS', contract=dict(training=False, new_target_calls=0,
        model_selection=False, test_usage='post-test exploratory diagnosis only; not a new independent validation',
        fees='E19 same-host rotated/reversed scalar backend timing',
        decomposition='additive fractions of the matched linear rate; ML decision then ML fee',
        oracle='hindsight aggregate-ratio oracle on sampled serial prefixes, common linear fee; not an online bound'),
        input_sha256=inputs, transfer_manifest_sha256=sha(root/'transfer-manifest.json'),
        source_sha256=sha(Path(__file__)), helper_sha256=helper_hashes,
        frozen_costs=fit17['costs'], cpu_predict_ms=fees,
        quality=quality_result, actual_gap=comps['mlp_fee_mlp'],
        semantic_audit_positions=semantic_positions,
        test_coverage=dict(prompts=len(records), roots=len({r['group'] for r in records}),
                           max_prompt_tokens=max(r['prompt_tokens'] for r in records),
                           max_context_rows=max(r['context_rows'] for r in tests),
                           positions_at_context_768_or_more=sum(r['context_rows'] >= 768 for r in tests),
                           max_output_cap=max(r['output_cap'] for r in records)),
        gap_decomposition=dict(decisions_at_linear_fee=equal_gap,extra_fee=fee_component),
        counterfactuals={k:dict(tokens_per_ms=rate(v),width_counts=dict(Counter(p['width'] for p in v)),
                              changed_from_linear=sum(x['width'] != y['width'] for x, y in zip(v, reference)),
                              versus_linear=comps[k]) for k, v in arms.items()},
        fee_changed_actions=dict(mlp=sum(x['width'] != y['width'] for x,y in zip(arms['mlp_fee_mlp'],arms['mlp_fee_linear'])),
                                 linear=sum(x['width'] != y['width'] for x,y in zip(arms['linear_fee_mlp'],reference))),
        oracle_diagnosis=oracle,
        online_cost_diagnosis=cost_diagnosis(read('e17-online.json'), fit17['costs']))
    report['per_language'] = {}
    for lang in ('en', 'zh', 'mixed'):
        indices = [i for i, r in enumerate(tests) if r['language'] == lang]
        report['per_language'][lang] = {k: comparison([v[i] for i in indices], [reference[i] for i in indices])
                                       for k, v in arms.items() if k in ('mlp_fee_mlp', 'mlp_fee_linear')}
    out.mkdir(parents=True, exist_ok=True)
    (out/'2026-10-23-predictor-diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_worker', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = run(args.raw_worker, args.output)
    print(json.dumps({k:report[k] for k in ('status','cpu_predict_ms','actual_gap','gap_decomposition','fee_changed_actions')}))
