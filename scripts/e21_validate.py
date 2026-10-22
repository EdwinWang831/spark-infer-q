"""Resident paired full-collector validation; hidden features must match bitwise."""
import hashlib
import json
from pathlib import Path
import statistics
import time

from e21_tasks import get_tasks

SOURCES = ['e21_validate.py', 'e21_tasks.py', 'e21_collect.py',
           'spark_infer_q/collection_budget.py', 'e20_tasks.py', 'artifacts/e16-selected-model.json', 'calibration.json',
           'spark_infer_q/verification_budget.py', 'spark_infer_q/local_policy.py', 'spark_infer_q/collection_trajectory.py', 'spark_infer_q/collector.py',
           'spark_infer_q/confidence_features.py', 'spark_infer_q/confidence_head.py',
           'spark_infer_q/draft_fidelity.py', 'spark_infer_q/draft_context.py',
           'spark_infer_q/draft_head.py', 'spark_infer_q/prompt.py', 'spark_infer_q/speculative.py',
           'e3_target.py', 'e3_wide_target.py', 'e16_tasks.py', 'e17_run_contract.py', 'e17_new_tasks.json']
CONTRACT = dict(tasks=6, repetitions=3, modes=['serial', 'speculative'], capacity=2048,
                verification_width=8, maximum_positions=64,
                scope='training-only paired collector; same labels, all committed features and live state; full collection wall time',
                gate='all exact; aggregate and four fresh roots speedup > 1; diagnosed Chinese task regression <= 5%')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_payload(record):
    keys = ('position', 'x', 'y', 'emitted', 'context_rows', 'remaining', 'max_rows')
    return [{k: r[k] for k in keys} for r in record['rows']]


def compatible_record(golden, candidate):
    for key in ('prompt_tokens', 'completion_tokens', 'eos', 'token_sha256', 'prompt_sha256',
                'serial_positions', 'selected_positions'):
        assert golden[key] == candidate[key], key
    assert row_payload(golden) == row_payload(candidate), 'sampled features or labels changed'
    assert [(s['width'], s['keep'], s['exact']) for s in golden['cost_samples']] == [
        (s['width'], s['keep'], s['exact']) for s in candidate['cost_samples']], 'calibration outcomes changed'


def run(session, arguments):
    from e3_wide_target import WideTarget
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_context import CachedContextDraft
    from spark_infer_q.collection_budget import BudgetedTrajectoryDraft
    from spark_infer_q.local_policy import ProgressModel
    from spark_infer_q.confidence_features import ConfidenceDraft
    from spark_infer_q.collector import collect_record
    from spark_infer_q.collection_trajectory import collect_trajectory
    from spark_infer_q.prompt import prompt_state
    root, torch = session.root, session.torch
    hashes = {n: sha(root/n) for n in SOURCES}
    tasks = get_tasks(root)
    manifest = hashlib.sha256(json.dumps(tasks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    output = root/'artifacts/e21-collector-validation.json'
    raw = root/'artifacts/e21-collector-oracle-rows.jsonl'
    assert not output.exists() and not raw.exists()
    target = WideTarget(session.weights, capacity=2048)
    base = HeadDraft.attach(session.draft, session.projection)
    base.max_context = 2048
    native = CachedContextDraft.attach(base)
    featured = ConfidenceDraft.attach(base)
    prior_path = root/'artifacts/e16-selected-model.json'
    prior = json.loads(prior_path.read_text())
    assert json.loads((root/'calibration.json').read_text())['conditional_prior'] == session.calibration['conditional_prior']
    trajectory = BudgetedTrajectoryDraft(ConfidenceDraft.attach(base), ProgressModel(prior['metadata']),
        prior['costs'], prior['cpu_predict_ms'], session.calibration['conditional_prior'])
    report = dict(status='RUNNING', contract=CONTRACT, source_sha256=hashes,
                  task_manifest_sha256=manifest, tasks=[], started_unix=time.time(),
                  target_object_id=id(session.weights), draft_object_id=id(session.draft),
                  dataset=raw.name, boundary_checks=[])

    def save():
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(output)

    def check_result(reference, current, full_logits=False):
        assert current['tokens'] == reference['tokens']
        assert torch.equal(current['features'], reference['features']), 'committed hidden features changed'
        assert not target.state_difference(current['state'], reference['state']), 'live state changed'
        if full_logits:
            a, b = reference['logits'], current['logits']
            assert (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b)), 'committed logits changed'
        assert len(current['early_states']) == len(reference['early_states'])
        for a, b in zip(reference['early_states'], current['early_states']):
            assert not target.state_difference(a, b)

    raw.write_text('')
    save()
    try:
        # Actual-GPU first-token EOS and cap=1, plus explicit cache reset on reuse.
        task = tasks[0]
        trajectory.configure(task, calibrate=False)
        ids = session.tokenizer.apply_chat_template([dict(role='user', content=task['prompt'])],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        start, context, logits = prompt_state(target, ids, chunk_size=8)
        first = int(logits[-1].argmax())
        for cap, eos in [(1, session.eos), (8, (first,)), (2, session.eos), (9, session.eos)]:
            trajectory.configure(dict(task, output_cap=cap), calibrate=False)
            a = collect_trajectory(target, start.clone(), context, first, output_cap=cap,
                                   eos_ids=eos, trace=True)
            b = collect_trajectory(target, start.clone(), context, first, output_cap=cap,
                                   eos_ids=eos, draft=trajectory, request=('e21-boundary', cap), trace=True)
            check_result(a, b, True)
            report['boundary_checks'].append(dict(cap=cap, forced_first_eos=eos == (first,), exact=True))
            del a, b
        del start, context, logits
        for index, task in enumerate(tasks):
            kwargs = dict(calibrate=task['probe_calibration'], maximum=64)
            trajectory.configure(task, calibrate=task['probe_calibration'])
            golden_rec, golden = collect_record(session, target, native, featured, trajectory,
                                                 task, mode='serial', trace=True, **kwargs)
            fast_rec, fast = collect_record(session, target, native, featured, trajectory,
                                             task, mode='speculative', trace=True, **kwargs)
            check_result(golden, fast, True)
            compatible_record(golden_rec, fast_rec)
            feature_sha = hashlib.sha256(golden['features'].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            entry = dict(id=task['id'], group=task['group'], language=task['language'], family=task['family'],
                         source_split=task['split'], output_cap=task['output_cap'],
                         calibration_probe=task['probe_calibration'], prompt_tokens=golden_rec['prompt_tokens'],
                         completion_tokens=golden_rec['completion_tokens'], eos=golden_rec['eos'],
                         sampled_positions=len(golden_rec['rows']),
                         committed_feature_rows=len(golden['features']), committed_feature_sha256=feature_sha,
                         token_sha256=golden_rec['token_sha256'], committed_logit_rows=len(golden['tokens'])-1,
                         exact_tokens=True, exact_features=True, exact_logits=True, exact_state=True,
                         exact_training_rows=True, timings=[])
            report['tasks'].append(entry)
            with raw.open('a') as handle:
                for rec in (golden_rec, fast_rec):
                    handle.write(json.dumps(rec, ensure_ascii=False, allow_nan=False)+'\n')
            del fast
            save()
            for repetition in range(CONTRACT['repetitions']):
                modes = CONTRACT['modes'][:]
                if (index+repetition) % 2:
                    modes.reverse()
                for slot, mode in enumerate(modes):
                    rec, current = collect_record(session, target, native, featured, trajectory,
                                                  task, mode=mode, trace=False, **kwargs)
                    check_result(golden, current)
                    compatible_record(golden_rec, rec)
                    entry['timings'].append(dict(mode=mode, repetition=repetition, order=slot,
                        collection_ms=rec['collection_ms'], trajectory_ms=rec['trajectory_ms'],
                        sampled_prefix_and_calibration_ms=rec['sampled_prefix_and_calibration_ms'],
                        target_calls=rec['trajectory_target_calls'], draft_calls=rec['trajectory_draft_calls'],
                        width_counts=rec['trajectory_width_counts'], exact=True))
                    del current
                save()
                print(json.dumps(dict(e21_collector=task['id'], repetition=repetition)), flush=True)
            del golden, golden_rec
        totals = {}
        for phase in ('collection_ms', 'trajectory_ms', 'sampled_prefix_and_calibration_ms'):
            totals[phase] = {mode: sum(statistics.median(t[phase] for t in task['timings'] if t['mode'] == mode)
                                      for task in report['tasks']) for mode in CONTRACT['modes']}
        factor = totals['collection_ms']['serial']/totals['collection_ms']['speculative']
        fresh = {mode: sum(statistics.median(t['collection_ms'] for t in task['timings'] if t['mode'] == mode)
                           for task in report['tasks'][2:]) for mode in CONTRACT['modes']}
        chinese = {mode: statistics.median(t['collection_ms'] for t in report['tasks'][1]['timings'] if t['mode'] == mode)
                   for mode in CONTRACT['modes']}
        fresh_factor = fresh['serial']/fresh['speculative']
        chinese_regression = chinese['speculative']/chinese['serial']-1
        passed = factor > 1 and fresh_factor > 1 and chinese_regression <= .05
        report.update(summary=dict(sum_task_median_ms=totals, full_collection_speedup=factor,
            time_reduction=1-1/factor, fresh_roots_speedup=fresh_factor, diagnosed_chinese_regression=chinese_regression, sampled_positions=sum(t['sampled_positions'] for t in report['tasks']),
            committed_logit_rows=sum(t['committed_logit_rows'] for t in report['tasks']),
            formal_runs=len(tasks)*2*CONTRACT['repetitions'], oracle_runs=len(tasks)*2),
            status='PASS_E21_FAST_COLLECTOR' if passed else 'COMPLETE_E21_GATE_FAILED',
            gate_pass=passed, dataset_sha256=sha(raw), finished_unix=time.time())
        assert session.draft.max_context == 1024
        for name, digest in hashes.items():
            assert sha(root/name) == digest, name
        save()
        production_smoke = None
        if report['gate_pass']:
            # Exercise the actual subsequent-run entry, including SHA gating and
            # dataset persistence, without generating another full-size corpus.
            from e21_collect import run as collect_fast
            smoke_manifest = root/'e21-fast-smoke-manifest.json'
            with smoke_manifest.open('x') as handle:
                handle.write(json.dumps(dict(tasks=[dict(t, output_cap=16) for t in tasks[:2]]), ensure_ascii=False)+'\n')
            production_smoke = collect_fast(session, dict(manifest=smoke_manifest.name,
                manifest_sha256=sha(smoke_manifest), validation_sha256=sha(output),
                run_name='smoke', shard=0, shard_size=2))
            assert production_smoke['status'] == 'PASS_E21_FAST_COLLECTION'
        return dict(status=report['status'], artifact=output.name, speedup=factor,
                    production_smoke=production_smoke,
                    target_object_id=id(session.weights), draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}', finished_unix=time.time())
        save()
        raise
