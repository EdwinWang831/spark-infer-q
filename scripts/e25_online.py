"""Source-bound online comparison after the E24 held-out gate; no weight reload."""
import hashlib
import json
from pathlib import Path
import random
import time

MODES = ('expanded', 'frozen_reference', 'linear_reference')
CONTRACT = dict(
    modes=list(MODES), repetitions=3, prompts=12, output_cap=384,
    capacity=2048, max_depth=7, cpu_threads=2,
    selection='E24 test manifest only; salted SHA order within prespecified family/language quotas',
    quotas={'external_qa/en': 2, 'external_qa/zh': 2, 'code/en': 2, 'math/en': 2,
            'multi_turn/en': 1, 'multi_turn/zh': 1, 'long_context/en': 1, 'long_context/zh': 1},
    timing='wall-clock decode after first prompt token, including draft/features/predict/schedule/verify/commit; setup and correctness checks excluded',
    limits=['12-root controlled online comparison; not representative user traffic',
            'reuses E24 test roots without retraining or selecting models from online results',
            'no fixed-draft arm; differences quantify predictor choices, not total speculative gain',
            'no automatic deployment; all candidates keep the same target/draft execution'],
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_tasks(candidates):
    assert len(candidates) == len({t['group'] for t in candidates}) == 72
    chosen = []
    for key, count in CONTRACT['quotas'].items():
        family, language = key.split('/')
        pool = [t for t in candidates if t['family'] == family and t['language'] == language]
        pool.sort(key=lambda t: hashlib.sha256(('e25-online-v1/' + t['id']).encode()).hexdigest())
        assert len(pool) >= count
        chosen.extend(pool[:count])
    assert len(chosen) == len({t['group'] for t in chosen}) == CONTRACT['prompts']
    assert all(t['split'] == 'test' and t['output_cap'] == 384 and not t['thinking'] for t in chosen)
    return sorted(chosen, key=lambda t: hashlib.sha256(('e25-order-v1/' + t['id']).encode()).hexdigest())


def order(task_index, repetition):
    modes = list(MODES)
    if task_index % 2:
        modes.reverse()
    offset = (task_index + repetition) % 3
    return modes[offset:] + modes[:offset]


def summarize(tasks, samples=4000):
    """Pair at prompt level, preserving all repetitions in each bootstrap cluster."""
    assert len(tasks) == CONTRACT['prompts'] and len({t['group'] for t in tasks}) == len(tasks)
    rows = []
    for task in tasks:
        assert len(task['timings']) == 3 * CONTRACT['repetitions']
        totals = {}
        for mode in MODES:
            ts = [t for t in task['timings'] if t['mode'] == mode]
            assert sorted(t['repetition'] for t in ts) == list(range(CONTRACT['repetitions']))
            assert all(t['exact'] and t['decode_ms'] > 0 for t in ts)
            totals[mode] = sum(t['decode_ms'] for t in ts)
        rows.append((task, totals))

    def rates(indices):
        tokens = sum((rows[i][0]['completion_tokens'] - 1) * CONTRACT['repetitions'] for i in indices)
        return {mode: tokens * 1000 / sum(rows[i][1][mode] for i in indices) for mode in MODES}

    point = rates(list(range(len(rows))))
    rng = random.Random(20261023)
    draws = {mode: [] for mode in MODES[1:]}
    for _ in range(samples):
        r = rates(rng.choices(range(len(rows)), k=len(rows)))
        for mode in draws:
            draws[mode].append(r['expanded'] / r[mode] - 1)
    comparisons = {}
    for mode, values in draws.items():
        values.sort()
        comparisons[mode] = dict(relative_gain=point['expanded'] / point[mode] - 1,
            prompt_bootstrap_gain_95=[values[int(.025 * (samples - 1))], values[int(.975 * (samples - 1))]])
    per_task = []
    for task, totals in rows:
        per_task.append(dict(id=task['id'], family=task['family'], language=task['language'],
            decode_tps={m: (task['completion_tokens'] - 1) * CONTRACT['repetitions'] * 1000 / totals[m] for m in MODES}))
    return dict(decode_tps=point, versus=comparisons, per_task=per_task,
                prompts=len(rows), timed_runs=sum(len(t['timings']) for t in tasks),
                timed_tokens_per_arm=sum((t['completion_tokens'] - 1) * CONTRACT['repetitions'] for t in tasks))


def run(session, arguments):
    from e24_contract import check_seal
    from e24_run import expected_prompt
    from e3_wide_target import WideTarget
    from e3_mechanism_checks import entry, params
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.confidence_features import ConfidenceDraft, feature_vector
    from spark_infer_q.local_policy import ProgressModel
    from spark_infer_q.mlp_runtime import NumpyMLPProgressModel
    from spark_infer_q.draft_signals import observed_labels
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.speculative import greedy_result
    import spark_q_e3_varlen

    assert not arguments
    root, torch = session.root, session.torch
    check_seal(root)
    assert sha(Path(spark_q_e3_varlen.__file__)) == '722b6ea835fa3e7e349af7f22ac727045a65af487a35f6709f3838e066bd678e'
    seal_path = root/'artifacts/e25-online-seal.json'
    seal = json.loads(seal_path.read_text())
    assert seal['contract'] == CONTRACT
    before = json.loads((root/'worker-status.json').read_text())
    assert before['load_count'] == 1 and before['pid'] == seal['resident_pid']

    def check_sources():
        check_seal(root)
        for name, digest in seal['source_sha256'].items():
            assert sha(root/name) == digest, name
        for name, digest in seal['input_sha256'].items():
            assert sha(root/name) == digest, name
        assert sha(seal_path) == seal_digest

    seal_digest = sha(seal_path)
    check_sources()
    fit_path, eval_path = root/'artifacts/e24-fit.json', root/'artifacts/e24-evaluate.json'
    fit, evaluation = json.loads(fit_path.read_text()), json.loads(eval_path.read_text())
    assert fit['status'] == 'PASS_E24_FIT' and fit['selected_model'] == 'expanded'
    assert evaluation['status'] == 'PASS_E24_EVALUATION' and evaluation['gate_pass']
    assert evaluation['fit_sha256'] == sha(fit_path)
    assert fit['selected_model_sha256'] == sha(root/'artifacts/e24-selected-model.json')
    models = {}
    for name in MODES[:2]:
        p = root/f'artifacts/e24-{name}-model.json'
        assert sha(p) == fit['model_sha256'][name]
        models[name] = NumpyMLPProgressModel(json.loads(p.read_text())['metadata'])
    old = json.loads((root/'artifacts/e17-fit.json').read_text())
    models['linear_reference'] = ProgressModel(old['models'][old['selected_model']]['metadata'])
    tasks = get_tasks(json.loads((root/'e24_test_tasks.json').read_text())['tasks'])
    assert [t['id'] for t in tasks] == seal['task_ids']
    expected = {t['id']: t for i in range(3)
                for t in json.loads((root/f'artifacts/e24-test-{i:02d}.json').read_text())['tasks']}
    preflight = {t['id']: t['prompt_sha256'] for t in json.loads((root/'artifacts/e24-preflight.json').read_text())['lengths']}
    out = root/'artifacts/e25-online.json'
    assert not out.exists()
    report = dict(status='RUNNING', contract=CONTRACT, tasks=[], started_unix=time.time(),
                  source_seal_sha256=seal_digest, model_sha256=fit['model_sha256'],
                  fit_sha256=sha(fit_path), evaluation_sha256=sha(eval_path),
                  target_object_id=id(session.weights), draft_object_id=id(session.draft),
                  worker_pid=before['pid'], load_count=1)

    def save():
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        tmp.replace(out)

    save()
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(CONTRACT['cpu_threads'])
        target = WideTarget(session.weights, capacity=2048)
        base = HeadDraft.attach(session.draft, session.projection)
        base.max_context = 2048
        learned = ConfidenceDraft.attach(base)
        learned.feature_group = 'f1'

        def generate(task, start, context, first, mode, oracle=None):
            state, ctx = start.clone(), context.clone()
            tokens, steps, logs = [first], [], []
            stop = first in session.eos
            budget = spark_q_e3_varlen.BudgetProbe()
            current_cost = None

            def cost_at(n):
                return fit['costs']['long' if n >= 768 else 'short']

            def update_budget(c):
                paid = c['native_draft_ms'] + c['feature_increment_ms'] + fit['timings'][mode]['cpu_predict_ms']
                budget.curve([(r, v + paid) for r, v in enumerate(c['verify_commit_ms'], 1)])

            if mode != 'serial':
                learned.begin((task['id'], mode, time.monotonic_ns()))
                current_cost = cost_at(len(ctx))
                update_budget(current_cost)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            while len(tokens) < task['output_cap'] and not stop:
                anchor, offset = tokens[-1], len(tokens) - 1
                if mode == 'serial':
                    lg, _ = target.step(state, [anchor], capture=False)
                    nxt = int(lg[-1].argmax())
                    tokens.append(nxt)
                    logs.append(lg.cpu())
                    stop = nxt in session.eos
                    continue
                c = cost_at(len(ctx))
                if c is not current_cost:
                    current_cost = c
                    update_budget(c)
                limit, width, rates = min(8, task['output_cap'] - len(tokens)), 1, []
                t = time.perf_counter()
                proposals, info = learned.propose(ctx, anchor) if limit > 1 else ([], {})
                if limit > 1:
                    x = feature_vector(info['signals'], len(ctx), len(tokens), task['output_cap'] - len(tokens), task['thinking'], 'f1')
                    rates = models[mode].predict(x)[:limit - 1]
                    width = budget.schedule(params(), [entry(rates, rates, process=len(steps)+2, schedule=len(steps)+2)])[0]
                assert 1 <= width <= limit
                proposals = proposals[:width - 1]
                torch.cuda.synchronize()
                dm = (time.perf_counter() - t) * 1000
                t = time.perf_counter()
                lg, ft, buf = target.run(state, [anchor] + proposals)
                vm = (time.perf_counter() - t) * 1000
                pred = lg.argmax(-1).tolist()
                emitted, keep, _, stop = greedy_result(anchor, proposals, pred, session.eos, task['output_cap'] - len(tokens))
                labels = observed_labels(proposals, pred, len(emitted), session.eos)
                t = time.perf_counter()
                target.commit(state, buf, width, keep)
                cm = (time.perf_counter() - t) * 1000
                ctx = torch.cat([ctx, ft[:keep]])
                tokens.extend(emitted)
                assert len(ctx) == state.pos == start.pos + len(tokens) - 1
                if oracle is not None:
                    assert emitted == oracle['tokens'][offset+1:offset+1+len(emitted)]
                    for j in range(keep):
                        assert torch.equal(lg[j:j+1].cpu(), oracle['logits'][offset+j])
                steps.append(dict(width=width, emitted=keep, accepted=sum(v for _, v in labels),
                    observed=len(labels), draft_feature_budget_ms=dm, verify_ms=vm, commit_ms=cm))
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            return dict(tokens=tokens, state=state, logits=logs, steps=steps, decode_ms=elapsed)

        for ti, task in enumerate(tasks):
            report['active_task'] = dict(index=ti, id=task['id'], phase='prefill')
            save()
            ids, prompt_sha = expected_prompt(session, task)
            assert prompt_sha == preflight[task['id']] == expected[task['id']]['prompt_sha256']
            assert len(ids) + task['output_cap'] + 8 <= 2048
            torch.cuda.synchronize()
            t = time.perf_counter()
            start, ctx, lg = prompt_state(target, ids, chunk_size=8)
            prompt_ms = (time.perf_counter() - t) * 1000
            first = int(lg[-1].argmax())
            report['active_task']['phase'] = 'serial_reference'
            save()
            golden = generate(task, start, ctx, first, 'serial')
            token_sha = hashlib.sha256(json.dumps(golden['tokens']).encode()).hexdigest()
            assert token_sha == expected[task['id']]['token_sha256']
            rec = dict(id=task['id'], group=task['group'], family=task['family'], language=task['language'],
                prompt_tokens=len(ids), prompt_sha256=prompt_sha, prompt_ms=prompt_ms,
                completion_tokens=len(golden['tokens']), output_cap=task['output_cap'],
                eos=golden['tokens'][-1] in session.eos, token_sha256=token_sha,
                checks=[], timings=[])
            report['tasks'].append(rec)
            for mode in MODES:
                report['active_task'].update(phase='correctness_and_warmup', mode=mode)
                save()
                result = generate(task, start, ctx, first, mode, oracle=golden)
                assert result['tokens'] == golden['tokens']
                assert not target.state_difference(result['state'], golden['state'])
                rec['checks'].append(dict(mode=mode, tokens_exact=True, committed_logits_exact=True, live_state_exact=True))
                del result
            for rep in range(CONTRACT['repetitions']):
                for slot, mode in enumerate(order(ti, rep)):
                    report['active_task'].update(phase='timing', repetition=rep, mode=mode, slot=slot)
                    save()
                    result = generate(task, start, ctx, first, mode)
                    assert result['tokens'] == golden['tokens']
                    assert not target.state_difference(result['state'], golden['state'])
                    rec['timings'].append(dict(mode=mode, repetition=rep, order=slot,
                        decode_ms=result['decode_ms'], steps=result['steps'], exact=True))
                    del result
                save()
                print(json.dumps(dict(e25_online=task['id'], repetition=rep, task_index=ti)), flush=True)
            del start, ctx, golden
        check_sources()
        report.update(status='PASS_E25_ONLINE', summary=summarize(report['tasks']), finished_unix=time.time())
        report.pop('active_task', None)
        save()
        return dict(status=report['status'], artifact=out.name, target_object_id=id(session.weights), draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}', finished_unix=time.time())
        save()
        raise
    finally:
        torch.set_num_threads(old_threads)
