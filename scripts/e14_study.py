"""Resident online budget trial, using frozen E13 training calibration only."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from e14_tasks import CONTRACT, TASKS


def run(session, arguments):
    from e3_wide_target import WideTarget
    from e3_mechanism_checks import entry, params
    from spark_infer_q.confidence_budget import BudgetDraft, MarginTable
    from spark_infer_q.draft_context import CachedContextDraft
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_signals import SignalDraft, observed_labels
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.verification_budget import LaggedAcceptance
    import spark_q_e3_varlen
    torch = session.torch
    assert hashlib.sha256(Path(spark_q_e3_varlen.__file__).read_bytes()).hexdigest() == '722b6ea835fa3e7e349af7f22ac727045a65af487a35f6709f3838e066bd678e'
    smoke = arguments.get('smoke', False)
    contract = dict(CONTRACT)
    tasks = TASKS
    if smoke:
        contract.update(output_cap=32, repetitions=1)
        tasks = [dict(id='smoke', thinking=False, prompt='Explain optimistic concurrency control with a concrete example in five sentences.')]
    names = ['e14_study.py', 'e14_tasks.py', 'e14-calibration.json', 'e3_mechanism_checks.py',
             'e3_target.py', 'e3_wide_target.py', 'spark_infer_q/confidence_budget.py',
             'spark_infer_q/draft_context.py', 'spark_infer_q/draft_head.py',
             'spark_infer_q/draft_signals.py', 'spark_infer_q/prompt.py',
             'spark_infer_q/speculative.py', 'spark_infer_q/verification_budget.py']
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {name: sha(session.root/name) for name in names}
    calibration = json.loads((session.root/'e14-calibration.json').read_text())
    table = MarginTable(calibration['model'], max_depth=contract['max_depth'])
    target = WideTarget(session.weights, capacity=contract['capacity'])
    base = HeadDraft.attach(session.draft, session.projection)
    base.max_context = contract['capacity']
    native_draft, post_draft = CachedContextDraft.attach(base), BudgetDraft.attach(base)
    signal_reference = SignalDraft.attach(base)
    curve = session.calibration['verify_commit_curve_ms'][:contract['max_depth']+1]
    draft_cost = session.calibration['draft_median_ms']['8']
    prior = session.calibration['conditional_prior'][:contract['max_depth']]
    output = session.root/'artifacts'/('e14-smoke.json' if smoke else 'e14-online.json')
    report = dict(status='RUNNING', contract=contract, smoke=smoke, tasks=[], source_sha256=hashes,
                  calibration=calibration, verify_commit_curve_ms=curve, frozen_draft_cost_ms=draft_cost,
                  started_unix=time.time(), target_object_id=id(session.weights), draft_object_id=id(session.draft),
                  shared_weights=target.w is session.weights and base.weights is session.draft.weights)
    def save(): output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    def emit(**items): print(json.dumps(items), flush=True)

    def generate(task, start, context, first, mode, oracle=None):
        state, ctx = start.clone(), context.clone()
        tokens, logs, steps = [first], [], []
        stop = first in session.eos
        worker = native_draft if mode == 'native_cached' else post_draft
        if mode != 'serial': worker.begin((task['id'], time.monotonic_ns()))
        post_draft.capture_margin = mode == 'margin_budget'
        policy = LaggedAcceptance(prior)
        budget = spark_q_e3_varlen.BudgetProbe()
        budget.curve([(r, ms+(draft_cost if mode != 'native_cached' or r > 1 else 0.)) for r, ms in curve])
        if mode == 'native_cached':
            for _ in range(5): budget.runtime(1, 1, curve[0][1], 0.)
        torch.cuda.synchronize(); started = time.perf_counter()
        while len(tokens) < contract['output_cap'] and not stop:
            anchor, offset = tokens[-1], len(tokens)-1
            if mode == 'serial':
                lg, _ = target.step(state, [anchor], capture=False)
                logs.append(lg.cpu())
                nxt = int(lg[-1].argmax()); tokens.append(nxt); stop = nxt in session.eos
                continue
            max_k = min(contract['max_depth'], contract['output_cap']-len(tokens)-1)
            width, rates = 1, []
            t = time.perf_counter()
            if mode == 'native_cached':
                if max_k:
                    rates = policy.current[:max_k]
                    width = budget.schedule(params(), [entry(rates, policy.previous[:max_k], process=len(steps)+2, schedule=len(steps)+2)])[0]
                k = width-1
                proposals, info = worker.propose(ctx, anchor) if k else ([], {})
            else:
                proposals, info = worker.propose(ctx, anchor) if max_k else ([], {})
                if max_k:
                    rates = table.rates(info.get('edge_margin') if mode == 'margin_budget' else None, limit=max_k)
                    width = budget.schedule(params(), [entry(rates, rates, process=len(steps)+2, schedule=len(steps)+2)])[0]
                k = width-1
            proposals = proposals[:k]
            torch.cuda.synchronize(); dm = (time.perf_counter()-t)*1000
            t = time.perf_counter(); lg, ft, buffers = target.run(state, [anchor]+proposals)
            vm = (time.perf_counter()-t)*1000
            predictions = lg.argmax(-1).tolist()
            out, keep, _, stop = greedy_result(anchor, proposals, predictions, eos_ids=session.eos,
                                               max_new=contract['output_cap']-len(tokens))
            labels = observed_labels(proposals, predictions, len(out), session.eos)
            t = time.perf_counter(); target.commit(state, buffers, k+1, keep)
            cm = (time.perf_counter()-t)*1000
            ctx = torch.cat([ctx, ft[:keep]]); tokens.extend(out)
            assert len(ctx) == state.pos == start.pos+len(tokens)-1
            if oracle is not None:
                assert out == oracle['tokens'][offset+1:offset+1+len(out)], (task['id'], mode, 'tokens')
                for j in range(keep):
                    assert torch.equal(lg[j:j+1].cpu(), oracle['logits'][offset+j]), (task['id'], mode, 'logits', offset+j)
            accepted = sum(value for _, value in labels)
            policy.observe(len(labels), accepted)
            if mode == 'native_cached': budget.runtime(1, k+1, dm+vm+cm, 0.)
            steps.append(dict(context_rows=state.pos-keep, width=width, verified_rows=k+1,
                              emitted=keep, accepted=accepted, observed=len(labels),
                              cumulative_rates=rates, edge_margins=info.get('edge_margin'),
                              draft_and_budget_ms=dm, verify_ms=vm, commit_ms=cm))
        torch.cuda.synchronize(); elapsed = (time.perf_counter()-started)*1000
        return dict(tokens=tokens, state=state, logits=logs, steps=steps, decode_ms=elapsed, eos=stop)

    save()
    try:
        for ti, task in enumerate(tasks):
            ids = session.tokenizer.apply_chat_template([dict(role='user', content=task['prompt'])], tokenize=True,
                add_generation_prompt=True, enable_thinking=task['thinking'], return_dict=False)
            if 'target_prompt_tokens' in task:
                n = task['target_prompt_tokens']; assert len(ids) > n
                ids = ids[:n-128]+ids[-128:]
            assert len(ids)+contract['output_cap']+8 <= target.capacity
            start, ctx, lg = prompt_state(target, ids, chunk_size=8)
            first = int(lg[-1].argmax())
            # Independent full-path reference, excluded from all formal timings.
            native_draft.begin((task['id'], 'probe'))
            reference, _ = native_draft.propose(ctx, first)
            signal_path, signal_info = signal_reference.propose(ctx, first)
            assert reference == signal_path
            for capture in (False, True):
                post_draft.capture_margin = capture; post_draft.begin((task['id'], 'probe', capture))
                path, info = post_draft.propose(ctx, first)
                assert path == reference
                if capture: assert info['edge_margin'] == [v[0] for v in signal_info['signals']]
            golden = generate(task, start, ctx, first, 'serial')
            rec = dict(id=task['id'], prompt_tokens=len(ids), completion_tokens=len(golden['tokens']),
                       eos=golden['eos'], token_sha256=hashlib.sha256(json.dumps(golden['tokens']).encode()).hexdigest(),
                       selector_path_margin_exact=True, checks=[], timings=[])
            report['tasks'].append(rec)
            for mode in contract['modes']:
                result = generate(task, start, ctx, first, mode, oracle=golden)
                assert result['tokens'] == golden['tokens'] and not target.state_difference(result['state'], golden['state'])
                rec['checks'].append(dict(mode=mode, committed_logit_rows=len(golden['tokens'])-1,
                                         all_committed_logits_exact=True, live_state_exact=True))
                del result
            for rep in range(contract['repetitions']):
                modes = contract['modes'][:]
                if rep % 2: modes.reverse()
                shift = (rep+ti) % len(modes); modes = modes[shift:]+modes[:shift]
                for slot, mode in enumerate(modes):
                    result = generate(task, start, ctx, first, mode)
                    assert result['tokens'] == golden['tokens'] and not target.state_difference(result['state'], golden['state'])
                    rec['timings'].append(dict(mode=mode, repetition=rep, order=slot, decode_ms=result['decode_ms'],
                                              steps=result['steps'], tokens_and_state_exact=True))
                    del result
                save(); emit(e14_task=task['id'], repetition=rep, completed_tasks=ti)
            del start, ctx, golden
        for name, digest in hashes.items(): assert sha(session.root/name) == digest, name
        assert session.draft.max_context == 1024
        report.update(status='PASS_E14_SMOKE' if smoke else 'PASS_E14_ONLINE', finished_unix=time.time(),
                      resource=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        save()
        return dict(status=report['status'], artifact=output.name, tasks=len(tasks), shared_weights=report['shared_weights'],
                    target_object_id=id(session.weights), draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}', finished_unix=time.time()); save(); raise
