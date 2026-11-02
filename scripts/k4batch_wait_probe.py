"""One-variable Target wait probe; every mode retains completion fences."""
import hashlib
import importlib
import json
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType


def run(session, arguments):
    root = Path(arguments['source_root'])
    manifest = json.loads((root / 'manifest.json').read_text())

    def check_sources():
        for name, digest in manifest.items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name

    check_sources()
    tag = '_k4_wait_probe_' + hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()[:16]
    package = ModuleType(tag)
    package.__path__ = [str(root / 'spark_infer_q')]
    sys.modules[tag] = package
    modes = ('optimized_single', 'batch_sleep', 'batch_spin', 'batch_event')
    # Each variant gets the same three completed requests; reverse and rotate
    # order to expose carryover/drift rather than grouping variants together.
    orders = (modes, tuple(reversed(modes)),
              ('batch_spin', 'optimized_single', 'batch_event', 'batch_sleep'))
    target = solo = provider = None
    gold = policy = None
    identity = id(session.weights), id(session.draft)
    report = dict(status='RUNNING', source_sha256=manifest, model_reloaded=False,
        production_changed=False, checks=[], timings=[],
        contract=dict(modes=list(modes), repetitions=3, max_tokens=64,
            policy='unchanged E17/W8/E33 and per-request post-draft budget',
            timing='after first output token through terminal/native retirement; excludes prefill/capture',
            intervention='single request; only the incomplete-poll wait differs within batch modes',
            waits=dict(batch_sleep='explicit regression control: sleep 0.5 ms between incomplete polls',
                       batch_spin='poll immediately, no sleep',
                       batch_event='synchronize the current verify/head/commit event after an incomplete poll'),
            fences='all original poll checks, commit events and retirement checks remain; native completion follows execute',
            instrumentation='inclusive execute wall and per-propose wall; stream intervals are not kernel timing'))

    def save(active=None):
        if active is not None:
            report['active'] = active
        path = root / 'wait-probe-result.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    save('initializing')
    try:
        opt = importlib.import_module(tag + '.kaifra_optimized')
        batch = importlib.import_module(tag + '.kaifra_batch')
        spec = importlib.import_module(tag + '.kaifra_speculative')
        native_module = importlib.import_module(tag + '.native_scheduler')
        target_module = importlib.import_module(tag + '.kaifra_joint_target')
        head_module = importlib.import_module(tag + '.screened_head_fused')
        from e3_wide_target import WideTarget

        torch = session.torch
        provider, metadata = opt.optimized_provider(session)
        report['profile_metadata'] = metadata
        solo = opt.optimized_target(session)
        solo.warmup()
        target = target_module.JointTarget(WideTarget(session.weights, capacity=2048),
                                          head_module.FusedScreenedHead(session.weights.head))
        target.warmup()
        ids = session.tokenizer.apply_chat_template(
            [dict(role='user', content='请用三点解释缓存与队列的区别。')], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        report['prompt_sha256'] = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        prior = session.calibration['conditional_prior']

        with native_module.NativeScheduler(session.root.parent / 'DeepInfer',
                session.root.parent / 'build-release/lib/deepinfer', max_drafts=7) as native:
            def generate(mode, reference=False):
                nonlocal gold, policy
                if mode == 'optimized_single':
                    engine = spec.speculative_engine(native, solo, provider, eos_ids=session.eos,
                        prior=prior, capacity=768, mode='adaptive')
                else:
                    engine = batch.batch_engine(native, target, provider, eos_ids=session.eos,
                        prior=prior, capacity=768)
                with engine:
                    handle = engine.submit_tokens('wait-probe', ids, max_tokens=64)
                    bridge = engine.bridge
                    chosen_provider = bridge.backend.provider
                    screen = solo.screen if mode == 'optimized_single' else target.screen
                    jit_before = set(screen.kernel_sha256)
                    counters_before = dict(target.counters)
                    started = None
                    tokens, trace, draft_wall, draft_budget, execute_wall = [], [], [], [], []
                    waits = dict(incomplete_polls=0, event_waits=0)
                    original_propose = chosen_provider.propose
                    original_execute = bridge.execute

                    def propose(*args, **kwargs):
                        if started is None:
                            return original_propose(*args, **kwargs)
                        t = time.perf_counter()
                        result = original_propose(*args, **kwargs)
                        draft_wall.append((time.perf_counter() - t) * 1000)
                        return result

                    def execute(plan):
                        t = time.perf_counter()
                        if mode == 'optimized_single':
                            original_execute(plan)
                        else:
                            bridge.execute_async(plan)
                            while not bridge.poll(plan):
                                waits['incomplete_polls'] += 1
                                if mode == 'batch_sleep':
                                    time.sleep(.0005)
                                    continue
                                if mode == 'batch_event':
                                    execution = plan.execution
                                    if execution.phase == 'verify':
                                        event = execution.target_ticket.event
                                    else:
                                        assert execution.phase == 'commit', execution.phase
                                        event = execution.event
                                    assert event is not None
                                    event.synchronize()
                                    waits['event_waits'] += 1
                        if started is not None:
                            execute_wall.append((time.perf_counter() - t) * 1000)

                    chosen_provider.propose = propose
                    bridge.execute = execute
                    try:
                        while engine.active_requests:
                            events = engine.step()
                            for event in events:
                                tokens.extend(event.token_ids)
                            for metric in engine.last_step_metrics:
                                if not metric['prompt']:
                                    trace.append({k: metric[k] for k in
                                        ('context_rows', 'verified_rows', 'emitted', 'accepted', 'observed', 'rates')})
                                    draft_budget.append(metric['draft_ms'])
                            if started is None and tokens:
                                assert len(tokens) == 1
                                started = time.perf_counter()
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - started) * 1000
                        assert not bridge.backend.states and not bridge.native.has_unfinished_seqs()
                        assert bridge.blocks.get_num_free_blocks() == bridge.block_count - 1
                        if reference:
                            gold, policy = tokens, trace
                        else:
                            assert tokens == gold and trace == policy, ('output/policy differs', mode)
                        result = dict(mode=mode, decode_ms=elapsed, decode_tokens=len(tokens)-1,
                            decode_tps=(len(tokens)-1)*1000/elapsed, output_policy_exact=True,
                            completion_tokens=len(tokens), rounds=len(trace),
                            verified_rows=sum(m['verified_rows'] for m in trace),
                            widths=[m['verified_rows'] for m in trace],
                            draft_propose_ms=draft_wall, draft_propose_total_ms=sum(draft_wall),
                            draft_propose_median_ms=statistics.median(draft_wall),
                            draft_budget_ms=draft_budget, draft_budget_total_ms=sum(draft_budget),
                            execute_ms=execute_wall, execute_total_ms=sum(execute_wall),
                            waits=waits, new_jit_shapes=sorted(set(screen.kernel_sha256)-jit_before),
                            counters={k: target.counters[k]-counters_before.get(k, 0) for k in target.counters}
                                if mode != 'optimized_single' else None)
                    finally:
                        chosen_provider.__dict__.pop('propose', None)
                        bridge.__dict__.pop('execute', None)
                return result

            for mode in modes:
                save(dict(mode=mode, phase='warmup_correctness'))
                report['checks'].append(generate(mode, reference=mode == 'optimized_single'))
                save()
            for repetition, order in enumerate(orders):
                for position, mode in enumerate(order):
                    save(dict(mode=mode, repetition=repetition, order=position, phase='timing'))
                    result = generate(mode)
                    assert not result['new_jit_shapes'], result['new_jit_shapes']
                    result.update(repetition=repetition, order=position)
                    report['timings'].append(result)
                    save()
                    print(json.dumps({k: result[k] for k in
                        ('mode', 'repetition', 'decode_tps', 'draft_propose_total_ms', 'execute_total_ms')}), flush=True)

        assert len(solo.graphs) == 16 and len(target.graphs) == 80 and not target.fallbacks
        summary = {}
        for mode in modes:
            rows = [r for r in report['timings'] if r['mode'] == mode]
            summary[mode] = dict(decode_tps=sum(r['decode_tokens'] for r in rows)*1000/sum(r['decode_ms'] for r in rows),
                draft_propose_total_ms=statistics.mean(r['draft_propose_total_ms'] for r in rows),
                draft_propose_median_ms=statistics.median(v for r in rows for v in r['draft_propose_ms']),
                execute_total_ms=statistics.mean(r['execute_total_ms'] for r in rows))
        assert identity == (id(session.weights), id(session.draft))
        check_sources()
        report.update(status='PASS_K4_WAIT_PROBE', summary=summary, final_capabilities=target.capabilities,
            model_identity_preserved=True, gloo_released=not torch.distributed.is_initialized(),
            limitations=['single short Chinese request; not a batch throughput benchmark',
                'a wait-mode effect does not by itself prove CPU frequency, migration or GPU clocks',
                'busy polling is a diagnostic intervention, not a production recommendation'])
        report.pop('active', None)
        save()
        return report
    finally:
        for execution in (target, solo):
            if execution is None:
                continue
            session.torch.cuda.synchronize()
            if getattr(execution, 'inflight', None) is not None:
                execution.discard(execution.inflight)
            if execution is target:
                for state, leased in zip(target.states, target.leased):
                    if leased:
                        target.release_state(state)
            elif execution.leased:
                execution.release_state(execution.workspace)
            execution.close()
        for name in list(sys.modules):
            if name == tag or name.startswith(tag + '.'):
                sys.modules.pop(name, None)
