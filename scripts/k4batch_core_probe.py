"""Read-only CPU placement/frequency corroboration of K4's wait intervention.

This is a diagnostic, not a throughput benchmark. It never changes CPU affinity,
frequency, governor or CUDA scheduling flags. Sysfs samples surround each draft
outside its call timer; the complete decode timer includes their overhead.
"""
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import ModuleType


def _read_integer(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _cpu_observer():
    """Return Linux sched_getcpu/sysfs observations without changing policy."""
    library = ctypes.CDLL(None, use_errno=True)
    try:
        getcpu = library.sched_getcpu
    except AttributeError as error:
        raise RuntimeError('CPU probe requires libc.sched_getcpu') from error
    getcpu.argtypes = []
    getcpu.restype = ctypes.c_int
    root = Path('/sys/devices/system/cpu')
    topology = {}
    for path in sorted(root.glob('cpu[0-9]*'), key=lambda p: int(p.name[3:])):
        cpu = int(path.name[3:])
        topology[str(cpu)] = dict(
            capacity=_read_integer(path / 'cpu_capacity'),
            cpuinfo_max_freq_khz=_read_integer(path / 'cpufreq/cpuinfo_max_freq'),
            scaling_max_freq_khz=_read_integer(path / 'cpufreq/scaling_max_freq'),
            core_id=_read_integer(path / 'topology/core_id'),
            cluster_id=_read_integer(path / 'topology/cluster_id'),
            package_id=_read_integer(path / 'topology/physical_package_id'))
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None

    def sample():
        started = time.perf_counter()
        cpu = int(getcpu())
        if cpu < 0:
            raise OSError(ctypes.get_errno(), 'sched_getcpu failed')
        path = root / ('cpu' + str(cpu)) / 'cpufreq'
        frequency = _read_integer(path / 'scaling_cur_freq')
        source = 'scaling_cur_freq'
        if frequency is None:
            frequency = _read_integer(path / 'cpuinfo_cur_freq')
            source = 'cpuinfo_cur_freq' if frequency is not None else None
        after = int(getcpu())
        if after < 0:
            raise OSError(ctypes.get_errno(), 'sched_getcpu failed')
        return dict(cpu=cpu, cpu_after_read=after,
            stable_during_sample=cpu == after, frequency_khz=frequency,
            frequency_source=source, sample_ms=(time.perf_counter() - started) * 1000)

    return sample, topology, affinity


def _summarize_drafts(rows, topology):
    by_cpu = {}
    for row in rows:
        cpu = str(row['cpu_start'])
        by_cpu.setdefault(cpu, []).append(row)
    result = {}
    for cpu, drafts in sorted(by_cpu.items(), key=lambda pair: int(pair[0])):
        frequencies = [sample['frequency_khz'] for row in drafts
            for sample in (row['before'], row['after'])
            if sample['stable_during_sample'] and sample['cpu'] == int(cpu)
            and sample['frequency_khz'] is not None]
        result[cpu] = dict(count=len(drafts), topology=topology.get(cpu),
            median_call_wall_ms=statistics.median(row['call_wall_ms'] for row in drafts),
            median_thread_cpu_ms=statistics.median(row['thread_cpu_ms'] for row in drafts),
            changed_cpu_during_call=sum(row['cpu_start'] != row['cpu_end'] for row in drafts),
            frequency_samples=len(frequencies),
            median_frequency_khz=statistics.median(frequencies) if frequencies else None,
            minimum_frequency_khz=min(frequencies) if frequencies else None,
            maximum_frequency_khz=max(frequencies) if frequencies else None)
    return result


def run(session, arguments):
    root = Path(arguments['source_root'])
    manifest = json.loads((root / 'manifest.json').read_text())

    def check_sources():
        for name, digest in manifest.items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name

    check_sources()
    sample_cpu, topology, affinity = _cpu_observer()
    tag = '_k4_core_probe_' + hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()[:16]
    package = ModuleType(tag)
    package.__path__ = [str(root / 'spark_infer_q')]
    sys.modules[tag] = package
    modes = ('optimized_single', 'batch_sleep', 'batch_event')
    orders = (modes, tuple(reversed(modes)))
    target = solo = None
    gold = policy = None
    identity = id(session.weights), id(session.draft)
    report = dict(status='RUNNING', source_sha256=manifest, model_reloaded=False,
        production_changed=False, cpu_topology=topology, allowed_cpu_ids=affinity,
        checks=[], timings=[], contract=dict(modes=list(modes), repetitions=2,
            max_tokens=32, policy='unchanged E17/W8/E33 and per-request post-draft budget',
            intervention='only wait after incomplete batch poll: explicit 0.5 ms sleep or current event synchronize',
            timing='diagnostic decode wall after first token to retirement includes CPU observation overhead',
            draft_timing='proposal call wall/thread CPU exclude entry and exit sysfs observations',
            cpu='sched_getcpu at entry/exit; frequency is a sysfs sample, not a continuous hardware clock measurement',
            operating_system='no affinity, governor, frequency or CUDA scheduling changes'))

    def save(active=None):
        if active is not None:
            report['active'] = active
        path = root / 'core-probe-result.json'
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
            [dict(role='user', content='请用三点解释缓存与队列的区别。')],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        report['prompt_sha256'] = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        prior = session.calibration['conditional_prior']

        with native_module.NativeScheduler(session.root.parent / 'DeepInfer',
                session.root.parent / 'build-release/lib/deepinfer', max_drafts=7) as native:
            def generate(mode, reference=False, reject_new_jit=False):
                nonlocal gold, policy
                if mode == 'optimized_single':
                    engine = spec.speculative_engine(native, solo, provider,
                        eos_ids=session.eos, prior=prior, capacity=768, mode='adaptive')
                else:
                    engine = batch.batch_engine(native, target, provider,
                        eos_ids=session.eos, prior=prior, capacity=768)
                with engine:
                    engine.submit_tokens('core-probe', ids, max_tokens=32)
                    bridge = engine.bridge
                    chosen_provider = bridge.backend.provider
                    tokens, trace, drafts = [], [], []
                    while not tokens:
                        for event in engine.step():
                            tokens.extend(event.token_ids)
                    assert len(tokens) == 1
                    original_propose = chosen_provider.propose
                    original_execute = bridge.execute
                    waits = dict(incomplete_polls=0, event_waits=0)

                    def propose(handle, state, anchor, remaining):
                        envelope_started = time.perf_counter()
                        before = sample_cpu()
                        call_started = time.perf_counter()
                        thread_started = time.thread_time()
                        result = original_propose(handle, state, anchor, remaining)
                        thread_ms = (time.thread_time() - thread_started) * 1000
                        call_ms = (time.perf_counter() - call_started) * 1000
                        after = sample_cpu()
                        drafts.append(dict(context_rows=len(state.context), remaining=remaining,
                            call_wall_ms=call_ms, thread_cpu_ms=thread_ms,
                            envelope_ms=(time.perf_counter() - envelope_started) * 1000,
                            cpu_start=before['cpu_after_read'], cpu_end=after['cpu'],
                            before=before, after=after))
                        return result

                    def execute(plan):
                        if mode == 'optimized_single':
                            return original_execute(plan)
                        bridge.execute_async(plan)
                        while not bridge.poll(plan):
                            waits['incomplete_polls'] += 1
                            if mode == 'batch_sleep':
                                # Deliberately preserve the regression control,
                                # independent of the production execute helper.
                                time.sleep(.0005)
                                continue
                            execution = plan.execution
                            while isinstance(execution, batch.SplitExecution):
                                execution = execution.current
                            if execution.phase == 'verify':
                                event = execution.target_ticket.event
                            else:
                                assert execution.phase == 'commit', execution.phase
                                event = execution.event
                            assert event is not None
                            event.synchronize()
                            waits['event_waits'] += 1

                    screen = solo.screen if mode == 'optimized_single' else target.screen
                    jit_before = set(screen.kernel_sha256)
                    chosen_provider.propose = propose
                    bridge.execute = execute
                    started = time.perf_counter()
                    try:
                        while engine.active_requests:
                            for event in engine.step():
                                tokens.extend(event.token_ids)
                            trace.extend({key: metric[key] for key in
                                ('context_rows', 'verified_rows', 'emitted', 'accepted', 'observed', 'rates')}
                                for metric in engine.last_step_metrics if not metric['prompt'])
                        torch.cuda.synchronize()
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        assert not bridge.backend.states and not bridge.native.has_unfinished_seqs()
                        assert bridge.blocks.get_num_free_blocks() == bridge.block_count - 1
                        if reject_new_jit:
                            assert not set(screen.kernel_sha256) - jit_before, 'new head JIT shape'
                        if reference:
                            gold, policy = tokens, trace
                        else:
                            assert tokens == gold and trace == policy, ('output/policy differs', mode)
                        return dict(mode=mode, decode_ms=elapsed_ms,
                            diagnostic_decode_tps=(len(tokens) - 1) * 1000 / elapsed_ms,
                            completion_tokens=len(tokens), output_policy_exact=True,
                            widths=[metric['verified_rows'] for metric in trace],
                            drafts=drafts, waits=waits,
                            proposal_call_wall_ms=sum(row['call_wall_ms'] for row in drafts),
                            proposal_thread_cpu_ms=sum(row['thread_cpu_ms'] for row in drafts),
                            observation_ms=sum(row['before']['sample_ms'] + row['after']['sample_ms']
                                for row in drafts),
                            by_cpu=_summarize_drafts(drafts, topology))
                    finally:
                        chosen_provider.__dict__.pop('propose', None)
                        bridge.__dict__.pop('execute', None)

            for mode in modes:
                save(dict(mode=mode, phase='warmup_correctness'))
                report['checks'].append(generate(mode, reference=mode == 'optimized_single'))
                save()
            for repetition, order in enumerate(orders):
                for position, mode in enumerate(order):
                    save(dict(mode=mode, repetition=repetition, order=position, phase='observation'))
                    result = generate(mode, reject_new_jit=True)
                    result.update(repetition=repetition, order=position)
                    report['timings'].append(result)
                    save()

        assert len(solo.graphs) == 16 and len(target.graphs) == 80 and not target.fallbacks
        assert identity == (id(session.weights), id(session.draft))
        check_sources()
        summary = {}
        for mode in modes:
            runs = [row for row in report['timings'] if row['mode'] == mode]
            drafts = [draft for row in runs for draft in row['drafts']]
            summary[mode] = dict(runs=len(runs), drafts=len(drafts),
                mean_proposal_call_wall_ms=statistics.mean(row['proposal_call_wall_ms'] for row in runs),
                median_draft_call_wall_ms=statistics.median(row['call_wall_ms'] for row in drafts),
                changed_cpu_during_call=sum(row['cpu_start'] != row['cpu_end'] for row in drafts),
                by_cpu=_summarize_drafts(drafts, topology))
        report.update(status='PASS_K4_CPU_CORE_PROBE', summary=summary,
            model_identity_preserved=True, gloo_released=not torch.distributed.is_initialized(),
            limitations=['single short Chinese request, two repetitions; diagnostic-only timing',
                'sysfs reads perturb placement and frequency; their overhead is included in decode wall',
                'entry/exit CPU equality cannot exclude migration during a call',
                'missing capacity/frequency fields mean unavailable, not identical cores or zero frequency',
                'observed association alone does not establish migration versus frequency as the OS mechanism'])
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
