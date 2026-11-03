"""Native four-request prefill, matched against a sealed v0.4.1 engine."""
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType


def run(session, arguments):
    source = Path(arguments['source_root'])
    frozen = session.root.parent/'k4batch-20261023-engine-v041-20261103-live-v1'
    manifest = json.loads((source/'manifest.json').read_text())
    old_manifest = json.loads((frozen/'manifest.json').read_text())
    for root, files in ((source, manifest), (frozen, old_manifest)):
        for name, digest in files.items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    tag = '_engine_batch_study_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    old_tag = tag+'_baseline'
    for name, root in ((tag, source), (old_tag, frozen)):
        package = ModuleType(name)
        package.__path__ = [str(root/'spark_infer_q')]
        sys.modules[name] = package
    es = importlib.import_module(tag+'.engine_server')
    shared = importlib.import_module(tag+'.shared_graph')
    optimized = importlib.import_module(tag+'.kaifra_optimized')
    native_module = importlib.import_module(tag+'.native_scheduler')
    batch = importlib.import_module(tag+'.prefill_batch')
    experts = importlib.import_module(tag+'.prefill_experts')
    control_module = importlib.import_module(tag+'.prompt_batch')
    capacity = importlib.import_module(tag+'.long_capacity')
    old_prefill = importlib.import_module(old_tag+'.prefill')
    old_ops = importlib.import_module(old_tag+'.prefill_tensorfold')
    engines = {arm: importlib.import_module(prefix+'.kaifra_speculative')
        for arm, prefix in (('baseline', old_tag), ('candidate', tag))}
    helper_spec = importlib.util.spec_from_file_location(tag+'.helpers', source/'scripts/k4batch_engine_long.py')
    helpers = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helpers)
    torch = session.torch
    target = provider = executor = None
    report = dict(status='RUNNING_BATCH_STUDY', source_sha256=manifest,
        baseline_source_sha256=old_manifest, worker_pid=os.getpid(), model_reloaded=False,
        target_object_id=id(session.weights), draft_object_id=id(session.draft),
        runs=[], warmups=[], memory=[], lifecycle=[])

    def save(phase):
        report['phase'] = phase
        temp = source/'engine-batch-study-result.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(source/'engine-batch-study-result.json')

    def memory(phase):
        value = es.check_memory(torch)
        report['memory'].append(dict(phase=phase, **value))
        return value

    def allocate(arm):
        if arm == 'baseline':
            return old_prefill.PrefillExecutor(old_ops.TensorFoldPrefillProvider(session.weights, torch,
                capacity=131088, index_ring_rows=2048, expert_backend=tile), chunk_rows=1024)
        return batch.BatchPrefillExecutor(batch.BatchPrefillProvider(session.weights, torch,
            capacity=131088, index_ring_rows=2048, expert_backend=tile), chunk_rows=1024)

    save('prepare')
    try:
        numeric_path = session.root.parent/'k4batch-20261023-engine-batch-numeric-20261103-v1/engine-batch-numeric-result.json'
        numeric = json.loads(numeric_path.read_text())
        assert numeric['status'] == 'PASS_BATCH_NUMERIC'
        assert numeric['target_object_id'] == id(session.weights) and numeric['draft_object_id'] == id(session.draft)
        for name in ('prefill.py', 'prefill_tensorfold.py', 'prefill_batch.py', 'prefill_workspace.py'):
            assert numeric['source_sha256']['spark_infer_q/'+name] == manifest['spark_infer_q/'+name]
        report['numeric_gate_sha256'] = hashlib.sha256(numeric_path.read_bytes()).hexdigest()
        reference_path = session.root.parent/'k4batch-20261023-engine-context-20261103-v3/engine-context-result.json'
        previous = json.loads(reference_path.read_text())
        assert previous['status'] == 'PASS_ENGINE_CONTEXT'
        assert previous['target_object_id'] == id(session.weights) and previous['draft_object_id'] == id(session.draft)
        report['reference_gate_sha256'] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        binary = session.root.parent/'e44-prefill-20261024-build-v1/build/binary/spark_q_e44_pipeline_v1.so'
        tile = experts.PrefillExpertTile.from_file(binary, expected_sha256=experts.E44_BINARY_SHA256)
        control_path = session.root.parent/'k4batch-20261023-engine-batch-memory-20261103-v1/build/prompt-control/kaifra_prompt_control_v1.so'
        inputs = [helpers.make_messages(session, i, prompt_tokens=131008) for i in range(4)]
        report['inputs'] = [{k: v for k, v in item.items() if k not in ('messages', 'ids')} for item in inputs]
        assert report['inputs'] == previous['inputs']
        with torch.inference_mode():
            provider, report['profile'] = optimized.optimized_provider(session, history_window_rows=2040,
                max_request_context=131088, query_graphs=True, request_projection_cache=True,
                feature_history_layout='linear')
            provider.worker.warmup()
            target = shared.SharedGraphTarget.from_session(session, capacity=131088,
                kv_dtype='bf16', slots=4, index_ring_rows=2048)
            for width in range(1, 9):
                target.screen.prepare(width)
            executor = allocate('candidate')
            executor.workspace = executor.provider.allocate(executor.workspace_rows)
            report['initial_admission'] = es.warm_long_slots(target, provider, torch)
            executor.close()
            executor = None
            gc.collect()
            memory('four-slots-ready')
            # Exact-head subset compilation stays outside every timed arm.
            for rows in range(1, 9):
                hidden = torch.zeros((rows, target.screen.k), device=session.weights.device, dtype=torch.bfloat16)
                for count in range(1, 129):
                    target.screen.subset(hidden, torch.arange(count, device=hidden.device))
            del hidden
            with native_module.NativeScheduler(session.root.parent/'DeepInfer',
                    session.root.parent/'build-release/lib/deepinfer', max_drafts=7) as native:
                control = control_module.PromptBatchControl.from_file(control_path, native)
                report['native'] = native.metadata
                report['prompt_control_sha256'] = control_module.CONTROL_SHA256

                def generate(arm, prompts, limit, name, *, warmup=False, cancel_prepared=False):
                    nonlocal executor
                    executor = allocate(arm)
                    executor.workspace = executor.provider.allocate(getattr(executor, 'workspace_rows', 1024))
                    snap = memory(name+':allocated')
                    feature_bytes = 2040*12800*2+sum(getattr(provider.worker, field).numel()
                        *getattr(provider.worker, field).element_size() for field in ('_context_rows', '_keys', '_values'))
                    budget = capacity.LongCapacityBudget(snap['allocated_bytes'], feature_bytes,
                        1024**3, 88*1024**3, 8*1024**3)
                    admission = budget.require(4, allocated_bytes=snap['allocated_bytes'],
                        reserved_bytes=snap['reserved_bytes'], host_available_bytes=snap['host_available_bytes'])
                    retired, storage = {}, {}
                    outputs, events = ({str(i): [] for i in range(4)} for _ in range(2))
                    metrics, plans = [], []
                    capture0 = provider.worker.query_captures
                    head0 = dict(target.screen.kernel_sha256)
                    options = dict(prompt_control=control) if arm == 'candidate' else {}
                    try:
                        with engines[arm].speculative_engine(native, target, provider, eos_ids=session.eos,
                                prior=session.calibration['conditional_prior'], mode='adaptive', capacity=131072,
                                max_requests=4, token_budget=4096 if arm == 'candidate' else 1024,
                                tokenizer=session.tokenizer, prefill_executor=executor, output_cache=True,
                                serialized_postdraft=True, interleave_prefill=True, **options) as engine:
                            backend = engine.bridge.backend
                            original_release = backend.release

                            def release(handle, state):
                                retired[handle.request_id] = (state.target, state.context, state.context_start)
                                storage[handle.request_id] = 0 if state.feature_window is None else state.feature_window.nbytes
                                original_release(handle, state)

                            backend.release = release
                            try:
                                torch.cuda.synchronize()
                                torch.cuda.reset_peak_memory_stats()
                                started = time.perf_counter()
                                handles = [engine.submit_tokens(str(i), ids, max_tokens=limit)
                                    for i, ids in enumerate(prompts)]
                                for step in range(4096):
                                    if not engine.active_requests:
                                        break
                                    plan = engine.bridge.prepare()
                                    assert control.module.mode(engine.bridge.native) == 0
                                    if cancel_prepared and step == 0:
                                        assert handles[0] in [row.handle for row in plan.rows]
                                        engine.cancel(handles[0])
                                    plans.append(dict(prompt=bool(plan.native_output.prompt_run),
                                        requests=len(plan.rows), rows=sum(len(row.token_ids) for row in plan.rows),
                                        positions=[row.position for row in plan.rows],
                                        request_rows=[len(row.token_ids) for row in plan.rows]))
                                    engine.bridge.execute(plan)
                                    out = engine.finish_plan(plan)
                                    metrics.extend(engine.last_step_metrics)
                                    assert control.module.mode(engine.bridge.native) == 0
                                    for event in out:
                                        key = event.handle.request_id
                                        outputs[key].extend(event.token_ids)
                                        events[key].append(dict(emitted=len(event.token_ids), elapsed=event.elapsed_seconds,
                                            ttft=event.ttft_seconds, finish=event.finish_reason))
                                    if step % 32 == 0:
                                        sample = memory(name+':'+str(step))
                                        (source/'engine-batch-study-progress.json').write_text(json.dumps(dict(name=name,
                                            step=step, positions=[s.target.pos for s in backend.states.values()],
                                            outputs={k: len(v) for k, v in outputs.items()}, memory=sample))+'\n')
                                assert not engine.active_requests, 'bounded drain exhausted'
                                torch.cuda.synchronize()
                                wall = time.perf_counter()-started
                                # Complete state and feature hashes are outside the timer.
                                final = {key: dict(state=helpers.state_sha(torch, state),
                                    feature_sha256=None if context is None else helpers.tensor_sha(torch, context),
                                    context_start=start) for key, (state, context, start) in retired.items()}
                                assert len(final) == 4 and not backend.states and not target.owners
                                assert not engine.bridge.native.has_unfinished_seqs()
                                assert engine.bridge.blocks.get_num_free_blocks() == engine.bridge.block_count-1
                            finally:
                                backend.release = original_release
                        phases = dict(prefill=0., draft=0., verify=0., commit=0.)
                        for metric in metrics:
                            if metric['prompt']:
                                phases['prefill'] += metric['verify_ms']
                            else:
                                for label in ('draft', 'verify', 'commit'):
                                    phases[label] += metric[label+'_ms']
                        trace = {key: [{field: m[field] for field in
                            ('verified_rows', 'emitted', 'accepted', 'observed', 'context_rows', 'rates')}
                            for m in metrics if m['request_id'] == key and not m['prompt']] for key in outputs}
                        result = dict(name=name, arm=arm, warmup=warmup, wall_seconds=wall,
                            outputs=outputs, final=final, trace=trace, events=events, phase_ms=phases,
                            plans=plans, admission=admission, feature_storage_bytes=storage,
                            request_ttft_seconds={k: next((v['ttft'] for v in values if v['ttft'] is not None), None)
                                for k, values in events.items()},
                            request_completion_seconds={k: values[-1]['elapsed'] for k, values in events.items()},
                            max_output_gaps_seconds={k: max((b['elapsed']-a['elapsed']
                                for a, b in zip(values, values[1:])), default=0.) for k, values in events.items()},
                            draft_captures_added=provider.worker.query_captures-capture0,
                            head_kernels_unchanged=head0 == target.screen.kernel_sha256,
                            capabilities=target.capabilities, memory=memory(name+':done'))
                        result['last_ttft_seconds'] = max(v for v in result['request_ttft_seconds'].values() if v is not None)
                        result['prefill_plan_histogram'] = {str(n): sum(p['prompt'] and p['requests'] == n for p in plans)
                            for n in range(1, 5)}
                        result['prefill_capabilities'] = executor.capabilities
                        assert result['draft_captures_added'] == 0 and target.capabilities['graph_captures'] == 64
                        assert not any(target.capabilities['graph_fallbacks']) and not target.capabilities['head_fallbacks']
                        if not warmup:
                            assert result['head_kernels_unchanged'], 'formal exact-head JIT changed'
                        return result
                    finally:
                        executor.close()
                        executor = None
                        retired.clear()
                        gc.collect()
                        torch.cuda.synchronize()

                mixed = [item['ids'][:length] for item, length in zip(inputs, (17, 1025, 2049, 4097), strict=True)]
                for arm in ('baseline', 'candidate'):
                    save('mixed:'+arm)
                    report['warmups'].append(generate(arm, mixed, 16, 'mixed:'+arm, warmup=True))
                a, b = report['warmups']
                assert all(a[k] == b[k] for k in ('outputs', 'final', 'trace')), 'mixed native prompt mismatch'
                assert any(p['prompt'] and p['requests'] == 4 for p in b['plans'])
                for arm in ('baseline', 'candidate'):
                    save('cancel:'+arm)
                    report['lifecycle'].append(generate(arm, mixed, 16, 'cancel:'+arm,
                        warmup=True, cancel_prepared=True))
                a, b = report['lifecycle']
                assert a['outputs']['0'] == b['outputs']['0'] == []
                for key in ('1', '2', '3'):
                    assert all(a[field][key] == b[field][key] for field in ('outputs', 'final', 'trace'))
                # Warm both arms above the long-index radix transition.
                warm = [item['ids'][:32768]+item['ids'][-64:] for item in inputs]
                for arm in ('baseline', 'candidate'):
                    save('long-warmup:'+arm)
                    report['warmups'].append(generate(arm, warm, 16, 'long-warmup:'+arm, warmup=True))
                a, b = report['warmups'][-2:]
                assert all(a[k] == b[k] for k in ('outputs', 'final', 'trace')), 'long warmup mismatch'
                for index, arm in enumerate(('baseline', 'candidate', 'candidate', 'baseline')):
                    name = str(index)+':'+arm
                    save(name)
                    result = generate(arm, [item['ids'] for item in inputs], 64, name)
                    for key, values in result['outputs'].items():
                        gold = previous['target_only_controls'][key]
                        assert values == gold['tokens'] and result['final'][key]['state'] == gold['state']
                        assert result['final'][key]['feature_sha256'] == gold['feature_sha256']
                    if report['runs']:
                        assert all(result[k] == report['runs'][0][k] for k in ('outputs', 'final', 'trace'))
                    report['runs'].append(result)
                    save(name+':done')
                a, b, c, d = report['runs']
                pairs = ((a, b), (d, c))
                reduction = 1-(b['wall_seconds']+c['wall_seconds'])/(a['wall_seconds']+d['wall_seconds'])
                gate = dict(wall_reduction=reduction,
                    both_pairs_faster=all(y['wall_seconds'] < x['wall_seconds'] for x, y in pairs),
                    last_ttft_within_2_percent=all(y['last_ttft_seconds'] <= 1.02*x['last_ttft_seconds'] for x, y in pairs),
                    output_gaps_within_5_percent=all(max(y['max_output_gaps_seconds'].values()) <=
                        1.05*max(x['max_output_gaps_seconds'].values()) for x, y in pairs))
                gate['passed'] = reduction >= .10 and all(gate[k] for k in
                    ('both_pairs_faster', 'last_ttft_within_2_percent', 'output_gaps_within_5_percent'))
                report['promotion'] = gate
                report['status'] = 'PASS_BATCH_STUDY' if gate['passed'] else 'REJECT_BATCH_STUDY_COST'
                save('complete')
    except BaseException as exc:
        report.update(status='FAIL_BATCH_STUDY', error=type(exc).__name__+': '+str(exc)[:3000])
        save('failed')
        raise
    finally:
        if executor is not None:
            executor.close()
        if target is not None:
            for owner in tuple(target.owners.values()):
                target.release_state(owner.workspace)
            target.close()
        if provider is not None:
            provider.worker.close()
        for name in list(sys.modules):
            if name in (tag, old_tag) or name.startswith((tag+'.', old_tag+'.')):
                sys.modules.pop(name, None)
        gc.collect()
        torch.cuda.synchronize()
    return report
