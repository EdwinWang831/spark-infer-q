"""Profile real 128K prefill at early, middle and late context positions."""
from contextlib import contextmanager
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
    manifest = json.loads((source/'manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    tag = '_engine_hotspots_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    package = ModuleType(tag)
    package.__path__ = [str(source/'spark_infer_q')]
    sys.modules[tag] = package

    def script(name):
        spec = importlib.util.spec_from_file_location(tag+'.'+name, source/'scripts'/f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    helpers = script('k4batch_engine_long')
    recorders = script('e37_model')
    accounting = script('e37_accounting')
    es = importlib.import_module(tag+'.engine_server')
    shared = importlib.import_module(tag+'.shared_graph')
    prefill = importlib.import_module(tag+'.prefill')
    operators = importlib.import_module(tag+'.prefill_tensorfold')
    experts = importlib.import_module(tag+'.prefill_experts')
    windows = importlib.import_module(tag+'.feature_window')
    torch = session.torch
    target = executor = None
    report = dict(status='RUNNING_ENGINE_HOTSPOTS', worker_pid=os.getpid(),
        model_reloaded=False, target_object_id=id(session.weights), draft_object_id=id(session.draft),
        source_sha256=manifest, profiles=[], events=[], inputs=[], checks=[], memory_samples=[])

    def save(phase):
        report['phase'] = phase
        temp = source/'engine-hotspots-result.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(source/'engine-hotspots-result.json')

    @contextmanager
    def instrument(provider, recorder):
        # The forward namespace and provider are private to this experiment.
        originals = {name: getattr(provider, name) for name in ('_gdn_block', '_moe_block')}
        for name, category in (('_gdn_block', 'gdn_other'), ('_moe_block', 'moe')):
            original = originals[name]
            def wrapped(*args, _fn=original, _category=category, **kwargs):
                with recorder.span(_category):
                    return _fn(*args, **kwargs)
            setattr(provider, name, wrapped)
        try:
            with recorders.instrument(provider.ops, recorder):
                yield
        finally:
            for name, original in originals.items():
                setattr(provider, name, original)

    save('prepare')
    try:
        reference_path = session.root.parent/'k4batch-20261023-engine-context-20261103-v3/engine-context-result.json'
        reference = json.loads(reference_path.read_text())
        assert reference['status'] == 'PASS_ENGINE_CONTEXT'
        assert reference['target_object_id'] == id(session.weights)
        assert reference['draft_object_id'] == id(session.draft)
        report['reference_sha256'] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        tile = experts.PrefillExpertTile.from_file(arguments['prefill_experts'],
            expected_sha256=experts.E44_BINARY_SHA256)
        with torch.inference_mode():
            target = shared.SharedGraphTarget.from_session(session, capacity=131088,
                kv_dtype='bf16', slots=1, index_ring_rows=2048)
            target.warmup()
            provider = operators.TensorFoldPrefillProvider(session.weights, torch, capacity=131088,
                index_ring_rows=2048, expert_backend=tile)
            executor = prefill.PrefillExecutor(provider, chunk_rows=1024)
            for index in (0, 1):
                item = helpers.make_messages(session, index, prompt_tokens=131008)
                report['inputs'].append({k: v for k, v in item.items() if k not in ('ids', 'messages')})
                assert report['inputs'][-1] == reference['inputs'][index]
                state = target.state()
                history = windows.FeatureWindow(2040, layout='linear')
                chunks = []
                try:
                    for chunk, at in enumerate(range(0, len(item['ids']), 1024)):
                        ids = item['ids'][at:at+1024]
                        label = f'input{index}:position{at}'
                        save(label)
                        started = time.perf_counter()
                        if chunk in (1, 64, 126):
                            root = 'engine.hotspot.'+label
                            annotations = recorders.Annotations(torch)
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                                    profile_memory=False, with_stack=False) as prof:
                                with torch.profiler.record_function(root):
                                    with instrument(provider, annotations):
                                        result = executor.advance(state, ids)
                            trace = source/f'hotspots-{index}-{at}.json'
                            prof.export_chrome_trace(str(trace))
                            report['profiles'].append(dict(index=index, position=at, rows=len(ids),
                                trace=trace.name, trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                                root=root, cupti=accounting.analyze_trace(json.loads(trace.read_text()), root)))
                        elif chunk in (2, 65, 127):
                            recorder = recorders.StageRecorder(torch)
                            with recorder.span('control'):
                                with instrument(provider, recorder):
                                    result = executor.advance(state, ids)
                            report['events'].append(dict(index=index, position=at, rows=len(ids),
                                exclusive_stage_intervals=recorder.result()))
                        else:
                            result = executor.advance(state, ids)
                        chunks.append(dict(position=at, rows=len(ids), wall_seconds=time.perf_counter()-started,
                            diagnostic=chunk in (1, 2, 64, 65, 126, 127)))
                        history.append(result.features)
                        if chunk % 32 == 0:
                            report['memory_samples'].append(dict(phase=label, **es.check_memory(torch)))
                    token = int(result.logits[-1].argmax())
                    output = [token]
                    for _ in range(63):
                        logits, features, buffers = target.run(state, [token])
                        target.commit(state, buffers, 1, 1)
                        history.append(features)
                        token = int(logits[-1].argmax())
                        output.append(token)
                    gold = reference['target_only_controls'][str(index)]
                    actual = helpers.state_sha(torch, state)
                    feature_sha = helpers.tensor_sha(torch, history.view())
                    assert output == gold['tokens'], 'profiled output'
                    assert actual == gold['state'], 'profiled state'
                    assert feature_sha == gold['feature_sha256'], 'profiled retained features'
                    report['checks'].append(dict(index=index, output_exact=True, state_exact=True,
                        features_exact=True, chunks=chunks))
                finally:
                    target.release_state(state)
                del state, history, result, logits, features, buffers
                gc.collect()
            report['status'] = 'PASS_ENGINE_HOTSPOTS'
            save('complete')
    except BaseException as exc:
        report.update(status='FAIL_ENGINE_HOTSPOTS', error=type(exc).__name__+': '+str(exc)[:3000])
        save('failed')
        raise
    finally:
        if executor is not None:
            executor.close()
        if target is not None:
            for owner in tuple(target.owners.values()):
                target.release_state(owner.workspace)
            target.close()
        for name in list(sys.modules):
            if name == tag or name.startswith(tag+'.'):
                sys.modules.pop(name, None)
        gc.collect()
        torch.cuda.synchronize()
    return report
