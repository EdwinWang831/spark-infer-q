"""Screen exact writeback/norm fusion on the owned BF16 HC prefill path."""
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


def install_fused_hc(provider):
    """Use TensorFold's checked fusion through this provider's private namespace."""
    torch, f = provider.torch, provider.ops
    fused = f._hc_fuser(provider.w.device)
    if fused is None:
        raise RuntimeError('checked HC writeback/norm fusion unavailable')
    original = f.hc_block
    calls = []

    def block(hc, b, rows, eps, streams, low, mode, inject_prev, inject_out, h,
              branch=None, y=None, wts=None):
        if rows <= 16 or getattr(hc.down, 'kernel', None) != 'b16':
            return original(hc, b, rows, eps, streams, low, mode, inject_prev,
                inject_out, h, branch=branch, y=y, wts=wts)
        fused(h[:rows], b.pss[:rows], hc.scale, b.normed[:rows], b.xs_normed[:rows],
            streams, eps, mode, branch=branch, inject=inject_prev, y=y, wts=wts)
        got = f.bf16.matmul(b.normed[:rows], hc.down.b,
            out=torch.empty((rows, hc.down.n), dtype=torch.float32, device=h.device), f32=True)
        f.glue.hc_act(got, b.act[:rows], b.xs_act[:rows],
            inject_out[:rows] if hc.inject else None, streams, low)
        f._mm(b.act[:rows], hc.up, b.xs_act[:rows], b.up[:rows], b)
        f.glue.hc_mix(b.up[:rows], b.normed[:rows], b.mixed[:rows], b.xs_mixed[:rows], streams)
        calls.append(mode)

    f.hc_block = block
    return calls


def run(session, arguments):
    source = Path(arguments['source_root'])
    manifest = json.loads((source/'manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    tag = '_engine_hc_probe_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    package = ModuleType(tag)
    package.__path__ = [str(source/'spark_infer_q')]
    sys.modules[tag] = package
    prefill = importlib.import_module(tag+'.prefill')
    operators = importlib.import_module(tag+'.prefill_tensorfold')
    shared = importlib.import_module(tag+'.shared_graph')
    experts = importlib.import_module(tag+'.prefill_experts')
    es = importlib.import_module(tag+'.engine_server')
    spec = importlib.util.spec_from_file_location(tag+'.helpers', source/'scripts/k4batch_engine_long.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    torch = session.torch
    target = executor = None
    report = dict(status='RUNNING_ENGINE_HC_PROBE', worker_pid=os.getpid(), model_reloaded=False,
        target_object_id=id(session.weights), draft_object_id=id(session.draft),
        source_sha256=manifest, numeric=[], timings=[], memory=[])

    def save(phase):
        report['phase'] = phase
        temp = source/'engine-hc-probe-result.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(source/'engine-hc-probe-result.json')

    save('prepare')
    try:
        tile = experts.PrefillExpertTile.from_file(arguments['prefill_experts'],
            expected_sha256=experts.E44_BINARY_SHA256)
        with torch.inference_mode():
            target = shared.SharedGraphTarget.from_session(session, capacity=131088,
                kv_dtype='bf16', slots=2, index_ring_rows=2048)
            target.warmup()
            cases = [helpers.make_messages(session, i, prompt_tokens=131008) for i in (0, 1)]
            controls = {}
            for index, length in ((0, 1024), (1, 1025), (0, 4097)):
                states = [target.state(), target.state()]
                outputs = []
                try:
                    for arm, state in zip(('baseline', 'candidate'), states, strict=True):
                        provider = operators.TensorFoldPrefillProvider(session.weights, torch,
                            capacity=131088, index_ring_rows=2048, expert_backend=tile)
                        calls = install_fused_hc(provider) if arm == 'candidate' else []
                        executor = prefill.PrefillExecutor(provider, chunk_rows=1024)
                        result = executor.advance(state, cases[index]['ids'][:length])
                        outputs.append((result.logits, result.features))
                        executor.close()
                        executor = None
                        if arm == 'candidate':
                            assert calls and set(calls) == {0, 1, 2}
                    assert all(torch.equal(a, b) for a, b in zip(*outputs)), 'HC logits/features'
                    assert not target.eager.state_difference(*states), 'HC state'
                    controls[index, length] = helpers.tensor_sha(torch, outputs[0][1])
                    report['numeric'].append(dict(index=index, input_tokens=length, exact=True))
                finally:
                    for state in states:
                        target.release_state(state)
                del states, state, outputs, result
                gc.collect()
                report['memory'].append(es.check_memory(torch))
                save(f'numeric:{index}:{length}')
            for index in (0, 1):
                for repetition in range(4):
                    order = ('baseline', 'candidate') if repetition % 2 == 0 else ('candidate', 'baseline')
                    for arm in order:
                        provider = operators.TensorFoldPrefillProvider(session.weights, torch,
                            capacity=131088, index_ring_rows=2048, expert_backend=tile)
                        calls = install_fused_hc(provider) if arm == 'candidate' else []
                        executor = prefill.PrefillExecutor(provider, chunk_rows=1024)
                        executor.workspace = provider.allocate(1024)
                        state = target.state()
                        try:
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            result = executor.advance(state, cases[index]['ids'][:1024])
                            wall = time.perf_counter()-started
                            report['timings'].append(dict(index=index, repetition=repetition, arm=arm,
                                wall_seconds=wall, feature_sha256=helpers.tensor_sha(torch, result.features),
                                fused_calls=len(calls)))
                        finally:
                            target.release_state(state)
                            executor.close()
                            executor = None
                        del state, result
                    save(f'timing:{index}:{repetition}')
            report['totals'] = {arm: sum(r['wall_seconds'] for r in report['timings'] if r['arm'] == arm)
                for arm in ('baseline', 'candidate')}
            report['wall_reduction'] = 1-report['totals']['candidate']/report['totals']['baseline']
            for index in (0, 1):
                assert len({r['feature_sha256'] for r in report['timings'] if r['index'] == index}) == 1
            report['status'] = 'PASS_ENGINE_HC_PROBE'
            save('complete')
    except BaseException as exc:
        report.update(status='FAIL_ENGINE_HC_PROBE', error=type(exc).__name__+': '+str(exc)[:3000])
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
