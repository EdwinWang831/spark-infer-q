"""Compare a real 128K BF16 prefix against the full raw-index reference."""
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
    tag = '_engine_index_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    package = ModuleType(tag)
    package.__path__ = [str(source/'spark_infer_q')]
    sys.modules[tag] = package
    es = importlib.import_module(tag+'.engine_server')
    long = importlib.import_module(tag+'.long_target')
    optimized = importlib.import_module(tag+'.kaifra_optimized')
    prefill = importlib.import_module(tag+'.prefill')
    ops = importlib.import_module(tag+'.prefill_tensorfold')
    bounded = importlib.import_module(tag+'.bounded_index')
    helper_spec = importlib.util.spec_from_file_location(tag+'.helpers',source/'scripts/k4batch_engine_long.py')
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    torch = session.torch
    report = dict(status='RUNNING_ENGINE_INDEX', worker_pid=os.getpid(), model_reloaded=False,
        target_object_id=id(session.weights), draft_object_id=id(session.draft), source_sha256=manifest,
        memory_samples=[], prefill=[], numeric=[], rollback=[])
    actual = expected = candidate = reference = None
    state = control = None

    def save(phase):
        report['phase'] = phase
        temp = source/'engine-index-result.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(source/'engine-index-result.json')

    def memory(phase):
        row = dict(phase=phase, **es.check_memory(torch))
        report['memory_samples'].append(row)

    def compare_outputs(a, b):
        assert torch.equal(a.logits, b.logits), 'full/ring prefill logits'
        assert torch.equal(a.features, b.features), 'full/ring five-layer features'

    save('prepare')
    try:
        with torch.inference_mode():
            actual = optimized.optimized_target(session, capacity=131088, kv_dtype='bf16', index_ring_rows=2048)
            actual.warmup()
            expected = long.LongTarget(session.weights, capacity=131088, kv_dtype='bf16')
            state, control = actual.state(), expected.state()
            candidate = prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights, torch,
                capacity=131088, index_ring_rows=2048), chunk_rows=512)
            reference = prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights, torch,
                capacity=131088), chunk_rows=512)
            # Odd and boundary-crossing chunks exercise incomplete pool blocks.
            for length in (1023, 2, 1024, 7):
                ids = [83+index%100 for index in range(length)]
                a, b = candidate.advance(state, ids), reference.advance(control, ids)
                compare_outputs(a, b)
                assert not bounded.state_difference(torch, state, control)
                report['prefill'].append(dict(position=state.pos, rows=length, exact=True))
                save('boundary:'+str(state.pos))
            seed, gold = state.snapshot(), control.snapshot()
            prefix = helper.state_sha(torch, state)
            assert prefix == helper.state_sha(torch, control)
            for length, keep in ((8, 1), (7, 3), (2, 1)):
                ids = [91]*length
                a, hidden, buffers = actual.run(state, ids)
                actual.commit(state, buffers, length, keep)
                b, features, scratch = expected.run(control, ids, capture=True)
                expected.commit(control, scratch, length, keep)
                assert torch.equal(a, b) and torch.equal(hidden, features)
                assert not bounded.state_difference(torch, state, control)
                state.restore(seed)
                control.restore(gold)
                assert prefix == helper.state_sha(torch, state) == helper.state_sha(torch, control)
                report['rollback'].append(dict(position=state.pos, rows=length, keep=keep, exact=True))
            state.reset(session.weights)
            control.reset(session.weights)
            item = helper.make_messages(session, 0, prompt_tokens=131008)
            assert item['prompt_tokens'] == 131008
            report['input_sha256'] = item['input_sha256']
            started = time.perf_counter()
            for at in range(0, len(item['ids']), 512):
                ids = item['ids'][at:at+512]
                a, b = candidate.advance(state, ids), reference.advance(control, ids)
                compare_outputs(a, b)
                if at % 16384 == 0:
                    memory('fill:'+str(at))
                    save('fill:'+str(at))
            assert not bounded.state_difference(torch, state, control)
            assert helper.state_sha(torch, state) == helper.state_sha(torch, control)
            report['full_prefix'] = dict(position=state.pos, chunks=(len(item['ids'])+511)//512,
                all_chunk_logits_and_features_exact=True, all_valid_state_exact=True,
                diagnostic_seconds=time.perf_counter()-started)
            memory('full-prefix')
            save('128k-windows')
            seed, gold = state.snapshot(), control.snapshot()
            prefix = helper.state_sha(torch, state)
            for parity in (0, 1):
                for width in range(1, 9):
                    state.restore(seed)
                    control.restore(gold)
                    for value in (state, control):
                        value.rec[parity].copy_(value.rec[value.cur[0]])
                        value.cur = [parity]*len(value.cur)
                    ids = [83+index for index in range(width)]
                    keep = width if width%2 else width//2
                    a, hidden, buffers = actual.run(state, ids)
                    actual.commit(state, buffers, width, keep)
                    b, features, scratch = expected.run(control, ids, capture=True)
                    expected.commit(control, scratch, width, keep)
                    assert torch.equal(a, b) and torch.equal(hidden, features)
                    assert not bounded.state_difference(torch, state, control)
                    state.restore(seed)
                    control.restore(gold)
                    assert prefix == helper.state_sha(torch, state) == helper.state_sha(torch, control)
                    report['numeric'].append(dict(position=state.pos, width=width, keep=keep, parity=parity, exact=True))
            memory('numeric-done')
            report['status'] = 'PASS_ENGINE_INDEX'
            save('complete')
        return report
    except BaseException as exc:
        report.update(status='FAIL_ENGINE_INDEX', error=type(exc).__name__+': '+str(exc))
        save('failed')
        raise
    finally:
        if actual is not None:
            if actual.leased:
                actual.release_state(state)
            actual.close()
        for executor in (candidate, reference):
            if executor is not None:
                executor.close()
        actual = expected = candidate = reference = state = control = None
        for name in list(sys.modules):
            if name == tag or name.startswith(tag+'.'):
                sys.modules.pop(name, None)
        gc.collect()
        torch.cuda.synchronize()
