"""Inventory prefill storage and build a pinned native prompt-phase control."""
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def run(session, arguments):
    source = Path(arguments['source_root'])
    manifest = json.loads((source/'manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    tag = '_engine_batch_memory_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    package = ModuleType(tag)
    package.__path__ = [str(source/'spark_infer_q')]
    sys.modules[tag] = package
    torch = session.torch
    provider_module = importlib.import_module(tag+'.prefill_tensorfold')
    es = importlib.import_module(tag+'.engine_server')
    native_module = importlib.import_module(tag+'.native_scheduler')
    bridge_module = importlib.import_module(tag+'.deepinfer_bridge')
    report = dict(status='RUNNING_BATCH_MEMORY', source_sha256=manifest,
        worker_pid=os.getpid(), target_object_id=id(session.weights), draft_object_id=id(session.draft),
        model_reloaded=False, allocations=[], before=es.check_memory(torch))

    def save(phase):
        report['phase'] = phase
        temp = source/'engine-batch-memory-result.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(source/'engine-batch-memory-result.json')

    def inventory(value, prefix='', seen=None):
        seen = set() if seen is None else seen
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            if key in seen:
                return []
            seen.add(key)
            return [dict(name=prefix, shape=list(value.shape), dtype=str(value.dtype),
                device=str(value.device), storage_bytes=storage.nbytes())]
        if not hasattr(value, '__dict__'):
            return []
        return [row for name, child in vars(value).items()
            for row in inventory(child, prefix+'.'+name if prefix else name, seen)]

    save('inventory')
    workspace = None
    try:
        cfg = session.weights.cfg
        names = ('hidden', 'streams', 'low', 'heads', 'kv_heads', 'head_dim',
            'index_heads', 'index_dim', 'index_budget', 'index_ratio', 'nk', 'nv',
            'conv_dim', 'conv_kernel', 'ple_dim', 'heads_per_ngram', 'top_k', 'experts', 'moe_width')
        report['geometry'] = {name: getattr(cfg, name) for name in names}
        report['native_mtp_present'] = session.weights.mtp is not None
        report['ple_table_bits'] = [getattr(l.ple.table, 'bits', 4)
            for l in session.weights.layers if l.ple is not None]
        with torch.inference_mode():
            provider = provider_module.TensorFoldPrefillProvider(session.weights, torch,
                capacity=131088, index_ring_rows=2048)
            workspace = provider.allocate(1024)
            torch.cuda.synchronize()
            tensors = inventory(workspace)
            report['allocations'] = sorted(tensors, key=lambda row: row['storage_bytes'], reverse=True)
            report['cuda_storage_bytes'] = sum(row['storage_bytes'] for row in tensors if row['device'].startswith('cuda'))
            report['allocated'] = es.check_memory(torch)
            workspace = None
            gc.collect()
            torch.cuda.synchronize()
            report['after_inventory'] = es.check_memory(torch)
        save('build-control')
        from torch.utils.cpp_extension import load
        upstream = session.root.parent/'DeepInfer'
        report['native_header_sha256'] = {name: hashlib.sha256((upstream/name).read_bytes()).hexdigest()
            for name in ('csrc/engine/scheduler.hpp', 'csrc/engine/sequence.hpp', 'csrc/engine/pybind_scheduler.cpp')}
        build = source/'build/prompt-control'
        build.mkdir(parents=True)
        previous_jobs = os.environ.get('MAX_JOBS')
        os.environ['MAX_JOBS'] = '1'
        try:
            control = load(name='kaifra_prompt_control_v1', sources=[str(source/'scripts/k4batch_prompt_control.cpp')],
                build_directory=str(build), extra_include_paths=[str(upstream/'csrc')],
                extra_cflags=['-O2', '-g0', '-std=c++20', '-fvisibility=hidden'], with_cuda=False, verbose=True)
        finally:
            if previous_jobs is None:
                os.environ.pop('MAX_JOBS', None)
            else:
                os.environ['MAX_JOBS'] = previous_jobs
        with native_module.NativeScheduler(upstream, session.root.parent/'build-release/lib/deepinfer', max_drafts=7) as native:
            bridge = bridge_module.SchedulerBridge(native.core, torch, native.group, native.sampling_class,
                SimpleNamespace(), vocab_size=provider.vocab_size, eos_ids=session.eos,
                capacity=131072, max_requests=4, token_budget=4096, max_drafts=7)
            try:
                assert control.mode(bridge.native) == 0
                control.set_prompt_mode(bridge.native, True)
                assert control.mode(bridge.native) == 1
                control.set_prompt_mode(bridge.native, False)
                assert control.mode(bridge.native) == 0
                report['native'] = native.metadata
                report['control_binary_sha256'] = hashlib.sha256(Path(control.__file__).read_bytes()).hexdigest()
                report['control_binary'] = str(Path(control.__file__).relative_to(source))
                report['native_mode_roundtrip'] = True
            finally:
                control.set_prompt_mode(bridge.native, False)
                bridge.close()
        report['status'] = 'PASS_BATCH_MEMORY'
        save('complete')
    except BaseException as exc:
        report.update(status='FAIL_BATCH_MEMORY', error=type(exc).__name__+': '+str(exc)[:3000])
        save('failed')
        raise
    finally:
        workspace = None
        for name in list(sys.modules):
            if name == tag or name.startswith(tag+'.'):
                sys.modules.pop(name, None)
        gc.collect()
        torch.cuda.synchronize()
    return report
