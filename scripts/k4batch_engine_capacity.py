"""Measure 128K shared/first-slot storage before admitting four real requests."""
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType


def run(session, arguments):
    source = Path(arguments['source_root'])
    manifest = json.loads((source/'manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    tag = '_engine_capacity_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    package = ModuleType(tag)
    package.__path__ = [str(source/'spark_infer_q')]
    sys.modules[tag] = package
    es = importlib.import_module(tag+'.engine_server')
    optimized = importlib.import_module(tag+'.kaifra_optimized')
    shared = importlib.import_module(tag+'.shared_graph')
    prefill = importlib.import_module(tag+'.prefill')
    ops = importlib.import_module(tag+'.prefill_tensorfold')
    experts = importlib.import_module(tag+'.prefill_experts')
    capacity = importlib.import_module(tag+'.long_capacity')
    torch = session.torch
    identities = (id(session.weights), id(session.draft), id(session.projection))
    report = dict(status='RUNNING_CAPACITY_PROBE', worker_pid=os.getpid(), model_reloaded=False,
                  source_sha256=manifest, identities=identities, profiles=[])

    def save():
        temp = source/'engine-capacity-result.tmp'
        temp.write_text(json.dumps(report, indent=2)+'\n')
        temp.replace(source/'engine-capacity-result.json')

    save()
    try:
        reference = importlib.import_module(tag+'.dflash2_reference')
        report['before_fused_weight_views'] = es.check_memory(torch)
        reference.share_fused_weights(session.draft)
        gc.collect()
        torch.cuda.synchronize()
        report['after_fused_weight_views'] = es.check_memory(torch)
        save()
        binary = session.root.parent/'e44-prefill-20261024-build-v1/build/binary/spark_q_e44_pipeline_v1.so'
        tile = experts.PrefillExpertTile.from_file(binary, expected_sha256=experts.E44_BINARY_SHA256)
        with torch.inference_mode():
            for ring, chunk in ((2048, 512),):
                target = provider = executor = None
                gc.collect()
                torch.cuda.synchronize()
                row = dict(index_ring_rows=ring, chunk_rows=chunk, before=es.check_memory(torch))
                report['profiles'].append(row)
                save()
                try:
                    provider, _ = optimized.optimized_provider(session, history_window_rows=2040,
                        max_request_context=131088, query_graphs=True, request_projection_cache=True)
                    provider.worker.warmup()
                    executor = prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights, torch,
                        capacity=131088, expert_backend=tile, index_ring_rows=ring), chunk_rows=chunk)
                    executor.workspace = executor.provider.allocate(chunk)
                    target = shared.SharedGraphTarget.from_session(session, capacity=131088,
                        kv_dtype='bf16', slots=4, index_ring_rows=ring)
                    for width in range(1, 9):
                        target.screen.prepare(width)
                    row['shared'] = es.check_memory(torch)
                    target.targets[0].warmup()
                    row['first_slot'] = es.check_memory(torch)
                    private = row['first_slot']['allocated_bytes']-row['shared']['allocated_bytes']
                    feature_bytes = 2*2040*12800*2 + sum(getattr(provider.worker, name).numel()
                        *getattr(provider.worker, name).element_size()
                        for name in ('_context_rows', '_keys', '_values'))
                    reserve = max(1024**3, row['first_slot']['reserved_bytes']-row['first_slot']['allocated_bytes'])
                    budget = capacity.LongCapacityBudget(row['shared']['allocated_bytes'],
                        private+feature_bytes, reserve, 88*1024**3, 8*1024**3)
                    row.update(private_bytes=private, feature_bytes=feature_bytes, budget=vars(budget),
                        admission=budget.evaluate(4, allocated_bytes=row['first_slot']['allocated_bytes'],
                            reserved_bytes=row['first_slot']['reserved_bytes'],
                            host_available_bytes=row['first_slot']['host_available_bytes']))
                    save()
                finally:
                    if target is not None:
                        target.close()
                    if executor is not None:
                        executor.close()
                    if provider is not None:
                        provider.worker.close()
                    target = provider = executor = None
                    gc.collect()
                    torch.cuda.synchronize()
                    row['after_cleanup'] = es.check_memory(torch)
                    save()
        assert identities == (id(session.weights), id(session.draft), id(session.projection))
        report['status'] = 'PASS_CAPACITY_PROBE'
        save()
        return report
    except BaseException as exc:
        report.update(status='FAIL_CAPACITY_PROBE', error=type(exc).__name__+': '+str(exc))
        save()
        raise
    finally:
        for name in list(sys.modules):
            if name == tag or name.startswith(tag+'.'):
                sys.modules.pop(name, None)
