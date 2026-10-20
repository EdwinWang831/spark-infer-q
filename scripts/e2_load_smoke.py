#!/usr/bin/env python3
"""Load the complete target, with original-layout MTP/vision adjunct storage.

N-gram tables remain mapped and are exercised by cross-shard row gathers.
This is a weight-loading experiment, not a quality or throughput benchmark.
"""
import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable unavailable')


def cuda_storage_bytes(value):
    """Include plain Python matrix faces; the upstream helper misses these."""
    import dataclasses
    import torch
    seen, storages = set(), {}
    def visit(v):
        if id(v) in seen:
            return
        seen.add(id(v))
        if isinstance(v, torch.Tensor):
            if v.is_cuda:
                storage = v.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
        elif dataclasses.is_dataclass(v):
            for field in dataclasses.fields(v):
                if field.name != 'table':  # n-gram mappings and thread pools are host objects
                    visit(getattr(v, field.name))
        elif isinstance(v, dict):
            for x in v.values():
                visit(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                visit(x)
        elif hasattr(v, '__dict__') and not isinstance(v, type):
            for x in vars(v).values():
                visit(x)
    visit(value)
    return sum(storages.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--layers', type=int, default=48, help='48 for the final full-target experiment')
    p.add_argument('--auxiliary', action='store_true', help='Also retain raw MTP/vision tensors on CUDA')
    p.add_argument('--forward-smoke', action='store_true', help='Exercise four complete-target token steps and restore/replay')
    a = p.parse_args()
    if not 1 <= a.layers <= 48:
        p.error('layers must be 1..48')
    root = a.root
    report = {'status': 'RUNNING', 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'requested_layers': a.layers, 'mtp_execution_enabled': False,
              'quality_evaluation': 'NOT_MEASURED', 'full_model_forward': 'NOT_RUN', 'layers': []}
    report['harness_sha256'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                               ['e2_load_smoke.py', 'e2_run_load.sh', 'e2_extract_expert.py', 'e2_target_forward.py',
                                'spark_infer_q/checkpoint.py', 'spark_infer_q/csf_loader.py']}

    def save():
        (root / 'artifacts/load-smoke.json').write_text(json.dumps(report, indent=2) + '\n')

    def guard():
        value = available()
        report['min_host_available_bytes'] = min(report.get('min_host_available_bytes', value), value)
        if value < 8 * 1024**3:
            raise RuntimeError('memory reserve guard: less than 8 GiB available')

    save()
    try:
        from spark_infer_q.checkpoint import Catalog, category
        guard()
        catalog = Catalog(a.checkpoint,
            config_sha256='b50dbc52a0c83b0cd2a44a372c1fb6b5e0e6827d22f2611c97d4d030f92f6b28',
            index_sha256='5fa166bbb1cf25adda455117563efa72a2072272001f459eedece131ffc913e1')
        report['catalog'] = catalog.summary()
        pin = json.loads((root / 'artifacts/source-manifest.json').read_text())
        for name, meta in pin.items():
            assert hashlib.sha256((root / 'reference/tensorfold' / name).read_bytes()).hexdigest() == meta['sha256']
        report['external_source_sha256'] = {n: m['sha256'] for n, m in pin.items()}
        sys.path.insert(0, str(root / 'reference'))
        for name in ['spark_q_moe_linear', 'spark_q_moe_plan', 'spark_q_moe_prompt']:
            sys.path.insert(0, str(root.parent / 'e2-moe/build' / name))
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        assert torch.__version__ == '2.13.0+cu132' and torch.cuda.get_device_capability() == (12, 1)
        assert os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '0'
        report.update(torch_version=torch.__version__, cuda_build=torch.version.cuda, capability=[12, 1],
                      tf32_override=os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'],
                      torch_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
        torch.cuda.set_per_process_memory_fraction(88 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        import spark_q_moe_linear, spark_q_moe_plan, spark_q_moe_prompt
        report['extensions_sha256'] = {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                       for m in [spark_q_moe_linear, spark_q_moe_plan, spark_q_moe_prompt]}
        from tensorfold.cuda import experts
        from tensorfold.cuda.nvfp4 import linear
        linear._ext = lambda: spark_q_moe_linear
        linear._prompt_ext = lambda: spark_q_moe_prompt
        experts._ext = lambda: spark_q_moe_plan
        from tensorfold.families.qwen4_exp.cuda import weights
        from spark_infer_q.csf_loader import reader_class
        lib = root.parent / 'e2-expert/artifacts/libcsf_reference.so'
        report['csf_library_sha256'] = hashlib.sha256(lib.read_bytes()).hexdigest()
        original_config = weights.Config.read
        if a.layers != 48:
            from dataclasses import replace
            weights.Config.read = lambda path: replace(original_config(path), layers=a.layers)
        start = time.perf_counter()

        def progress(count, rd):
            guard()
            entry = {'loaded_layers': count, 'elapsed_seconds': time.perf_counter() - start,
                     'torch_allocated_bytes': torch.cuda.memory_allocated(),
                     'torch_reserved_bytes': torch.cuda.memory_reserved(), 'host_available_bytes': available(),
                     'csf_decoded': rd.decoded, 'independent_python_scale_checks': rd.python_scale_checks}
            report['layers'].append(entry)
            save()
            print(json.dumps(entry), flush=True)

        cls = reader_class(weights._Reader, catalog, lib, progress, guard)
        weights._Reader = cls
        w = weights.load(a.checkpoint, mtp=False)
        torch.cuda.synchronize()
        assert len(w.layers) == a.layers
        rd = cls.instances[-1]
        assert rd.decoded == a.layers * 512 * 3
        report.update(target_load_seconds=time.perf_counter() - start, target_cuda_storage_bytes=cuda_storage_bytes(w),
                      upstream_reported_target_cuda_bytes=w.nbytes(),
                      target_layers=len(w.layers), centered_norms=not w.around_one,
                      routed_experts=sum(l.moe.experts.routed for l in w.layers),
                      routed_native_pack_bytes=sum(l.moe.experts.routed_experts.nbytes() for l in w.layers))
        assert [l.index for l in w.layers] == list(range(a.layers))
        for layer in w.layers:
            assert layer.moe.router.shape == (513, 2560) and torch.isfinite(layer.moe.router).all()
        tables = []
        mapped = set()
        import numpy as np
        for layer in w.layers:
            if layer.ple is None:
                continue
            table = layer.ple.table
            ids = np.array([i for lo, hi in zip(table.starts[:-1], table.starts[1:]) for i in (int(lo), int(hi)-1)], dtype=np.int64)
            rows = table.gather(ids)
            values = torch.from_numpy(rows.copy()).view(torch.bfloat16)
            assert torch.isfinite(values).all()
            tables.append({'layer': layer.index, 'rows': table.rows, 'width': table.width,
                           'mapped_payload_bytes': table.nbytes, 'shards': len(table.values),
                           'boundary_rows_gathered': len(ids),
                           'gather_bf16_sha256': hashlib.sha256(rows.tobytes()).hexdigest()})
            prefix = f'model.language_model.layers.{layer.index}.ple.ple_embedding.ngram_embedding.'
            mapped.update(n for n in catalog.entries if n.startswith(prefix))
        report['ngram_tables'] = tables
        report['ngram_policy'] = 'read-only host mappings; all shard boundary rows tested; full table residency/prefetch not required'
        guard()
        # Preserve adjunct weights in their original representation, without draft requantization or vision execution.
        auxiliary = {}
        if a.auxiliary:
            for name, e in catalog.entries.items():
                if category(name) in ('mtp', 'vision', 'other'):
                    guard()
                    auxiliary[name] = rd.get(name)
            torch.cuda.synchronize()
        report['auxiliary_raw_cuda_tensors'] = len(auxiliary)
        report['auxiliary_raw_cuda_bytes'] = sum(t.numel() * t.element_size() for t in auxiliary.values())
        unread = set(catalog.entries) - rd.consumed - mapped
        allowed = {n for n in unread if n.endswith('.input_scale')}
        # Static activation scales are unused by the selected BF16-activation target path; retain them in one CUDA buffer.
        if a.layers == 48 and a.auxiliary:
            assert unread == allowed, sorted(unread - allowed)[:8]
            chunks = []
            for name in sorted(allowed):
                e = catalog.entries[name]
                assert e['dtype'] == 'F32' and e['bytes'] == 4
                with (catalog.root / e['shard']).open('rb') as f:
                    f.seek(e['offset'])
                    raw = f.read(4)
                assert len(raw) == 4
                chunks.append(raw)
            activation_scale_storage = torch.frombuffer(bytearray(b''.join(chunks)), dtype=torch.float32).cuda()
            assert torch.isfinite(activation_scale_storage).all() and (activation_scale_storage > 0).all()
            rd.consumed.update(allowed)
            unread -= allowed
            report['activation_scales_retained_but_not_applied'] = len(allowed)
        report.update(uncovered_tensor_count=len(unread),
                      covered_tensor_count=len(catalog.entries)-len(unread),
                      materialized_source_tensor_count=len(rd.consumed), mapped_source_tensor_count=len(mapped-rd.consumed),
                      fully_read_source_payload_bytes=sum(catalog.entries[n]['bytes'] for n in rd.consumed),
                      mapped_source_payload_bytes=sum(catalog.entries[n]['bytes'] for n in mapped-rd.consumed),
                      full_checkpoint_payload_hash_verified=False,
                      torch_allocated_bytes=torch.cuda.memory_allocated(), torch_reserved_bytes=torch.cuda.memory_reserved(),
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(), torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      base_global_torch_libraries_loaded='/usr/local/lib/python3.12/dist-packages/torch/lib/' in Path('/proc/self/maps').read_text())
        assert not report['base_global_torch_libraries_loaded']
        report['status'] = 'PASS_FULL_TARGET_WITH_RAW_ADJUNCTS' if a.layers == 48 and a.auxiliary else 'PASS_SELECTED_LOADING_SCOPE'
        report['weights_retained_until_verification_finished'] = True
        save()
        if a.forward_smoke:
            from e2_target_forward import probe
            report['target_forward_smoke'] = probe(w, root)
            report['full_model_forward'] = 'SELECTED_TARGET_ONLY_EXECUTION'
            report['torch_peak_allocated_including_forward_bytes'] = torch.cuda.max_memory_allocated()
            report['torch_peak_reserved_including_forward_bytes'] = torch.cuda.max_memory_reserved()
            guard()
            save()
        print(json.dumps({k:v for k,v in report.items() if k not in ('catalog','external_source_sha256','layers','harness_sha256')}), flush=True)
        del w, auxiliary
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    return 1 if report['status'] == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
