#!/usr/bin/env python3
"""Real-block request lifecycle, independent GPU reference, and local timings.

No complete-model logits, answer-quality or model-throughput result is produced.
"""
import argparse
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--layer', type=int, choices=[0, 3])
    args = parser.parse_args()
    root = args.root
    report = {'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'bounded target-only executor on independent real layers 0/GDN and 3/QSA',
              'quality_evaluation': 'NOT_MEASURED', 'full_model_performance': 'NOT_MEASURED',
              'mtp_enabled': False, 'layers': [],
              'reference_policy': 'same represented weights/BF16 boundaries with independent PyTorch FP32 GPU arithmetic; differences reported separately from lifecycle checks'}
    report['harness_sha256'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                               ['e2_executor_smoke.py', 'e2_block_reference.py', 'e2_block_support.py', 'e2_run_executor.sh',
                                'spark_infer_q/__init__.py', 'spark_infer_q/executor.py', 'spark_infer_q/tensorfold_block.py']}

    def save():
        (root / 'artifacts/executor-smoke.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        assert os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '0'
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        assert torch.__version__ == '2.13.0+cu132' and torch.cuda.get_device_capability() == (12, 1)
        torch.cuda.set_per_process_memory_fraction(3 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        block_root, moe_root = root.parent / 'e2-block', root.parent / 'e2-moe'
        pin = json.loads((block_root / 'artifacts/source-manifest.json').read_text())
        for name, meta in pin.items():
            assert hashlib.sha256((block_root / 'reference/tensorfold' / name).read_bytes()).hexdigest() == meta['sha256']
        sys.path.insert(0, str(block_root / 'reference'))
        for name in ['spark_q_moe_linear', 'spark_q_moe_plan', 'spark_q_moe_prompt']:
            sys.path.insert(0, str(moe_root / 'build' / name))
        for name in ['spark_q_block_gdn', 'spark_q_block_gdn_io', 'spark_q_block_gdn_shared']:
            sys.path.insert(0, str(block_root / 'build' / name))
        import spark_q_moe_linear, spark_q_moe_plan, spark_q_moe_prompt
        import spark_q_block_gdn, spark_q_block_gdn_io, spark_q_block_gdn_shared
        report['extensions_sha256'] = {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in
                                     [spark_q_moe_linear, spark_q_moe_plan, spark_q_moe_prompt,
                                      spark_q_block_gdn, spark_q_block_gdn_io, spark_q_block_gdn_shared]}
        report['external_source_files_sha256'] = {name: meta['sha256'] for name, meta in pin.items()}
        from tensorfold.cuda import experts
        from tensorfold.cuda.nvfp4 import linear
        from tensorfold.cuda.kernels import gdn as shared_gdn
        from tensorfold.families.qwen4_exp.cuda import forward, gdn, gdn_io
        from tensorfold.families.qwen4_exp.cuda.state import Buffers, State
        from tensorfold.families.qwen4_exp.cuda.weight_types import Config, Weights
        from e2_block_support import BlockWeights, MoEWeights, make_layer
        from e2_block_reference import Reference
        from spark_infer_q.executor import Executor, Row, Busy, StaleHandle, WorkClosed
        from spark_infer_q.tensorfold_block import TensorFoldBlockBackend
        linear._ext = lambda: spark_q_moe_linear
        linear._prompt_ext = lambda: spark_q_moe_prompt
        experts._ext = lambda: spark_q_moe_plan
        gdn._ext = lambda: spark_q_block_gdn
        gdn_io._ext = lambda: spark_q_block_gdn_io
        shared_gdn._ext = lambda: spark_q_block_gdn_shared
        rd = BlockWeights(block_root / 'nonexpert')
        cfg = Config.read(block_root / 'nonexpert')
        means = [float(rd.get(f'model.language_model.layers.{i}.attn_hyper_connection.hc_norm.weight').mean()) for i in (0, 3)]
        assert all(-.5 <= m <= .25 for m in means)
        report.update(torch=torch.__version__, cuda_build=torch.version.cuda, capability=[12, 1],
                      checkpoint_revision=rd.manifest['checkpoint_revision'],
                      tf32_override=os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'],
                      torch_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      float32_matmul_precision=torch.get_float32_matmul_precision())

        def error(a, b, limit=None):
            a, b = a.detach().float().cpu(), b.detach().float().cpu()
            assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
            d = a - b
            stat = {'relative_l2': float(d.norm() / b.norm().clamp_min(1e-12)),
                    'max_abs': float(d.abs().max()) if d.numel() else 0., 'bit_exact': torch.equal(a, b)}
            if limit is not None:
                assert stat['relative_l2'] <= limit, stat
            return stat

        for index in ([args.layer] if args.layer is not None else [0, 3]):
            item = {'index': index, 'cases': [], 'hidden_input_seed': 20261040 + index,
                    'hidden_shape': [40, 10240], 'hidden_dtype': 'bfloat16', 'kv_dtype': 'bfloat16'}
            report['layers'].append(item)
            torch.cuda.reset_peak_memory_stats()
            mr = MoEWeights(moe_root / 'layer' if index == 0 else block_root / 'moe3')
            layer, cpu = make_layer(rd, mr, index, False)
            inv = torch.pow(torch.tensor(cfg.rope_theta, dtype=torch.float64),
                            -torch.arange(32, dtype=torch.float64) * 2 / 64).float().cuda()
            w = Weights(cfg, (), [layer], None, SimpleNamespace(n=1), inv, mtp=None)
            torch.manual_seed(20261040 + index)
            h = torch.randn(40, 10240).bfloat16().cuda()

            def backend(**kwargs):
                return TensorFoldBlockBackend(w, layer, **kwargs)

            def step(ex, handle, x, mode='decode'):
                pos = ex.view(handle)['position']
                ticket = ex.prepare([Row(handle, pos + r) for r in range(x.shape[0])], mode)
                ex.dispatch(ticket, x)
                out = ex.finish(ticket)
                ex.backend.synchronize()
                return out.output

            def direct(x, st, prefill=False):
                n = x.shape[0]
                st.ensure(st.pos + n, step=8)
                buf = Buffers(w, n, st.capacity, prefill=prefill)
                buf.h.copy_(x)
                pending = forward.layer_forward(layer, w, [(st, 0, n)], buf, n, None, context=st.capacity)
                forward._writeback(buf.h, buf, n, cfg, pending)
                out = buf.h.clone()
                forward.commit(w, st, buf, n, n)
                return out, buf

            def state_equal(a, b):
                assert a['position'] == b['position']
                keys = [k for k in a if isinstance(a[k], torch.Tensor) and a[k].numel()]
                return {k: error(a[k], b[k], 0) for k in keys}

            def case(name, fn):
                entry = {'name': name}
                try:
                    entry.update(fn())
                    torch.cuda.synchronize()
                    entry['status'] = 'PASS'
                except Exception as exc:
                    entry.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
                    traceback.print_exc()
                item['cases'].append(entry)
                save()
                print(json.dumps({'layer': index, **entry}), flush=True)

            def reference_case():
                b = backend(graphs=False)
                ex = Executor(b)
                req = ex.admit('reference')
                actual = step(ex, req, h[:17], 'prefill')
                # Independent tensor-level math, with one selected expert decoded at a time.
                ref = Reference(cpu, mr, cfg, index, False, device='cuda')
                expected = ref.block(h[:17], True)
                state = ex.debug_state(req)
                metric = error(actual, expected)
                native, buf = direct(h[:17], State(w, 32, 32, limit=64), True)
                error(actual, native, 0)
                ap, rp = buf.moe.pick[:17, :10].long(), ref.last['picks']
                router = layer.moe.router.float()
                same_input_logits = buf.mixed[:17].float() @ router.T
                same_input_picks = torch.argsort(same_input_logits[:, :512], descending=True, stable=True)[:, :10]
                same_logits_picks = torch.argsort(buf.moe.logits[:17, :512], descending=True, stable=True)[:, :10]
                assert torch.equal(ap, same_logits_picks)
                different = (ap.sort(1).values != rp.sort(1).values).any(1)
                ref_logits = ref.last['moe_in'].float() @ router.T
                margins = []
                for row in torch.where(different)[0].tolist():
                    vals = torch.sort(ref_logits[row, :512], descending=True).values
                    margins.append({'row': row, 'reference_top10': rp[row].tolist(), 'native_top10': ap[row].tolist(),
                                    'reference_10th_minus_11th_logit': float(vals[9] - vals[10]),
                                    'max_logit_change_from_upstream_input': float((same_input_logits[row] - ref_logits[row]).abs().max())})
                out = {'rows': 17, 'output_difference': metric,
                       'review_threshold_relative_l2': .01, 'requires_numerical_review': metric['relative_l2'] > .01,
                       'answer_quality_conclusion': 'NOT_MEASURED', 'reference_device': str(ref.device),
                       'executor_matches_direct_native_output_exactly': True,
                       'attention_branch_difference': error(buf.branch[:17], ref.last['attention_branch']),
                       'moe_input_difference': error(buf.mixed[:17], ref.last['moe_in']),
                       'route_rows_with_set_difference': int(different.sum()),
                       'route_pick_slots_different': int((ap != rp).sum()),
                       'topk_given_same_native_logits_exact': True,
                       'pytorch_router_given_native_input_sets_equal': torch.equal(ap.sort(1).values, same_input_picks.sort(1).values),
                       'router_difference_given_native_input': error(buf.moe.logits[:17], same_input_logits),
                       'routing_boundary_diagnostics': margins}
                if index == 0:
                    out['state_difference'] = error(state['rec'], ref.rec)
                    out['conv_difference'] = error(state['conv'][0], ref.conv)
                else:
                    out['key_difference'] = error(state['k'], ref.k)
                    out['value_difference'] = error(state['v'], ref.v)
                del ref
                ex.cancel(req)
                return out
            case('independent_pytorch_gpu_reference', reference_case)
            gc.collect()
            torch.cuda.empty_cache()

            def mapping_case():
                b, ex = backend(), None
                ex = Executor(b)
                a, c = ex.admit('A'), ex.admit('B')
                sa, sb = State(w, 8, 32, limit=64), State(w, 8, 32, limit=64)
                first = [(a, 0, 0), (c, 0, 7), (a, 1, 1), (None, None, None),
                         (c, 1, 8), (a, 2, 2), (c, 2, 9), (c, 3, 10), (c, 4, 11), (None, None, None)]
                values = torch.full((len(first), 10240), float('nan'), device='cuda', dtype=torch.bfloat16)
                for r, (_, _, src) in enumerate(first):
                    if src is not None:
                        values[r] = h[src]
                ticket = ex.prepare([Row(req, p) for req, p, _ in first], 'prefill')
                ex.dispatch(ticket, values)
                got = ex.finish(ticket)
                b.synchronize()
                da, _ = direct(h[:3], sa, True)
                db, _ = direct(h[7:12], sb, True)
                for r, (req, p, _) in enumerate(first):
                    if req is None:
                        assert torch.all(got.output[r] == 0) and got.row_status[r] == 'PADDING'
                    else:
                        error(got.output[r], (da if req == a else db)[p], 0)
                second = [(c, 5, 12), (None, None, None), (a, 3, 3), (c, 6, 13), (a, 4, 4)]
                values = torch.full((5, 10240), float('nan'), device='cuda', dtype=torch.bfloat16)
                for r, (_, _, src) in enumerate(second):
                    if src is not None:
                        values[r] = h[src]
                ticket = ex.prepare([Row(req, p) for req, p, _ in second])
                ex.dispatch(ticket, values)
                got = ex.finish(ticket)
                b.synchronize()
                da, _ = direct(h[3:5], sa)
                db, _ = direct(h[12:14], sb)
                for r, (req, p, _) in enumerate(second):
                    if req is None:
                        assert torch.all(got.output[r] == 0)
                    else:
                        error(got.output[r], (da if req == a else db)[p-(3 if req == a else 5)], 0)
                state_equal(ex.debug_state(a), b.export_state(sa))
                state_equal(ex.debug_state(c), b.export_state(sb))
                result = {'interleaved_positions_and_outputs_exact': True, 'nan_padding_zero_and_unexecuted': True,
                          'committed_positions': [ex.view(a)['position'], ex.view(c)['position']], 'graph_stats': dict(b.stats)}
                ex.cancel(a); ex.cancel(c)
                return result
            case('interleaved_requests_positions_nan_padding', mapping_case)

            def cancel_case():
                b = backend()
                ex = Executor(b, max_requests=1)
                old = ex.admit('reuse')
                step(ex, old, h[:5], 'prefill')
                snap = ex.checkpoint(old)
                ticket = ex.prepare([Row(old, 5)])
                ex.dispatch(ticket, h[5:6])
                assert ex.view(old)['position'] == 5
                ex.cancel(old)
                try:
                    ex.admit('reuse')
                except Busy:
                    pass
                else:
                    raise AssertionError('in-flight state reused')
                got = ex.finish(ticket)
                b.synchronize()
                assert got.row_status == ('CANCELED',) and torch.all(got.output == 0)
                new = ex.admit('reuse')
                assert old.slot == new.slot and new.generation > old.generation
                try:
                    ex.finish(ticket)
                except WorkClosed:
                    pass
                else:
                    raise AssertionError('late completion accepted')
                try:
                    ex.restore(new, snap)
                except StaleHandle:
                    pass
                else:
                    raise AssertionError('old snapshot accepted by replacement')
                actual = step(ex, new, h[8:9], 'prefill')
                expected, _ = direct(h[8:9], State(w, 8, 32, limit=64), True)
                error(actual, expected, 0)
                result = {'canceled_output_suppressed': True, 'slot_quarantined_until_finish': True,
                          'generation_changed': True, 'late_completion_and_checkpoint_rejected': True,
                          'fresh_request_output_exact': True, 'graph_stats': dict(b.stats)}
                ex.cancel(new)
                return result
            case('inflight_cancel_drain_generation_reuse', cancel_case)

            def checkpoint_case():
                b = backend()
                ex = Executor(b)
                req = ex.admit('snapshot')
                empty = ex.checkpoint(req)
                step(ex, req, h[:5], 'prefill')
                kept = ex.checkpoint(req)
                original = step(ex, req, h[5:8])
                target_state = ex.debug_state(req)
                ex.restore(req, empty)
                step(ex, req, h[8:13], 'prefill')  # overwrite the old prefix, including K/V rows
                ex.restore(req, kept)
                replay = step(ex, req, h[5:8])
                error(replay, original, 0)
                state_equal(ex.debug_state(req), target_state)
                result = {'prefix_overwritten_before_restore': True, 'checkpoint_copies_kv_prefix': True,
                          'replayed_output_and_state_exact': True, 'graph_stats': dict(b.stats)}
                ex.drop_checkpoint(empty); ex.drop_checkpoint(kept); ex.cancel(req)
                return result
            case('full_checkpoint_restores_overwritten_prefix', checkpoint_case)

            def growth_case():
                b = backend()
                ex = Executor(b)
                req = ex.admit('growth')
                baseline = State(w, 16, 32, limit=64)
                step(ex, req, h[:5], 'prefill'); direct(h[:5], baseline, True)
                step(ex, req, h[5:8]); direct(h[5:8], baseline)
                old_version = ex.view(req)['layout_version']
                actual = step(ex, req, h[8:10])
                expected, _ = direct(h[8:10], baseline)
                assert ex.view(req)['layout_version'] > old_version and b.stats['graph_invalidations'] > 0
                error(actual, expected, 0)
                state_equal(ex.debug_state(req), b.export_state(baseline))
                result = {'layout_version_changed': True, 'stale_graph_bindings_invalidated': True,
                          'output_and_state_exact': True, 'graph_stats': dict(b.stats)}
                ex.cancel(req)
                return result
            case('capacity_growth_invalidates_graph_bindings', growth_case)

            def dynamic_graph_case():
                b = backend(capacity=32)
                ex = Executor(b)
                req = ex.admit('decode')
                baseline = State(w, 32, 32, limit=64)
                step(ex, req, h[:3], 'prefill'); direct(h[:3], baseline, True)
                for r in range(3, 9):
                    actual = step(ex, req, h[r:r+1])
                    expected, _ = direct(h[r:r+1], baseline)
                    error(actual, expected, 0)
                state_equal(ex.debug_state(req), b.export_state(baseline))
                assert b.stats['graph_hits'] >= 4
                result = {'decode_steps': 6, 'committed_position': ex.view(req)['position'],
                          'graph_reuse_with_position_updates': True, 'commit_outside_graph': True,
                          'outputs_and_state_exact': True, 'graph_stats': dict(b.stats)}
                ex.cancel(req)
                return result
            case('graph_reuse_across_committed_decode_steps', dynamic_graph_case)

            def prepared_case():
                b = backend()
                ex = Executor(b)
                req = ex.admit('prepared')
                ticket = ex.prepare([Row(req, 0), Row()])
                ex.cancel(req)
                try:
                    ex.dispatch(ticket, h[:2])
                except WorkClosed:
                    pass
                else:
                    raise AssertionError('canceled plan executed')
                ticket = ex.prepare([Row(), Row()])
                ex.dispatch(ticket, torch.full_like(h[:2], float('nan')))
                got = ex.finish(ticket)
                b.synchronize()
                assert got.row_status == ('PADDING', 'PADDING') and torch.all(got.output == 0)
                assert b.stats['model_calls'] == 0
                return {'canceled_prepared_work_rejected': True, 'all_padding_model_calls': 0}
            case('prepared_cancel_and_padding_only_batch', prepared_case)

            def stream_case():
                b = backend(graphs=False)
                ex = Executor(b)
                req = ex.admit('streams')
                producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
                producer.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(producer):
                    inputs = h[:2].clone()
                    ticket = ex.prepare([Row(req, 0), Row(req, 1)])
                    ex.dispatch(ticket, inputs)
                    del inputs
                with torch.cuda.stream(consumer):
                    got = ex.finish(ticket)
                    copied = got.output.clone()  # must wait for backend's asynchronous output mask
                consumer.synchronize()  # deliberately do not call backend.synchronize first
                expected, _ = direct(h[:2], State(w, 8, 32, limit=64))
                error(copied, expected, 0)
                ex.cancel(req)
                return {'producer_backend_consumer_streams_distinct': True,
                        'input_released_after_dispatch': True, 'consumer_output_exact_without_backend_host_sync': True}
            case('cross_stream_input_lifetime_and_output_dependency', stream_case)

            def rollback_case():
                class Fault(TensorFoldBlockBackend):
                    fail = False
                    def launch(self, *a):
                        result = super().launch(*a)
                        if self.fail:
                            self.fail = False
                            raise RuntimeError('injected host error after CUDA launch')
                        return result
                b = Fault(w, layer)
                ex = Executor(b)
                req = ex.admit('rollback')
                step(ex, req, h[:5], 'prefill')
                before = ex.debug_state(req)
                b.fail = True
                ticket = ex.prepare([Row(req, 5), Row(req, 6)])
                try:
                    ex.dispatch(ticket, h[5:7])
                except RuntimeError as exc:
                    assert 'injected host error' in str(exc)
                else:
                    raise AssertionError('fault did not occur')
                state_equal(ex.debug_state(req), before)
                assert ex.view(req)['position'] == 5
                actual = step(ex, req, h[5:7])
                baseline = State(w, 8, 32, limit=64)
                direct(h[:5], baseline, True)
                expected, _ = direct(h[5:7], baseline)
                error(actual, expected, 0)
                ex.cancel(req)
                return {'fault_kind': 'host exception after successful CUDA dispatch',
                        'prefix_state_restored_exactly': True, 'next_execution_exact': True,
                        'cuda_device_fault_recovery_tested': False}
            case('transaction_rollback_after_dispatched_host_failure', rollback_case)

            def timings_case():
                pairs = []
                for enabled in (False, True):
                    b = backend(graphs=enabled, capacity=64)
                    ex = Executor(b)
                    req = ex.admit('timing')
                    step(ex, req, h[:4], 'prefill')
                    for r in range(4, 8):
                        step(ex, req, h[r:r+1])
                    pairs.append((b, ex, req))
                samples = [[], []]
                for r in range(8, 32):
                    results = {}
                    for which in ((0, 1) if r % 2 == 0 else (1, 0)):
                        b, ex, req = pairs[which]
                        start = time.perf_counter()
                        results[which] = step(ex, req, h[r:r+1])
                        samples[which].append((time.perf_counter() - start) * 1000)
                    error(results[0], results[1], 0)
                stats = []
                for values in samples:
                    stats.append({'samples': len(values), 'median_ms': statistics.median(values),
                                  'p95_ms': sorted(values)[math.ceil(.95 * len(values))-1]})
                for b, ex, req in pairs:
                    ex.cancel(req)
                return {'scope': 'one real block; serialized request bookkeeping, full-state backup, forward/commit/masking and completion wait',
                        'alternating_order': True, 'warmup_decode_steps': 4,
                        'eager': stats[0], 'graph': stats[1],
                        'median_eager_over_graph_ratio': stats[0]['median_ms'] / stats[1]['median_ms'],
                        'outputs_bit_exact': True, 'full_model_speedup_claimed': False}
            case('local_executor_graph_vs_eager_wall_time', timings_case)

            item.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            item['status'] = 'PASS' if all(c['status'] == 'PASS' for c in item['cases']) else 'PARTIAL_OR_FAIL'
            save()
            del w, layer, cpu, mr, inv, h
            gc.collect()
            torch.cuda.empty_cache()
        report['status'] = 'PASS' if all(l['status'] == 'PASS' for l in report['layers']) else 'PARTIAL_OR_FAIL'
        report['base_global_torch_libraries_loaded'] = '/usr/local/lib/python3.12/dist-packages/torch/lib/' in Path('/proc/self/maps').read_text()
        assert not report['base_global_torch_libraries_loaded']
    except Exception as exc:
        report.update(status='SETUP_OR_RUNTIME_FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    report.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save()
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
