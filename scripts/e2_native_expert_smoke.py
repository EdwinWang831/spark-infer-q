#!/usr/bin/env python3
"""Exercise original TensorFold packed-weight/BF16-input expert kernels on SM121.

The routing plan is constructed explicitly for one expert; routing is not tested.
Only CPU references dequantize weights. GPU weights retain native packed blocks.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--expert-root', type=Path, required=True)
    a = p.parse_args()
    root, previous = a.root, a.expert_root
    report = {'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'original TensorFold native one-expert forward, BF16 activations, compressed GPU weights',
              'routing_tested': False, 'full_model_tested': False, 'performance_measured': False,
              'cases': [], 'packing_checks': {}}

    def save():
        (root / 'artifacts/native-smoke.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        import torch
        import torch.nn.functional as F
        assert torch.__version__ == '2.13.0+cu132' and torch.cuda.get_device_capability() == (12, 1)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(512 * 1024**2 / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        pin = json.loads((root / 'artifacts/source-manifest.json').read_text())
        for name, item in pin.items():
            assert hashlib.sha256((root / 'reference/tensorfold' / name).read_bytes()).hexdigest() == item['sha256']
        sys.path.insert(0, str(root / 'reference'))
        sys.path.insert(0, str(root / 'build'))
        import spark_q_native_expert_probe as extension
        from tensorfold.cuda import experts as grouped
        from tensorfold.cuda.nvfp4 import experts as native, linear
        # Reuse the original public wrappers, substituting only the built extension provider.
        linear._ext = lambda: extension
        binary = Path(extension.__file__)
        report.update(torch=torch.__version__, cuda_build=torch.version.cuda, capability=[12, 1],
                      native_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      source_files_unchanged=True, activation_dtype='torch.bfloat16',
                      extension_provider='thin binding to original experts.cu; linear._ext override only')
        spec = importlib.util.spec_from_file_location('tf_dequant_native_reference', previous / 'reference/tensorfold_nvfp4.py')
        dequant = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = dequant
        spec.loader.exec_module(dequant)
        manifest = json.loads((previous / 'expert/manifest.json').read_text())
        report['checkpoint_revision'] = manifest['model_revision']
        report['expert'] = manifest['expert']
        triples, refweights = {}, {}
        for name, item in manifest['projections'].items():
            n, k = item['shape']
            for filename, meta in item['artifacts'].items():
                data = (previous / 'expert' / filename).read_bytes()
                assert len(data) == meta['bytes'] and hashlib.sha256(data).hexdigest() == meta['sha256']
            codes = torch.frombuffer(bytearray((previous / 'expert' / (name + '-weight.bin')).read_bytes()), dtype=torch.uint8).reshape(n, k // 2)
            scales = torch.frombuffer(bytearray((previous / 'expert' / (name + '-scales.bin')).read_bytes()), dtype=torch.uint8).reshape(n, k // 16)
            triples[name] = (codes.unsqueeze(0), scales.unsqueeze(0), torch.tensor([item['weight_scale_2']]))
            refweights[name] = dequant.dequantize(codes, scales, item['weight_scale_2'])
        packed_cpu = native.make(triples['gate_proj'], triples['up_proj'], triples['down_proj'])
        for name, which in [('gate_proj', 'gate'), ('up_proj', 'up'), ('down_proj', 'down')]:
            actual, expected = native.dense(packed_cpu, 0, which), refweights[name]
            torch.testing.assert_close(actual, expected, atol=0, rtol=2e-7)
            report['packing_checks'][name] = {'status': 'PASS',
                'max_abs_difference': (actual - expected).abs().max().item(),
                'relative_l2_difference': ((actual - expected).norm() / expected.norm()).item()}
        allocated_before = torch.cuda.memory_allocated()
        packed = native.Experts4(packed_cpu.up.cuda(), packed_cpu.down.cuda(),
                                packed_cpu.up_scale.cuda(), packed_cpu.down_scale.cuda(),
                                packed_cpu.width, packed_cpu.dims, packed_cpu.limit)
        report['gpu_persistent_weight_bytes'] = packed.nbytes()
        report['gpu_weight_allocator_delta_bytes'] = torch.cuda.memory_allocated() - allocated_before
        report['cpu_only_fp32_reference_weight_bytes'] = sum(w.numel() * 4 for w in refweights.values())
        report['gpu_fp32_dequantized_weights_present'] = False
        report['numerical_gates'] = {'complete_relative_l2': .003, 'hidden_relative_l2': .002,
                                    'elementwise_atol': .005, 'elementwise_rtol': .01}

        def plan_for(rows):
            plan = grouped.Plan(rows, 1, 1, 'cuda')
            # Explicit one-expert plan, reversed members exercise nontrivial row mapping.
            plan.members.copy_(torch.arange(rows - 1, -1, -1, device='cuda', dtype=torch.int32))
            items = [[0, offset, min(16, rows - offset)] for offset in range(0, rows, 16)]
            plan.items[:len(items)].copy_(torch.tensor(items, device='cuda', dtype=torch.int32))
            plan.counts.copy_(torch.tensor([len(items), 1], device='cuda', dtype=torch.int32))
            return plan

        def reference(x, output_dtype):
            g = (x.float() @ refweights['gate_proj'].T).bfloat16()
            u = (x.float() @ refweights['up_proj'].T).bfloat16()
            h = (F.silu(g.float()).bfloat16().float() * u.float()).bfloat16()
            y = (h.float() @ refweights['down_proj'].T).to(output_dtype)
            return h, y

        def compare(y, expected, rel_limit):
            actual, expected = y.float().cpu(), expected.float().cpu()
            assert torch.isfinite(actual).all()
            delta = actual - expected
            stat = {'relative_l2_error': (delta.norm() / expected.norm().clamp_min(1e-12)).item(),
                    'max_abs_error': delta.abs().max().item(), 'rmse': delta.square().mean().sqrt().item()}
            assert stat['relative_l2_error'] <= rel_limit, stat
            torch.testing.assert_close(actual, expected, atol=.005, rtol=.01)
            return stat

        def run(name, fn):
            case = {'name': name}
            try:
                case.update(fn())
                torch.cuda.synchronize()
                case['status'] = 'PASS'
            except AssertionError as exc:
                case.update(status='NUMERIC_OR_CONTRACT_FAIL', error=str(exc))
                traceback.print_exc()
            except Exception as exc:
                case.update(status='KERNEL_FAIL', error=f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
            report['cases'].append(case)
            save()
            print(json.dumps(case), flush=True)

        def expert_case(rows, seed, dtype, strided=False, zero=False):
            torch.manual_seed(seed)
            cpu = torch.zeros(rows, 2560).bfloat16() if zero else torch.randn(rows, 2560).bfloat16()
            if strided:
                storage = torch.zeros(rows, 2576, dtype=torch.bfloat16, device='cuda')
                x = storage[:, :2560]
                x.copy_(cpu)
            else:
                x = cpu.cuda()
            plan = plan_for(rows)
            hb = torch.full((rows + 2, 640), 13., device='cuda', dtype=torch.bfloat16)
            yb = torch.full((rows + 2, 2560), 13., device='cuda', dtype=dtype)
            h, y = hb[1:-1], yb[1:-1]
            native.gate_up(x, packed, plan, h, rows)
            native.down(h, packed, plan, y, rows)
            rh, ry = reference(cpu, dtype)
            hm, ym = compare(h, rh, .002), compare(y, ry, .003)
            # Check down in isolation using the actual hidden tensor.
            isolated = (h.cpu().float() @ refweights['down_proj'].T).to(dtype)
            local = compare(y, isolated, .002 if dtype == torch.bfloat16 else 1e-4)
            assert torch.all(hb[[0, -1]] == 13.) and torch.all(yb[[0, -1]] == 13.)
            return {'batch': rows, 'seed': seed, 'input_row_stride': x.stride(0), 'output_dtype': str(dtype),
                    'hidden_error': hm, 'complete_error': ym, 'down_error_actual_hidden': local,
                    'output_guards_unchanged': True}

        for rows, seed in [(1, 20261020), (8, 20261021), (17, 20261023), (32, 20261022)]:
            for dtype in [torch.bfloat16, torch.float32]:
                run(f'complete_b{rows}_{dtype}', lambda r=rows, s=seed, d=dtype: expert_case(r, s, d, r == 17))
        run('zero_input', lambda: expert_case(1, 7, torch.float32, zero=True))

        def skip_case():
            x = torch.randn(17, 2560, device='cuda').bfloat16()
            plan = plan_for(17)
            h = torch.full((17, 640), 9., device='cuda', dtype=torch.bfloat16)
            y = torch.full((17, 2560), 9., device='cuda', dtype=torch.float32)
            native.gate_up(x, packed, plan, h, 17, skip=0)
            native.down(h, packed, plan, y, 17, skip=0)
            assert torch.all(h == 9.) and torch.all(y == 9.)
            return {'skipped_expert_outputs_unchanged': True}
        run('skip_expert', skip_case)

        def graph_case():
            rows = 8
            x = torch.randn(rows, 2560, device='cuda').bfloat16()
            plan = plan_for(rows)
            h = torch.empty(rows, 640, device='cuda', dtype=torch.bfloat16)
            y = torch.empty(rows, 2560, device='cuda', dtype=torch.float32)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                native.gate_up(x, packed, plan, h, rows)
                native.down(h, packed, plan, y, rows)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                native.gate_up(x, packed, plan, h, rows)
                native.down(h, packed, plan, y, rows)
            x.copy_(torch.randn_like(x))
            graph.replay()
            torch.cuda.synchronize()
            rh, ry = reference(x.cpu(), torch.float32)
            return {'changed_input_replay': True, 'hidden_error': compare(h, rh, .002),
                    'complete_error': compare(y, ry, .003)}
        run('cuda_graph_changed_input_replay', graph_case)
        report.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      base_global_torch_libraries_loaded='/usr/local/lib/python3.12/dist-packages/torch/lib/' in Path('/proc/self/maps').read_text())
        assert not report['base_global_torch_libraries_loaded']
        report['status'] = 'PASS' if all(c['status'] == 'PASS' for c in report['cases']) else 'PARTIAL_OR_FAIL'
    except Exception as exc:
        report.update(status='IMPORT_OR_SETUP_FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save()
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
