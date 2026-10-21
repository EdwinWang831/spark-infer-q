#!/usr/bin/env python3
"""Small real-GPU numerical probes; run in the separate E1 container.

No weights, inference requests, timings or speedup claims. Each operation is
checked against FP32 CPU arithmetic. An outer process timeout is required.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--component', choices=['tilelang', 'deepgemm'], required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    sha = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != '774ce244b87cc0eb8d4b94ab9f5de2ab22049f01':
        raise RuntimeError('Unexpected upstream revision')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', '--ignore-submodules'], check=True)
    sys.path.insert(0, str(source))
    report = {'upstream_sha': sha, 'component': args.component,
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'weights_loaded': False, 'performance_measured': False, 'cases': []}

    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        import torch
        report['torch'] = torch.__version__
        report['cuda_build'] = torch.version.cuda
        assert torch.__version__ == '2.13.0+cu132'
        assert torch.cuda.get_device_capability() == (12, 1)
        report['device'] = torch.cuda.get_device_name()
        report['capability'] = list(torch.cuda.get_device_capability())
        total = torch.cuda.get_device_properties(0).total_memory
        # Allocator limit only: CUDA context and external libraries are additional.
        torch.cuda.set_per_process_memory_fraction(min(0.02, 512 * 1024**2 / total))
        report['torch_allocator_limit_bytes'] = min(int(total * .02), 512 * 1024**2)
        torch.set_num_threads(2)
        torch.manual_seed(20261020)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.cuda.reset_peak_memory_stats()

        def compare(actual, expected, atol, rtol):
            actual = actual.detach().float().cpu()
            expected = expected.float()
            assert torch.isfinite(actual).all()
            delta = actual - expected
            metrics = {'max_abs_error': delta.abs().max().item(),
                       'rmse': delta.square().mean().sqrt().item(),
                       'relative_l2_error': (delta.norm() / expected.norm().clamp_min(1e-12)).item(),
                       'atol': atol, 'rtol': rtol}
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            return metrics

        def run(name, fn):
            case = {'name': name, 'status': 'KERNEL_FAIL'}
            try:
                case.update(fn())
                torch.cuda.synchronize()
                case['status'] = 'PASS'
            except AssertionError as exc:
                case.update(status='NUMERIC_FAIL', error=str(exc))
                traceback.print_exc()
            except Exception as exc:
                case['error'] = f'{type(exc).__name__}: {exc}'
                traceback.print_exc()
            report['cases'].append(case)
            save()
            print(json.dumps(case), flush=True)

        def matmul_control():
            a = torch.randn(32, 128).bfloat16()
            b = torch.randn(64, 128).bfloat16()
            y = a.cuda() @ b.cuda().T
            return compare(y, a.float() @ b.float().T, .125, .01)

        run('torch_bf16_matmul_control_32x64x128', matmul_control)
        if args.component == 'tilelang':
            import tilelang
            from deepinfer.model_executor.layers.tilelang_ops.norm import norm
            import deepinfer._C
            report.update(tilelang=tilelang.__version__,
                          tile_kernels=importlib.metadata.version('tile-kernels'),
                          core_extension_import=True)
            for dtype in [torch.float32, torch.bfloat16]:
                for shape in [(1, 128), (17, 2560), (2, 17, 2560)]:
                    for residual in [False, True]:
                        def norm_case(dtype=dtype, shape=shape, residual=residual):
                            x = torch.randn(shape).to(dtype)
                            w = torch.randn(shape[-1]).to(dtype)
                            r = torch.randn(shape).to(dtype) if residual else None
                            z = (x + r).to(dtype) if residual else x
                            f = z.float()
                            ref = (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6) * w.float()).to(dtype)
                            y = norm(x.cuda(), w.cuda(), residual=r.cuda() if residual else None)
                            if residual:
                                y, rout = y
                                torch.testing.assert_close(rout.cpu(), z, atol=0, rtol=0)
                            return compare(y, ref, .03125 if dtype == torch.bfloat16 else 2e-5,
                                           .008 if dtype == torch.bfloat16 else 2e-5)
                        run(f'norm_{str(dtype)}_{"x".join(map(str, shape))}_residual{residual}', norm_case)

            def graph_case():
                x = torch.randn(17, 2560, device='cuda', dtype=torch.bfloat16)
                w = torch.randn(2560, device='cuda', dtype=torch.bfloat16)
                y = torch.empty_like(x)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    norm(x, w, out=y)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    norm(x, w, out=y)
                # Change input after capture: validates replay rather than stale output.
                x.copy_(torch.randn_like(x))
                graph.replay()
                torch.cuda.synchronize()
                f = x.cpu().float()
                ref = (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6) * w.cpu().float()).bfloat16()
                return compare(y, ref, .03125, .008)
            run('norm_cuda_graph_capture_changed_input_replay', graph_case)
        else:
            import deep_gemm
            report['deepgemm_import'] = True
            report['deepgemm_version'] = getattr(deep_gemm, '__version__', None)
            def gemm_case(kind):
                a = torch.randn(32, 128).bfloat16()
                b = torch.randn(64, 128).bfloat16()
                d = torch.empty(32, 64, device='cuda', dtype=torch.bfloat16)
                if kind == 'cublaslt':
                    deep_gemm.cublaslt_gemm_nt(a.cuda(), b.cuda(), d)
                else:
                    # Disable opportunistic cuBLASLt so this probes native dispatch.
                    deep_gemm.use_deterministic_algorithms(True)
                    deep_gemm.bf16_gemm_nt(a.cuda(), b.cuda(), d)
                return compare(d, a.float() @ b.float().T, .125, .01)
            run('deepgemm_cublaslt_bf16', lambda: gemm_case('cublaslt'))
            run('deepgemm_native_bf16', lambda: gemm_case('native'))
        report['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['torch_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        maps = Path('/proc/self/maps').read_text()
        report['base_global_torch_libraries_loaded'] = any(
            '/usr/local/lib/python3.12/dist-packages/torch/lib/' in line for line in maps.splitlines())
        assert not report['base_global_torch_libraries_loaded']
        report['status'] = 'PASS' if all(c['status'] == 'PASS' for c in report['cases']) else 'PARTIAL_OR_FAIL'
    except Exception as exc:
        report['status'] = 'IMPORT_OR_SETUP_FAIL'
        report['error'] = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save()
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
