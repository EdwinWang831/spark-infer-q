#!/usr/bin/env python3
"""One complete real expert; no router, full model, generation or timing claim.

CPU references retain stored quantized weights as FP32 and round linear outputs
and SwiGLU boundaries to BF16. CUDA activation quantization calls the external
TensorFold nvfp4q.cuh through a small probe wrapper. Activation quantization
error is reported separately from arithmetic agreement with that representation.
"""
import argparse
import ctypes
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root
    report_path = root / 'artifacts/expert-smoke.json'
    report = {'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'one layer-0 expert, synthetic BF16 inputs, full dimensions',
              'performance_measured': False, 'cases': [], 'quantizer_cases': []}

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        import torch
        import torch.nn.functional as F
        import deep_gemm
        assert torch.__version__ == '2.13.0+cu132'
        assert torch.cuda.get_device_capability() == (12, 1)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.cuda.set_per_process_memory_fraction(512 * 1024**2 / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        report.update(torch=torch.__version__, cuda_build=torch.version.cuda,
                      deep_gemm=deep_gemm.__version__, capability=[12, 1], allocator_limit_bytes=512 * 1024**2)
        manifest = json.loads((root / 'expert/manifest.json').read_text())
        assert manifest['status'] == 'PASS'
        report['extraction'] = manifest
        spec = importlib.util.spec_from_file_location('tf_nvfp4_reference', root / 'reference/tensorfold_nvfp4.py')
        tf = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = tf
        spec.loader.exec_module(tf)
        quantlib = ctypes.CDLL(str((root / 'artifacts/libquantize_reference.so').resolve()))
        quantlib.quantize_bf16.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
        quantlib.quantize_bf16.restype = ctypes.c_int
        lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
        f32 = lambda v: struct.unpack('<f', struct.pack('<f', v))[0]

        def decode(data, sf, global_scale=1.):
            code = torch.stack((data & 15, data >> 4), -1).flatten(-2).long()
            values = lut[code & 7] * torch.where((code & 8) != 0, -1., 1.)
            scales = sf.contiguous().view(torch.float8_e4m3fn).float() * global_scale
            assert torch.isfinite(scales).all() and (scales >= 0).all()
            return values * scales.repeat_interleave(16, -1)

        def sf_layout(sf):
            m, c = sf.shape
            pad = torch.zeros(((m + 127) // 128 * 128, (c + 3) // 4 * 4), dtype=torch.uint8, device=sf.device)
            pad[:m, :c] = sf
            return pad.view(pad.shape[0] // 128, 4, 32, pad.shape[1] // 4, 4).permute(0, 3, 2, 1, 4).contiguous()

        weights = {}
        report['weight_dequantization_checks'] = {}
        for name, item in manifest['projections'].items():
            n, k = item['shape']
            for filename, metadata in item['artifacts'].items():
                raw = (root / 'expert' / filename).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == metadata['sha256'] and len(raw) == metadata['bytes']
            packed = torch.frombuffer(bytearray((root / 'expert' / (name + '-weight.bin')).read_bytes()), dtype=torch.uint8).reshape(n, k // 2)
            sf = torch.frombuffer(bytearray((root / 'expert' / (name + '-scales.bin')).read_bytes()), dtype=torch.uint8).reshape(n, k // 16)
            w = decode(packed, sf, item['weight_scale_2'])
            reference = tf.dequantize(packed, sf, item['weight_scale_2'])
            mismatches = int((w != reference).sum())
            assert mismatches == 0
            report['weight_dequantization_checks'][name] = {'mismatches': mismatches, 'elements': w.numel(),
                                                          'fp32_sha256': hashlib.sha256(w.numpy().tobytes()).hexdigest()}
            weights[name] = dict(item, cpu=w, packed=packed.cuda().view(torch.int8), sf=sf_layout(sf.cuda()))

        def quant_cpu(x, scale):
            g = f32(1. / scale)
            groups = x.float().reshape(x.shape[0], -1, 16)
            raw = (groups.abs().amax(-1) * f32(1. / 6.)) * g
            sf = raw.clamp(0, 448).to(torch.float8_e4m3fn).float()
            mult = torch.where(sf != 0, g / sf, 0.)
            z = groups * mult.unsqueeze(-1)
            av = z.abs()
            code = torch.full_like(av, 7, dtype=torch.uint8)
            for bound, inclusive, value in [(5., True, 6), (3.5, False, 5), (2.5, True, 4),
                                             (1.75, False, 3), (1.25, True, 2), (.75, False, 1), (.25, True, 0)]:
                code = torch.where(av <= bound if inclusive else av < bound, value, code)
            code = code | (((z < 0) & (code != 0)).to(torch.uint8) * 8)
            code = code.flatten(1)
            packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
            sf_bytes = sf.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
            return packed, sf_bytes, {'scale_saturated_blocks': int((raw > 448).sum()),
                                      'zero_scale_blocks': int((sf == 0).sum())}

        def quant_gpu_checked(x, scale, label):
            m, k = x.shape
            out = torch.empty((m, k // 2), dtype=torch.uint8, device='cuda')
            sf = torch.empty((m, k // 16), dtype=torch.uint8, device='cuda')
            ret = quantlib.quantize_bf16(x.data_ptr(), out.data_ptr(), sf.data_ptr(), m, k,
                                        f32(1. / scale), torch.cuda.current_stream().cuda_stream)
            assert ret == 0, ret
            torch.cuda.synchronize()
            packed_cpu, sf_cpu, stats = quant_cpu(x.cpu(), scale)
            differences = {'packed_code_mismatches': int((out.cpu() != packed_cpu).sum()),
                           'scale_byte_mismatches': int((sf.cpu() != sf_cpu).sum())}
            report['quantizer_cases'].append(dict(label=label, **differences, **stats))
            assert differences['packed_code_mismatches'] == differences['scale_byte_mismatches'] == 0, differences
            return out.view(torch.int8), sf_layout(sf), decode(packed_cpu, sf_cpu, scale)

        def metric(y, ref):
            y, ref = y.float().cpu(), ref.float().cpu()
            assert torch.isfinite(y).all() and torch.isfinite(ref).all()
            delta = y - ref
            return {'max_abs_error': delta.abs().max().item(), 'rmse': delta.square().mean().sqrt().item(),
                    'relative_l2_error': (delta.norm() / ref.norm().clamp_min(1e-12)).item()}

        def check(y, ref, label, tolerance=.01):
            stat = metric(y, ref)
            assert stat['relative_l2_error'] <= tolerance, (label, stat)
            torch.testing.assert_close(y.float().cpu(), ref.float().cpu(), atol=.05, rtol=.02)
            return stat

        def swiglu(gate, up):
            return (F.silu(gate.float()).bfloat16().float() * up.float()).bfloat16()

        # Establish cuBLASLt output support on complete dimensions, retaining failures.
        torch.manual_seed(101)
        test_x = torch.randn(1, 2560).bfloat16().cuda()
        qa, qsf, qdecoded = quant_gpu_checked(test_x, weights['gate_proj']['input_scale'], 'output_dtype_probe')
        report['output_dtype_probes'] = []
        chosen = None
        for dtype in [torch.float32, torch.bfloat16]:
            probe = {'dtype': str(dtype)}
            try:
                out = torch.empty((1, 640), dtype=dtype, device='cuda')
                deep_gemm.cublaslt_nvfp4_gemm_nt((qa, qsf), (weights['gate_proj']['packed'], weights['gate_proj']['sf']), out)
                torch.cuda.synchronize()
                probe['status'] = 'PASS'
                if chosen is None:
                    chosen = dtype
            except Exception as exc:
                probe.update(status='KERNEL_FAIL', error=f'{type(exc).__name__}: {exc}')
            report['output_dtype_probes'].append(probe)
        assert chosen is not None
        report['selected_gemm_output_dtype'] = str(chosen)
        report['arithmetic_gate'] = {'per_projection_relative_l2_max': .01, 'end_to_end_relative_l2_max': .02,
                                     'elementwise_atol': .05, 'elementwise_rtol': .02,
                                     'activation_quantization_quality_gate': 'reported only; no model quality claim'}

        def gpu_linear(x, name, label):
            w = weights[name]
            qa, qs, xq = quant_gpu_checked(x, w['input_scale'], label)
            out = torch.empty((x.shape[0], w['shape'][0]), dtype=chosen, device='cuda')
            deep_gemm.cublaslt_nvfp4_gemm_nt((qa, qs), (w['packed'], w['sf']), out)
            # Real GPU global-scale epilogue, then BF16 output at the linear boundary.
            result = (out.float() * f32(w['input_scale'] * w['weight_scale_2'])).bfloat16()
            expected = (xq @ w['cpu'].T).bfloat16()
            return result, check(result, expected, label)

        def cpu_linear(x, name, quantized):
            w = weights[name]
            if quantized:
                q, sf, _ = quant_cpu(x, w['input_scale'])
                x = decode(q, sf, w['input_scale'])
            return (x.float() @ w['cpu'].T).bfloat16()

        # Quantizer edge cases; exact packed-byte comparisons, not float tolerances.
        scale = weights['gate_proj']['input_scale']
        for label, inp in [('zero', torch.zeros(1, 2560)),
                           ('alternating_outlier', torch.tensor([0., 1., -1., 1024., -1024.] * 512))]:
            quant_gpu_checked(inp.reshape(1, 2560).bfloat16().cuda(), scale, label)

        for batch, seed in [(1, 20261020), (8, 20261021), (32, 20261022)]:
            case = {'batch': batch, 'seed': seed, 'input': 'synthetic standard-normal rounded to BF16', 'status': 'KERNEL_FAIL'}
            try:
                torch.manual_seed(seed)
                x = torch.randn(batch, 2560).bfloat16()
                xgpu = x.cuda()
                g, gm = gpu_linear(xgpu, 'gate_proj', f'b{batch}_gate')
                u, um = gpu_linear(xgpu, 'up_proj', f'b{batch}_up')
                hidden = swiglu(g, u)
                cpu_hidden_from_gpu = swiglu(g.cpu(), u.cpu())
                torch.testing.assert_close(hidden.cpu(), cpu_hidden_from_gpu, atol=0, rtol=0)
                y, dm = gpu_linear(hidden, 'down_proj', f'b{batch}_down')
                gq, uq = cpu_linear(x, 'gate_proj', True), cpu_linear(x, 'up_proj', True)
                yq = cpu_linear(swiglu(gq, uq), 'down_proj', True)
                gf, uf = cpu_linear(x, 'gate_proj', False), cpu_linear(x, 'up_proj', False)
                yf = cpu_linear(swiglu(gf, uf), 'down_proj', False)
                arithmetic = check(y, yq, f'b{batch}_complete', .02)
                case.update(status='PASS', arithmetic_error_vs_quantized_reference=arithmetic,
                            activation_quantization_error_cpu_reference=metric(yq, yf),
                            total_error_vs_bf16_input_fp32_weight_reference=metric(y, yf),
                            projection_arithmetic_error={'gate': gm, 'up': um, 'down': dm},
                            input_quantization_error=metric(decode(*quant_cpu(x, scale)[:2], scale), x))
            except AssertionError as exc:
                case.update(status='NUMERIC_FAIL', error=str(exc))
                traceback.print_exc()
            except Exception as exc:
                case['error'] = f'{type(exc).__name__}: {exc}'
                traceback.print_exc()
            report['cases'].append(case)
            save()
            print(json.dumps(case), flush=True)
        report['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['torch_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['base_global_torch_libraries_loaded'] = '/usr/local/lib/python3.12/dist-packages/torch/lib/' in Path('/proc/self/maps').read_text()
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
