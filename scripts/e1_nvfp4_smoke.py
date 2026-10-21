#!/usr/bin/env python3
"""Compare DeepGEMM's cuBLASLt NVFP4 route with CPU dequantized FP32 math.

Only a small extracted matrix is loaded, never a complete checkpoint.
The probe evaluates the stored quantized representation, not model quality.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--slice', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    report = {'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'component': 'DeepGEMM cuBLASLt NVFP4', 'performance_measured': False, 'cases': []}

    def save():
        a.report.write_text(json.dumps(report, indent=2) + '\n')

    try:
        import torch
        import deep_gemm
        assert torch.__version__ == '2.13.0+cu132'
        assert torch.cuda.get_device_capability() == (12, 1)
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(512 * 1024**2 / torch.cuda.get_device_properties(0).total_memory)
        report.update(torch=torch.__version__, cuda_build=torch.version.cuda,
                      deepgemm_version=deep_gemm.__version__, capability=[12, 1])
        manifest = json.loads((a.slice / 'manifest.json').read_text())
        report['weight_slice'] = manifest
        n, k = manifest['slice_shape']
        assert n <= 128 and k <= 256
        for filename, artifact in manifest['artifacts'].items():
            raw = (a.slice / filename).read_bytes()
            assert len(raw) == artifact['bytes']
            assert hashlib.sha256(raw).hexdigest() == artifact['sha256']
        b = torch.frombuffer(bytearray((a.slice / 'weight.bin').read_bytes()), dtype=torch.uint8).reshape(n, k // 2)
        bs = torch.frombuffer(bytearray((a.slice / 'scales.bin').read_bytes()), dtype=torch.uint8).reshape(n, k // 16)
        global_scale = manifest['global_scale']
        assert global_scale > 0
        lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])

        def decode(packed, scale):
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
            values = lut[codes & 7] * torch.where(codes & 8 != 0, -1., 1.)
            sf = scale.contiguous().view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
            assert torch.isfinite(sf).all() and (sf >= 0).all()
            return values * sf

        def tile_sf(sf):
            # cuBLASLt vec16 layout, [ceil(rows/128),ceil(K/64),32,4,4].
            rows, cols = sf.shape
            pad = torch.zeros(((rows + 127) // 128 * 128, (cols + 3) // 4 * 4), dtype=torch.uint8)
            pad[:rows, :cols] = sf
            return pad.view(pad.shape[0] // 128, 4, 32, pad.shape[1] // 4, 4).permute(0, 3, 2, 1, 4).contiguous().cuda()

        torch.cuda.reset_peak_memory_stats()
        for seed, actual in [(11, False), (23, True), (47, True)]:
            case = {'seed': seed, 'actual_qwen_slice': actual, 'status': 'KERNEL_FAIL', 'shape_mnk': [32, n, k]}
            try:
                torch.manual_seed(seed)
                lhs = torch.randint(0, 256, (32, k // 2), dtype=torch.uint8)
                choices = torch.tensor([0x30, 0x38, 0x3c, 0x40], dtype=torch.uint8)
                ls = choices[torch.randint(0, 4, (32, k // 16))]
                rhs = b if actual else torch.randint(0, 256, b.shape, dtype=torch.uint8)
                rs = bs if actual else choices[torch.randint(0, 4, bs.shape)]
                ref = decode(lhs, ls) @ decode(rhs, rs).T
                out = torch.empty((32, n), dtype=torch.bfloat16, device='cuda')
                deep_gemm.cublaslt_nvfp4_gemm_nt((lhs.view(torch.int8).cuda(), tile_sf(ls)),
                                                (rhs.view(torch.int8).cuda(), tile_sf(rs)), out)
                torch.cuda.synchronize()
                result = out.float().cpu()
                torch.testing.assert_close(result, ref, atol=.0625, rtol=.008)
                scale = global_scale if actual else 1.
                delta = result - ref
                case.update(status='PASS', max_abs_error_before_global_scale=delta.abs().max().item(),
                            relative_l2_error=(delta.norm() / ref.norm().clamp_min(1e-12)).item(),
                            max_abs_error_after_global_scale=(delta * scale).abs().max().item(),
                            global_scale_applied_outside_gemm=scale, atol=.0625, rtol=.008)
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
