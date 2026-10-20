#!/usr/bin/env python3
"""Diagnostic control: BF16 activations, one expert's stored weights dequantized
to FP32. This temporary staging is not a full-model deployment strategy.
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
    root = p.parse_args().root
    report = {'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'BF16 inputs with temporary FP32-dequantized single-expert weights',
              'native_w4a16_kernel_tested': False, 'performance_measured': False, 'cases': []}
    try:
        import torch
        import torch.nn.functional as F
        assert torch.__version__ == '2.13.0+cu132' and torch.cuda.get_device_capability() == (12, 1)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(512 * 1024**2 / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        spec = importlib.util.spec_from_file_location('tf_nvfp4_control', root / 'reference/tensorfold_nvfp4.py')
        tf = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = tf
        spec.loader.exec_module(tf)
        manifest = json.loads((root / 'expert/manifest.json').read_text())
        cpu, gpu = {}, {}
        for name, meta in manifest['projections'].items():
            n, k = meta['shape']
            for filename, item in meta['artifacts'].items():
                data = (root / 'expert' / filename).read_bytes()
                assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
            data = torch.frombuffer(bytearray((root / 'expert' / (name + '-weight.bin')).read_bytes()), dtype=torch.uint8).reshape(n, k // 2)
            scale = torch.frombuffer(bytearray((root / 'expert' / (name + '-scales.bin')).read_bytes()), dtype=torch.uint8).reshape(n, k // 16)
            cpu[name] = tf.dequantize(data, scale, meta['weight_scale_2'])
            gpu[name] = cpu[name].cuda()

        def forward(x, weights):
            gate = (x.float() @ weights['gate_proj'].T).bfloat16()
            up = (x.float() @ weights['up_proj'].T).bfloat16()
            hidden = (F.silu(gate.float()).bfloat16().float() * up.float()).bfloat16()
            return (hidden.float() @ weights['down_proj'].T).bfloat16()

        for batch, seed in [(1, 20261020), (8, 20261021), (32, 20261022)]:
            case = {'batch': batch, 'seed': seed}
            try:
                torch.manual_seed(seed)
                x = torch.randn(batch, 2560).bfloat16()
                expected = forward(x, cpu).float()
                y = forward(x.cuda(), gpu).float().cpu()
                assert torch.isfinite(y).all()
                delta = y - expected
                err = (delta.norm() / expected.norm().clamp_min(1e-12)).item()
                case.update(relative_l2_error=err, max_abs_error=delta.abs().max().item(),
                            rmse=delta.square().mean().sqrt().item())
                assert err <= .002
                torch.testing.assert_close(y, expected, atol=.02, rtol=.01)
                case['status'] = 'PASS'
            except Exception as exc:
                case.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
            report['cases'].append(case)
            print(json.dumps(case), flush=True)
        report.update(status='PASS' if all(x['status'] == 'PASS' for x in report['cases']) else 'PARTIAL_OR_FAIL',
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      fp32_weight_staging_bytes=sum(w.numel() * 4 for w in cpu.values()),
                      relative_l2_gate=.002, elementwise_atol=.02, elementwise_rtol=.01,
                      base_global_torch_libraries_loaded='/usr/local/lib/python3.12/dist-packages/torch/lib/' in Path('/proc/self/maps').read_text())
        assert not report['base_global_torch_libraries_loaded']
    except Exception as exc:
        report.update(status='SETUP_OR_TEST_FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root / 'artifacts/expert-bf16-control.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
