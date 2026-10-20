#!/usr/bin/env python3
"""Build original TensorFold kernels, without changing the live runtime."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    root = p.parse_args().root
    import torch
    from torch.utils.cpp_extension import load
    assert torch.__version__ == '2.13.0+cu132'
    ref = root / 'reference/tensorfold'
    pin = json.loads((root / 'artifacts/source-manifest.json').read_text())
    for name, item in pin.items():
        assert hashlib.sha256((ref / name).read_bytes()).hexdigest() == item['sha256']
    builds = {
        'spark_q_moe_linear': [ref / 'cuda/nvfp4' / f for f in ('qmmf.cpp', 'qmmf.cu', 'experts.cu')],
        'spark_q_moe_plan': [root / 'e2_moe_plan_bindings.cpp', ref / 'cuda/experts.cu'],
        'spark_q_moe_prompt': [ref / 'cuda/nvfp4' / f for f in ('prompt.cpp', 'prompt.cu')],
    }
    report = {'source_patched': False, 'cuda_arch_target': '12.1', 'modules': {}}
    for name, sources in builds.items():
        dest = root / 'build' / name
        dest.mkdir(parents=True, exist_ok=True)
        mod = load(name=name, sources=list(map(str, sources)), build_directory=str(dest),
                   extra_cflags=['-O2', '-g0'], extra_cuda_cflags=['-O3', '--threads=1'], verbose=True)
        binary = Path(mod.__file__)
        report['modules'][name] = {'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
        (root / 'artifacts/build.json').write_text(json.dumps(report, indent=2) + '\n')
    report.update(status='PASS', cuda_initialized=torch.cuda.is_initialized())
    (root / 'artifacts/build.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
