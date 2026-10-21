#!/usr/bin/env python3
"""Build the original decoder GDN, prefill GDN and front/back CUDA kernels."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    import torch
    from torch.utils.cpp_extension import load
    assert torch.__version__ == '2.13.0+cu132'
    ref = root / 'reference/tensorfold'
    pins = json.loads((root / 'artifacts/source-manifest.json').read_text())
    for name, item in pins.items():
        assert hashlib.sha256((ref / name).read_bytes()).hexdigest() == item['sha256']
    family = ref / 'families/qwen4_exp/cuda'
    shared = ref / 'cuda/kernels'
    builds = {'spark_q_block_gdn': ([family/'gdn.cpp', family/'gdn.cu'], True),
              'spark_q_block_gdn_io': ([family/'gdn_io.cpp', family/'gdn_io.cu'], True),
              'spark_q_block_gdn_shared': ([shared/f for f in ('gdn.cpp', 'gdn.cu', 'gdn_prefill.cu')], True)}
    report = {'source_patched': False, 'cuda_arch_target': '12.1', 'modules': {}}
    for name, (sources, no_fma) in builds.items():
        dest = root / 'build' / name
        dest.mkdir(parents=True, exist_ok=True)
        mod = load(name=name, sources=list(map(str, sources)), build_directory=str(dest),
                   extra_cflags=['-O2', '-g0'],
                   extra_cuda_cflags=['-O3', '--threads=1'] + (['--fmad=false'] if no_fma else []), verbose=True)
        binary = Path(mod.__file__)
        report['modules'][name] = {'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
        (root / 'artifacts/build.json').write_text(json.dumps(report, indent=2) + '\n')
    report.update(status='PASS', cuda_initialized=torch.cuda.is_initialized())
    (root / 'artifacts/build.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
