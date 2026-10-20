#!/usr/bin/env python3
"""Build just TensorFold's original NVFP4 expert kernel in the isolated builder."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    report = {'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_patched': False, 'cuda_arch_target': '12.1', 'gpu_execution_tested': False}
    try:
        import torch
        from torch.utils.cpp_extension import load
        assert torch.__version__ == '2.13.0+cu132'
        source = root / 'reference/tensorfold/cuda/nvfp4/experts.cu'
        header = root / 'reference/tensorfold/cuda/experts.cuh'
        pin = json.loads((root / 'artifacts/source-manifest.json').read_text())
        for path, name in [(source, 'cuda/nvfp4/experts.cu'), (header, 'cuda/experts.cuh')]:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == pin[name]['sha256']
        module = load(name='spark_q_native_expert_probe',
                      sources=[str(root / 'e2_native_bindings.cpp'), str(source)],
                      extra_cflags=['-O2', '-g0'], extra_cuda_cflags=['-O3', '-lineinfo', '--threads=1'],
                      build_directory=str(root / 'build'), verbose=True)
        binary = Path(module.__file__)
        report.update(status='PASS', torch=torch.__version__, binary_name=binary.name,
                      binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      cuda_initialized=torch.cuda.is_initialized())
    except Exception as exc:
        report.update(status='BUILD_OR_IMPORT_FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (root / 'artifacts/native-build.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
