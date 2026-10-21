#!/usr/bin/env python3
"""Copy selected external sources unchanged to the private experiment root.

--source is TensorFold's src/tensorfold directory. Namespace package loading
avoids importing unrelated runtime backends. Third-party code is not vendored.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


FILES = ['cuda/moe.py', 'cuda/experts.py', 'cuda/experts.cu', 'cuda/experts.cuh',
         'cuda/prompt_precision.py', 'cuda/kernels/qmm_frag.cuh',
         'cuda/kernels/qmm.py', 'cuda/kernels/qmm_tiles.py',
         'families/qwen4_exp/cuda/nvfp4.py', 'families/qwen4_exp/cuda/nvfp4_moe.py'] + [
         'cuda/nvfp4/' + f for f in ('linear.py', 'experts.py', 'experts.cu',
                                   'qmmf.cpp', 'qmmf.cu', 'prompt.cpp', 'prompt.cu')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    for sub in ('artifacts', 'logs', 'build'):
        (a.root / sub).mkdir(parents=True, exist_ok=True)
    pin = {}
    for name in FILES:
        src, dest = a.source / name, a.root / 'reference/tensorfold' / name
        data = src.read_bytes()
        if dest.exists() and dest.read_bytes() != data:
            raise RuntimeError(f'existing reference differs: {name}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        pin[name] = {'sha256': hashlib.sha256(data).hexdigest()}
    (a.root / 'artifacts/source-manifest.json').write_text(json.dumps(pin, indent=2) + '\n')
    print(f'{len(pin)} unchanged source files staged')


if __name__ == '__main__':
    main()
