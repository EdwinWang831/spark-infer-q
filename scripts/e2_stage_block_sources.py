#!/usr/bin/env python3
"""Stage unchanged external decoder sources as namespace packages, privately."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True, help="TensorFold's src/tensorfold directory")
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--include-loader', action='store_true', help='Include optional EXL3 imports required by the original weights loader')
    a = p.parse_args()
    files = []
    folders = ['families/qwen4_exp/cuda', 'cuda/kernels', 'cuda/nvfp4']
    if a.include_loader:
        folders.append('cuda/exl3')
    for folder in folders:
        files.extend(p for p in (a.source / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.cpp', '.cu', '.cuh') and p.name != '__init__.py')
    files.extend(a.source / p for p in [
        'cuda/build.py', 'cuda/direct_read.py', 'cuda/moe.py', 'cuda/experts.py',
        'cuda/prompt_precision.py', 'cuda/geometry.py',
        'families/qwen4_exp/host_table.py', 'families/qwen4_exp/ssd_table.py'])
    for sub in ('artifacts', 'logs', 'build'):
        (a.root / sub).mkdir(parents=True, exist_ok=True)
    pin = {}
    for path in sorted(files):
        name = str(path.relative_to(a.source))
        dest = a.root / 'reference/tensorfold' / name
        data = path.read_bytes()
        if dest.exists() and dest.read_bytes() != data:
            raise RuntimeError(f'existing reference differs: {name}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        pin[name] = {'sha256': hashlib.sha256(data).hexdigest()}
    (a.root / 'artifacts/source-manifest.json').write_text(json.dumps(pin, indent=2) + '\n')
    print(f'{len(pin)} unchanged source files staged')


if __name__ == '__main__':
    main()
