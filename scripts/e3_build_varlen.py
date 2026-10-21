#!/usr/bin/env python3
"""Build only a thin binding to an externally pinned DeepInfer MTPVarlen."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--deepinfer', type=Path, required=True)
    a = p.parse_args()
    sha = subprocess.check_output(['git','-c','safe.directory='+str(a.deepinfer),'-C',str(a.deepinfer),'rev-parse','HEAD'], text=True).strip()
    assert sha == '774ce244b87cc0eb8d4b94ab9f5de2ab22049f01'
    subprocess.run(['git','-c','safe.directory='+str(a.deepinfer),'-C',str(a.deepinfer),'diff','--exit-code','--','csrc/engine'], check=True)
    from torch.utils.cpp_extension import load
    build = a.root / 'build/varlen'
    build.mkdir(parents=True,exist_ok=True)
    source = a.root / 'e3_varlen_bindings.cpp'
    mod = load(name='spark_q_e3_varlen', sources=[str(source)],
               extra_include_paths=[str(a.deepinfer/'csrc/engine')],
               extra_cflags=['-O2','-g0'], build_directory=str(build), verbose=True)
    report = {'deepinfer_commit':sha,'binding_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'external_headers_sha256':{n:hashlib.sha256((a.deepinfer/'csrc/engine'/n).read_bytes()).hexdigest()
                                         for n in ['mtp_varlen.hpp','sequence.hpp']},
              'extension_sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()}
    (a.root/'artifacts/varlen-build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
