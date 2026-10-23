"""Compile only a new W8 extension; no checkpoint load or inference job submission."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main(root):
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST']='12.1';os.environ['MAX_JOBS']='1'
    build=root/'build/draft_w8';build.mkdir(parents=True,exist_ok=True)
    src=root/'draft_w8_kernel.cu'
    module=load(name='spark_q_draft_w8',sources=[str(src)],build_directory=str(build),
        extra_cuda_cflags=['-O3','-lineinfo'],verbose=False)
    report=dict(status='BUILT_NOT_VALIDATED',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),binary=Path(module.__file__).name)
    dest=root/'artifacts/e28-build.json'
    with dest.open('x') as h:h.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
