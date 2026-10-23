"""Build one pinned expert candidate: packed FP8 scale conversion, no geometry changes."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
SOURCE_SHA = '5d42a8d9e60e98b0b9b3325a20b5543f7c37c9dde1f72ce282b023a562136630'
HEADER_SHA = '8de734885943e6f547feae4796aa1d7bdacf119494c99c776e4d0cc165d1fec0'

OLD = '''        const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);'''
NEW = '''        const __half2 hs = __half2(__nv_cvt_fp8x2_to_halfraw2(
            static_cast<__nv_fp8x2_storage_t>((sw >> sh) & 0xFFFFu), __NV_E4M3));
        const float2 fs = __half22float2(hs);
        const float s0 = fs.x, s1 = fs.y;'''


def transform(source):
    assert source.count(OLD) == 1
    assert source.count('void nvfp4_experts_cuda(') == 1
    return source.replace(OLD, NEW).replace('void nvfp4_experts_cuda(', 'void e31_experts_cuda(')


def main(root):
    sys.path[:0] = [str(root/'followup-deps'), str(root/'python-deps')]
    import torch
    assert torch.__version__ == '2.13.0+cu132', torch.__version__
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST'] = '12.1'
    os.environ['MAX_JOBS'] = '1'
    source = (root.parent/'e2-load/reference/tensorfold/cuda/nvfp4/experts.cu').read_bytes()
    header = (root.parent/'e10-target/external/cuda/experts.cuh').read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
    assert hashlib.sha256(header).hexdigest() == HEADER_SHA
    staged = root/'build/e31/cuda/nvfp4'; staged.mkdir(parents=True, exist_ok=True)
    (staged.parent/'experts.cuh').write_bytes(header)
    candidate = staged/'experts.cu'; candidate.write_text(transform(source.decode()))
    build = root/'build/e31/binary'; build.mkdir(exist_ok=True)
    mod = load(name='spark_q_e31_experts', sources=[str(root/'e31_bindings.cu'), str(candidate)],
        build_directory=str(build), extra_cflags=['-O2','-g0'],
        extra_cuda_cflags=['-O3','-lineinfo','--threads=1','--ptxas-options=-v'], verbose=True)
    from e28_probe import sha
    report = dict(status='PASS_BUILD', source_sha256=sha(candidate), header_sha256=HEADER_SHA,
        baseline_source_sha256=SOURCE_SHA,binary_sha256=sha(mod.__file__),
        builder_sha256=sha(root/'e31_build.py'),bindings_sha256=sha(root/'e31_bindings.cu'),
        binary=mod.__file__,torch=torch.__version__,cuda=torch.version.cuda,arch='sm_121')
    with (root/'artifacts/e31-build.json').open('x') as f: f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
