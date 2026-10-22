"""Stage a pinned external expert kernel and vary launch geometry/staging only."""
import argparse
import hashlib
import json
import os
from pathlib import Path

SOURCE_SHA = '5d42a8d9e60e98b0b9b3325a20b5543f7c37c9dde1f72ce282b023a562136630'
HEADER_SHA = '8de734885943e6f547feae4796aa1d7bdacf119494c99c776e4d0cc165d1fec0'


def transform(source):
    replacements = [
        ('template <int M>\n__device__ __forceinline__ void k_loop',
         'template <int M, int D>\n__device__ __forceinline__ void k_loop'),
        ('  constexpr int D = 2;\n', ''),
        ('template <int M, int EPI, int WARPS>', 'template <int M, int EPI, int WARPS, int D>'),
        ('k_loop<M>(acc', 'k_loop<M, D>(acc'),
        ('template <int M, int EPI>\nvoid launch', 'template <int M, int EPI, int WARPS, int D>\nvoid launch'),
        ('  constexpr int WARPS = 4;\n', ''),
    ]
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    assert source.count('nvfp4_expert_kernel<M, EPI, WARPS>') == 2
    source = source.replace('nvfp4_expert_kernel<M, EPI, WARPS>', 'nvfp4_expert_kernel<M, EPI, WARPS, D>')
    signature, body = source.split('void nvfp4_experts_cuda(', 1)
    args, body = body.split(') {', 1)
    prefix, dispatch = body.split('  if (epi == 2)', 1)
    assert dispatch.endswith('\n}\n')
    calls = []
    for warps in (1, 2, 4, 8):
        for depth in (1, 2):
            branch = '  if (epi == 2)' + dispatch[:-3]
            for m, epi in ((2, 2), (1, 0), (1, 3)):
                branch = branch.replace(f'launch<{m}, {epi}>', f'launch<{m}, {epi}, {warps}, {depth}>')
            calls.append(f'  if (warps == {warps} && depth == {depth}) {{\n{branch}\n    return;\n  }}')
    return (signature + 'void nvfp4_experts_tuned_cuda(' + args + ', int64_t warps, int64_t depth) {' +
            prefix + '\n'.join(calls) + '\n  TORCH_CHECK(false, "unsupported E10 geometry");\n}\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--external', type=Path, required=True)
    a = p.parse_args()
    import torch
    from torch.utils.cpp_extension import load
    assert torch.__version__ == '2.13.0+cu132'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '12.1'
    src = (a.external/'cuda/nvfp4/experts.cu').read_bytes()
    header = (a.external/'cuda/experts.cuh').read_bytes()
    assert hashlib.sha256(src).hexdigest() == SOURCE_SHA
    assert hashlib.sha256(header).hexdigest() == HEADER_SHA
    staged = a.root/'external/cuda/nvfp4'
    staged.mkdir(parents=True, exist_ok=True)
    (staged.parent/'experts.cuh').write_bytes(header)
    tuned = staged/'experts.cu'
    tuned.write_text(transform(src.decode()))
    build = a.root/'build'
    build.mkdir(exist_ok=True)
    mod = load(name='spark_q_e10_experts', sources=[str(a.root/'e10_expert_bindings.cpp'), str(tuned)],
               build_directory=str(build), extra_cflags=['-O2', '-g0'],
               extra_cuda_cflags=['-O3', '-lineinfo', '--threads=1', '--ptxas-options=-v'], verbose=True)
    report = dict(status='PASS_BUILD', original_source_sha256=SOURCE_SHA, header_sha256=HEADER_SHA,
                  transformed_source_sha256=hashlib.sha256(tuned.read_bytes()).hexdigest(),
                  binary_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
                  builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  cuda_arch='sm_121',
                  warps=[1, 2, 4, 8], staging_depth=[1, 2], arithmetic_changed=False)
    (a.root/'artifacts/build.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
