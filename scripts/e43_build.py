"""Replace only the pinned expert compute_stage, compile an isolated extension."""
import hashlib
import json
import os
from pathlib import Path

SOURCE_SHA='5d42a8d9e60e98b0b9b3325a20b5543f7c37c9dde1f72ce282b023a562136630'
HEADER_SHA='8de734885943e6f547feae4796aa1d7bdacf119494c99c776e4d0cc165d1fec0'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(source,compute):
    start='template <int M>\n__device__ __forceinline__ void compute_stage('
    end='template <int M>\n__device__ __forceinline__ void k_loop('
    assert source.count(start)==source.count(end)==1
    lo=source.index(start);hi=source.index(end);assert hi>lo
    return (source[:lo]+compute+'\n'+source[hi:]).replace('nvfp4_expert_kernel','e43_pipeline_kernel').replace('void nvfp4_experts_cuda(','void e43_experts_cuda(')


def build(session,source,manifest):
    from torch.utils.cpp_extension import load
    upstream=session.root.parent/'e2-load/reference/tensorfold/cuda/nvfp4/experts.cu'
    header=session.root.parent/'e10-target/external/cuda/experts.cuh'
    assert sha(upstream)==SOURCE_SHA and sha(header)==HEADER_SHA
    folder=source/'build/cuda/nvfp4';folder.mkdir(parents=True)
    (folder.parent/'experts.cuh').write_bytes(header.read_bytes())
    generated=folder/'experts.cu';generated.write_text(transform(upstream.read_text(),(source/'scripts/e43_compute.cuh').read_text()))
    binary=source/'build/binary';binary.mkdir()
    saved={k:os.environ.get(k) for k in ('TORCH_CUDA_ARCH_LIST','MAX_JOBS')}
    try:
        os.environ['TORCH_CUDA_ARCH_LIST']='12.1';os.environ['MAX_JOBS']='1'
        ext=load(name='spark_q_e43_pipeline_v1',sources=[str(source/'scripts/e43_bindings.cpp'),str(generated)],
            build_directory=str(binary),extra_cflags=['-O2','-g0'],
            extra_cuda_cflags=['-O3','-lineinfo','--threads=1','--ptxas-options=-v'],verbose=True)
    finally:
        for k,v in saved.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    return dict(status='PASS_E43_BUILD',baseline_source_sha256=SOURCE_SHA,header_sha256=HEADER_SHA,
        generated_sha256=sha(generated),binary_sha256=sha(ext.__file__),arch='sm_121',
        torch=session.torch.__version__,cuda=session.torch.version.cuda)
