"""Compile one isolated SM121 CTA cooperative candidate from pinned upstream source."""
import hashlib
import json
import os
from pathlib import Path

SOURCE_SHA='5d42a8d9e60e98b0b9b3325a20b5543f7c37c9dde1f72ce282b023a562136630'
HEADER_SHA='8de734885943e6f547feae4796aa1d7bdacf119494c99c776e4d0cc165d1fec0'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def transform(source,pair):
    mark='template <int M, int EPI>\nvoid launch('
    guard='    if (e == skip) continue;'
    tail='  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}'
    assert source.count(mark)==source.count(guard)==source.count(tail)==1
    source=source.replace('nvfp4_expert_kernel','e39_single_kernel')
    source=source.replace(guard,'    if (e == skip || cnt > 16) continue;')
    source=source.replace(tail,'  C10_CUDA_KERNEL_LAUNCH_CHECK();\n'
        '  launch_pair<M,EPI>(x,x_stride,slots,w,scale,kg,nb,items,counts,members,out,n,limit,skip,max_units);\n}')
    source=source.replace(mark,pair+'\n'+mark)
    return source.replace('void nvfp4_experts_cuda(','void e39_experts_cuda(')


def build(session,source,manifest):
    from torch.utils.cpp_extension import load
    root=session.root;upstream=root.parent/'e2-load/reference/tensorfold/cuda/nvfp4/experts.cu'
    header=root.parent/'e10-target/external/cuda/experts.cuh'
    assert digest(upstream)==SOURCE_SHA and digest(header)==HEADER_SHA
    folder=source/'build/cuda/nvfp4';folder.mkdir(parents=True)
    (folder.parent/'experts.cuh').write_bytes(header.read_bytes())
    generated=folder/'experts.cu';generated.write_text(transform(upstream.read_text(),(source/'scripts/e39_cta.cuh').read_text()))
    binary=source/'build/binary';binary.mkdir()
    saved={k:os.environ.get(k) for k in ('TORCH_CUDA_ARCH_LIST','MAX_JOBS')}
    try:
        os.environ['TORCH_CUDA_ARCH_LIST']='12.1';os.environ['MAX_JOBS']='1'
        ext=load(name='spark_q_e39_prefill_v1',sources=[str(source/'scripts/e39_bindings.cpp'),str(generated)],
            build_directory=str(binary),extra_cflags=['-O2','-g0'],
            extra_cuda_cflags=['-O3','-lineinfo','--threads=1','--ptxas-options=-v'],verbose=True)
    finally:
        for k,v in saved.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    report=dict(status='PASS_E39_BUILD',source_sha256=manifest,baseline_source_sha256=SOURCE_SHA,
        header_sha256=HEADER_SHA,generated_sha256=digest(generated),binary_sha256=digest(Path(ext.__file__)),
        binary=str(ext.__file__),torch=session.torch.__version__,cuda=session.torch.version.cuda,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),model_reloaded=False,arch='sm_121')
    (source/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
