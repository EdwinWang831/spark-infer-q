"""Build the pinned Mia shared-input tile with a bounded PyTorch entry point."""
import hashlib
import os
from pathlib import Path

SOURCE_SHA='5d42a8d9e60e98b0b9b3325a20b5543f7c37c9dde1f72ce282b023a562136630'
HEADER_SHA='8de734885943e6f547feae4796aa1d7bdacf119494c99c776e4d0cc165d1fec0'
MIA_SHA='d09b7a01489679580d1083d2b7ff3a9d4d893d0ef8621ed8b84514689dbb0393'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(original,adapted):
    # The original is a source identity check; computation comes from the
    # three pinned Mia variants, with the original external epilogue header.
    assert 'void nvfp4_experts_cuda(' in original and adapted.count('TF_TILE4(fn_tile4_')==3
    preamble='''#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include "../experts.cuh"
'''
    host=r'''
namespace {
template <int M, int EPI>
void launch(const at::Tensor& x,int stride,int slots,const at::Tensor& w,const at::Tensor& scale,int kg,int nb,
    const at::Tensor& items,const at::Tensor& counts,const at::Tensor& members,at::Tensor& out,
    int n,float limit,int skip,int64_t max_units) {
  constexpr int STAGES=EPI==2?4:3;
  constexpr int SHARED=STAGES*(4096+4*M*576);
  static int blocks=0,sms=0;
#define SELECT(F) do { \
  if(blocks==0) {cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,F,128,SHARED); \
    sms=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;blocks=blocks<1?1:blocks;} \
  const int64_t need=(max_units/nb)*((nb+3)/4); \
  const int grid=(int)(need<(int64_t)blocks*sms?need:(int64_t)blocks*sms); \
  if(grid<1)return; \
  F<<<grid,128,SHARED,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), \
    stride,slots,reinterpret_cast<const uint4*>(w.data_ptr()),scale.data_ptr<float>(),kg,nb,items.data_ptr<int>(), \
    counts.data_ptr<int>(),members.data_ptr<int>(),out.data_ptr(),n,limit,skip); \
} while(0)
  if constexpr(EPI==2) {SELECT(fn_tile4_gu_rt4_s4);}
  else if constexpr(EPI==0) {SELECT(fn_tile4_f32_rt2_s3);}
  else {SELECT(fn_tile4_b16_rt2_s3);}
#undef SELECT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
void e44_experts_cuda(int64_t epi,const at::Tensor& x,int64_t stride,int64_t slots,const at::Tensor& w,
    const at::Tensor& scale,int64_t kg,int64_t nb,const at::Tensor& items,const at::Tensor& counts,
    const at::Tensor& members,at::Tensor& out,int64_t n,double limit,int64_t skip,int64_t max_units) {
  const c10::cuda::CUDAGuard guard(x.device());
  if(epi==2)launch<2,2>(x,stride,slots,w,scale,kg,nb,items,counts,members,out,n,limit,skip,max_units);
  else if(epi==0)launch<1,0>(x,stride,slots,w,scale,kg,nb,items,counts,members,out,n,limit,skip,max_units);
  else if(epi==3)launch<1,3>(x,stride,slots,w,scale,kg,nb,items,counts,members,out,n,limit,skip,max_units);
  else TORCH_CHECK(false,"unsupported expert epilogue");
}
'''
    return preamble+adapted+host


def build(session,source,manifest):
    from torch.utils.cpp_extension import load
    upstream=session.root.parent/'e2-load/reference/tensorfold/cuda/nvfp4/experts.cu'
    header=session.root.parent/'e10-target/external/cuda/experts.cuh'
    mia=source/'results/raw/e44/reference/fn_experts_tile.cu'
    assert sha(upstream)==SOURCE_SHA and sha(header)==HEADER_SHA and sha(mia)==MIA_SHA
    expected=mia.read_text().split('TF_TILE4(fn_tile4_gu_rt4_s3,')[0]
    expected+=('TF_TILE4(fn_tile4_gu_rt4_s4, 2, 2, 4, 4)\nTF_TILE4(fn_tile4_f32_rt2_s3, 1, 0, 2, 3)\nTF_TILE4(fn_tile4_b16_rt2_s3, 1, 3, 2, 3)\n')
    expected=expected.replace('#include "experts.cuh"','// Shared epilogue and cp.async helpers supplied by the pinned external header.')
    adapted=(source/'scripts/e44_tile.cuh').read_text();assert adapted.endswith(expected)
    folder=source/'build/cuda/nvfp4';folder.mkdir(parents=True);(folder.parent/'experts.cuh').write_bytes(header.read_bytes())
    generated=folder/'experts.cu';generated.write_text(transform(upstream.read_text(),adapted))
    binary=source/'build/binary';binary.mkdir()
    saved={k:os.environ.get(k) for k in ('TORCH_CUDA_ARCH_LIST','MAX_JOBS')}
    try:
        os.environ['TORCH_CUDA_ARCH_LIST']='12.1';os.environ['MAX_JOBS']='1'
        ext=load(name='spark_q_e44_pipeline_v1',sources=[str(source/'scripts/e44_bindings.cpp'),str(generated)],build_directory=str(binary),
            extra_cflags=['-O2','-g0'],extra_cuda_cflags=['-O3','-lineinfo','--threads=1','--ptxas-options=-v'],verbose=True)
    finally:
        for k,v in saved.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    return dict(status='PASS_E44_BUILD',baseline_source_sha256=SOURCE_SHA,header_sha256=HEADER_SHA,mia_source_sha256=MIA_SHA,
        mia_commit='656879fd752749bbd11a6e1a065a326c00940911',generated_sha256=sha(generated),binary_sha256=sha(ext.__file__),
        arch='sm_121',torch=session.torch.__version__,cuda=session.torch.version.cuda,
        policy=dict(tile=64,gate_up='rt4_s4',down='rt2_s3',minimum_rows=1024))
