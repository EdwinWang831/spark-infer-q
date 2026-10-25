#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

// Owned W8 batch kernel: each warp reuses a channel's INT8 weights across
// up to 32 draft rows. Keep the original per-row FMA/reduction/scale order.
template<int ROWS>
__global__ void batch_w8(const __nv_bfloat16* x,const int8_t* w,const float* scales,
                         __nv_bfloat16* output,int rows,int n,int k) {
  int lane=threadIdx.x&31, channel=blockIdx.x*4+(threadIdx.x>>5);
  if(channel>=n)return;
  float sums[ROWS];
  #pragma unroll
  for(int m=0;m<ROWS;m++)sums[m]=0.f;
  for(int at=lane*4;at<k;at+=128) {
    int8_t q[4]={0,0,0,0};
    if(at+3<k && (k&3)==0) {
      int packed=*reinterpret_cast<const int*>(w+channel*k+at);
      #pragma unroll
      for(int j=0;j<4;j++)q[j]=static_cast<int8_t>((packed>>(8*j))&255);
    }else{
      #pragma unroll
      for(int j=0;j<4;j++)if(at+j<k)q[j]=w[channel*k+at+j];
    }
    #pragma unroll
    for(int m=0;m<ROWS;m++)if(m<rows){
      #pragma unroll
      for(int j=0;j<4;j++)if(at+j<k)
        sums[m]=fmaf(__bfloat162float(x[m*k+at+j]),float(q[j]),sums[m]);
    }
  }
  #pragma unroll
  for(int m=0;m<ROWS;m++)if(m<rows){
    #pragma unroll
    for(int d=16;d;d>>=1)sums[m]+=__shfl_down_sync(0xffffffff,sums[m],d);
    if(lane==0)output[m*n+channel]=__float2bfloat16(sums[m]*scales[channel]);
  }
}

torch::Tensor matmul(torch::Tensor x,torch::Tensor w,torch::Tensor scales) {
  TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&scales.is_cuda(),"CUDA tensors required");
  TORCH_CHECK(x.device()==w.device()&&x.device()==scales.device(),"devices differ");
  TORCH_CHECK(x.scalar_type()==torch::kBFloat16&&w.scalar_type()==torch::kInt8&&scales.scalar_type()==torch::kFloat32,"expected BF16/INT8/FP32");
  TORCH_CHECK(x.dim()==2&&w.dim()==2&&scales.dim()==1,"invalid dimensions");
  TORCH_CHECK(x.size(0)>=1&&x.size(0)<=32&&x.size(1)==w.size(1)&&w.size(0)==scales.size(0)&&w.size(1)>0&&w.size(0)>0,"invalid shapes");
  TORCH_CHECK(x.is_contiguous()&&w.is_contiguous()&&scales.is_contiguous(),"contiguous tensors required");
  c10::cuda::CUDAGuard guard(x.device());
  auto out=torch::empty({x.size(0),w.size(0)},x.options());
  auto stream=at::cuda::getCurrentCUDAStream(x.device().index());
  auto xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto op=reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  int rows=x.size(0),n=w.size(0),k=w.size(1),blocks=(n+3)/4;
  #define RUN(R) batch_w8<R><<<blocks,128,0,stream>>>(xp,w.data_ptr<int8_t>(),scales.data_ptr<float>(),op,rows,n,k)
  if(rows<=8){RUN(8);}else if(rows<=16){RUN(16);}else{RUN(32);}
  #undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("matmul",&matmul);}
