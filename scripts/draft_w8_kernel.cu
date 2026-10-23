#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

// Each warp owns one output channel and reuses its weight across all input rows.
// INT8 weights dequantize in registers; no full BF16 weight materialization.
template<int ROWS, bool F32>
__global__ void w8_kernel(const __nv_bfloat16* x,const int8_t* w,const float* scales,
                         void* output,int n,int k) {
  int lane=threadIdx.x&31;
  int row=blockIdx.x*4+(threadIdx.x>>5);
  if(row>=n)return;
  float sums[ROWS];
  #pragma unroll
  for(int m=0;m<ROWS;m++)sums[m]=0.f;
  for(int at=lane*4;at<k;at+=128) {
    int8_t q[4]={0,0,0,0};
    if(at+3<k && (k&3)==0) {
      int packed=*reinterpret_cast<const int*>(w+row*k+at);
      #pragma unroll
      for(int j=0;j<4;j++)q[j]=static_cast<int8_t>((packed>>(8*j))&255);
    }else{
      #pragma unroll
      for(int j=0;j<4;j++)if(at+j<k)q[j]=w[row*k+at+j];
    }
    #pragma unroll
    for(int m=0;m<ROWS;m++){
      #pragma unroll
      for(int j=0;j<4;j++)if(at+j<k)
        sums[m]=fmaf(__bfloat162float(x[m*k+at+j]),float(q[j]),sums[m]);
    }
  }
  #pragma unroll
  for(int m=0;m<ROWS;m++){
    #pragma unroll
    for(int d=16;d;d>>=1)sums[m]+=__shfl_down_sync(0xffffffff,sums[m],d);
    if(lane==0){
      float value=sums[m]*scales[row];
      if constexpr(F32)reinterpret_cast<float*>(output)[m*n+row]=value;
      else reinterpret_cast<__nv_bfloat16*>(output)[m*n+row]=__float2bfloat16(value);
    }
  }
}

template<bool F32>
void launch(torch::Tensor x,torch::Tensor w,torch::Tensor s,torch::Tensor out,cudaStream_t stream){
  int n=w.size(0),k=w.size(1),blocks=(n+3)/4;
  auto xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  #define RUN(M) case M: w8_kernel<M,F32><<<blocks,128,0,stream>>>(xp,w.data_ptr<int8_t>(),s.data_ptr<float>(),out.data_ptr(),n,k);break;
  switch(x.size(0)){RUN(1) RUN(2) RUN(3) RUN(4) RUN(5) RUN(6) RUN(7) RUN(8)}
  #undef RUN
}

torch::Tensor draft_w8_matmul(torch::Tensor x,torch::Tensor w,torch::Tensor scales,bool fp32){
  TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&scales.is_cuda(),"CUDA tensors required");
  TORCH_CHECK(x.device()==w.device()&&x.device()==scales.device(),"devices differ");
  TORCH_CHECK(x.scalar_type()==torch::kBFloat16&&w.scalar_type()==torch::kInt8&&scales.scalar_type()==torch::kFloat32,"expected BF16/INT8/FP32");
  TORCH_CHECK(x.dim()==2&&w.dim()==2&&scales.dim()==1,"invalid dimensions");
  TORCH_CHECK(x.size(0)>=1&&x.size(0)<=8&&x.size(1)==w.size(1)&&w.size(0)==scales.size(0)&&w.size(1)>0&&w.size(0)>0,"invalid shapes");
  TORCH_CHECK(x.is_contiguous()&&w.is_contiguous()&&scales.is_contiguous(),"contiguous tensors required");
  c10::cuda::CUDAGuard guard(x.device());
  auto out=torch::empty({x.size(0),w.size(0)},x.options().dtype(fp32?torch::kFloat32:torch::kBFloat16));
  auto stream=at::cuda::getCurrentCUDAStream(x.device().index());
  if(fp32)launch<true>(x,w,scales,out,stream);else launch<false>(x,w,scales,out,stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("matmul",&draft_w8_matmul,pybind11::arg("x"),pybind11::arg("w"),pybind11::arg("scales"),pybind11::arg("fp32")=false);
}
