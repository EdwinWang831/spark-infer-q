#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
void e31_experts_cuda(int64_t, const at::Tensor&, int64_t, int64_t,
    const at::Tensor&, const at::Tensor&, int64_t, int64_t,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
    int64_t, double, int64_t, int64_t);
__global__ void decode_pairs(float* a, float* b) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=65536) return;
    a[2*i]=__half2float(__half(__nv_cvt_fp8_to_halfraw(i&255,__NV_E4M3)));
    a[2*i+1]=__half2float(__half(__nv_cvt_fp8_to_halfraw(i>>8,__NV_E4M3)));
    float2 v=__half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(i,__NV_E4M3)));
    b[2*i]=v.x;b[2*i+1]=v.y;
}
void check_decode(at::Tensor a,at::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.scalar_type()==at::kFloat && b.scalar_type()==at::kFloat &&
        a.is_contiguous() && b.is_contiguous() && a.numel()==131072 && b.numel()==131072 && a.device()==b.device());
    const c10::cuda::CUDAGuard guard(a.device());
    decode_pairs<<<256,256,0,at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<float>(),b.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("experts",&e31_experts_cuda);m.def("check_decode",&check_decode);}
