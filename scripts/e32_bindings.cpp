#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
void e32_qmmf_cuda(const at::Tensor&,const at::Tensor&,const at::Tensor&,double,at::Tensor,int64_t,int64_t,int64_t,bool);
void qmmf(const at::Tensor& x,const at::Tensor& w,const at::Tensor& bs,double scale,at::Tensor out,
          int64_t n,int64_t sk,int64_t npad,bool compact) {
    TORCH_CHECK(x.is_cuda() && x.dim()==2 && x.scalar_type()==at::kBFloat16 && x.stride(1)==1 &&
        x.size(0)>=1 && x.size(0)<=8 && x.stride(0)%8==0 && x.data_ptr()!=nullptr,"x rows1..8, bf16, aligned stride");
    const int64_t k=x.size(1),m=x.size(0);
    TORCH_CHECK(sk>=1 && sk<=8 && (sk&(sk-1))==0 && k%64==0 && (k/64)%sk==0,"whole groups and split-K1/2/4/8");
    TORCH_CHECK(n>0 && npad>=n && npad%64==0 && w.is_contiguous() && w.scalar_type()==at::kByte && w.numel()==npad*k,"MXFP8 weight bytes");
    TORCH_CHECK(bs.is_contiguous() && bs.scalar_type()==at::kByte && bs.numel()==npad*(k/32),"MXFP8 scale bytes");
    TORCH_CHECK(out.is_contiguous() && out.scalar_type()==at::kBFloat16 && out.dim()==2 && out.size(0)==m && out.size(1)==n,"BF16 output shape");
    TORCH_CHECK(w.device()==x.device() && bs.device()==x.device() && out.device()==x.device(),"same device");
    const c10::cuda::CUDAGuard guard(x.device());
    e32_qmmf_cuda(x,w,bs,scale,out,n,sk,npad,compact);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("qmmf",&qmmf);}
