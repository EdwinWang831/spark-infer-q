#include <torch/extension.h>
void e44_experts_cuda(int64_t,const at::Tensor&,int64_t,int64_t,const at::Tensor&,const at::Tensor&,
    int64_t,int64_t,const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor&,int64_t,double,int64_t,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("experts",&e44_experts_cuda);}
