// Thin test binding to the external, unmodified TensorFold experts.cu.
#include <torch/extension.h>

void nvfp4_experts_cuda(int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots,
    const at::Tensor& w, const at::Tensor& scale, int64_t kg, int64_t nb,
    const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
    at::Tensor& out, int64_t n, double limit, int64_t skip, int64_t max_units);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("experts", &nvfp4_experts_cuda);
}
