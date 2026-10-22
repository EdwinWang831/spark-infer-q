// Binding for an externally staged TensorFold kernel; no upstream source vendored.
#include <torch/extension.h>
void nvfp4_experts_tuned_cuda(int64_t, const at::Tensor&, int64_t, int64_t,
    const at::Tensor&, const at::Tensor&, int64_t, int64_t,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
    int64_t, double, int64_t, int64_t, int64_t, int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("experts", &nvfp4_experts_tuned_cuda);
}
