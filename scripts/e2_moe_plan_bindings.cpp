// Only the provider changes: the planning implementation is original TensorFold.
#include <torch/extension.h>
void experts_plan_cuda(const at::Tensor&, int64_t, int64_t, int64_t,
    at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("plan", &experts_plan_cuda); }
