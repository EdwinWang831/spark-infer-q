// Access the existing pinned DeepInfer scheduler's public phase field.
#include <torch/extension.h>
#include "engine/scheduler.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("mode", [](py::object object) {
        return static_cast<int>(object.cast<Scheduler&>().engine_type);
    });
    module.def("set_prompt_mode", [](py::object object, bool enabled) {
        auto& scheduler = object.cast<Scheduler&>();
        TORCH_CHECK(scheduler.cp_align == 1, "prefill batching requires CP=1");
        TORCH_CHECK(scheduler.engine_type == EngineType::NORMAL ||
                    scheduler.engine_type == EngineType::PROMPT, "unexpected scheduler mode");
        scheduler.engine_type = enabled ? EngineType::PROMPT : EngineType::NORMAL;
    });
}
