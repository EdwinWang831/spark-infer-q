// Thin experiment binding: the budget algorithm remains the external DeepInfer header.
#include <pybind11/stl.h>
#include "mtp_varlen.hpp"

class BudgetProbe {
    MTPVarlen scheduler;
public:
    void curve(std::vector<std::pair<int, float>> values) {
        TORCH_CHECK(!values.empty(), "curve must be nonempty");
        int prev = 0;
        for (auto [n, ms] : values) {
            TORCH_CHECK(n > prev && std::isfinite(ms) && ms > 0, "invalid cost curve");
            prev = n;
        }
        scheduler.set_decode_perf_curve(std::move(values));
    }
    void runtime(int n, int rows, double forward_ms, double sampler_ms) {
        TORCH_CHECK(!scheduler.get_decode_perf_curve().empty() &&
                    rows <= scheduler.get_decode_perf_curve().back().first, "runtime outside curve");
        scheduler.update_runtime(n, rows, forward_ms, sampler_ms);
    }
    std::vector<int> schedule(py::object params, const std::vector<py::dict>& entries) {
        std::vector<std::shared_ptr<Sequence>> seqs;
        for (const auto& e : entries) {
            SequenceInit init;
            init.generation.params = params;
            init.mtp.draft_tokens = e["draft_tokens"].cast<std::vector<int>>();
            init.mtp.accept_rates = e["accept_rates"].cast<std::vector<float>>();
            init.mtp.prev_accept_rates = e["prev_accept_rates"].cast<std::vector<float>>();
            init.mtp.process_cnt = e["process_cnt"].cast<int>();
            init.mtp.schedule_cnt = e["schedule_cnt"].cast<int>();
            init.mtp.has_draft_info = true;
            for (const auto* key : {"accept_rates", "prev_accept_rates"}) {
                const auto rates = e[key].cast<std::vector<float>>();
                TORCH_CHECK(rates.empty() || rates.size() == init.mtp.draft_tokens.size(), "rate width");
                float previous = 1;
                for (float r : rates) {
                    TORCH_CHECK(std::isfinite(r) && r >= 0 && r <= previous, "rates must be cumulative probabilities");
                    previous = r;
                }
            }
            seqs.push_back(std::make_shared<Sequence>(std::move(init)));
        }
        TORCH_CHECK(!seqs.empty(), "empty batch");
        if (!scheduler.get_decode_perf_curve().empty())
            TORCH_CHECK(seqs.size() <= (size_t)scheduler.get_decode_perf_curve().back().first, "batch outside curve");
        std::vector<int> result;
        scheduler.schedule(seqs, result);
        return result;
    }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BudgetProbe>(m, "BudgetProbe")
        .def(py::init<>()).def("curve", &BudgetProbe::curve)
        .def("runtime", &BudgetProbe::runtime).def("schedule", &BudgetProbe::schedule);
}
