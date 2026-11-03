# Four-request 128K prefill batching

[简体中文](ENGINE_V07_USAGE.md) · [Report](ENGINE_V07_REPORT.md) · [Plan](ENGINE_V07_PLAN.md)

In the prepared single-DGX-Spark environment, explicitly select prompt forwards
with up to 1,024 rows per request and 4,096 rows in total. This profile requires
four 128K slots, adaptive DFlash, Graph decode, the E44 expert extension, linear
features and the validated native prompt-control extension.

```bash
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --capacity 131072 --max-requests 4 --mode adaptive --execution graph \
  --prefill-experts /path/to/spark_q_e44_pipeline_v1.so \
  --prefill-chunk-rows 1024 --feature-history-layout linear \
  --prefill-batch-control /path/to/kaifra_prompt_control_v1.so
```

Build arguments for the control extension are in
[`k4batch_engine_batch_memory.py`](../scripts/k4batch_engine_batch_memory.py),
using [`k4batch_prompt_control.cpp`](../scripts/k4batch_prompt_control.cpp).
The loader checks the sealed DeepInfer revision, core extension, three headers
and control binary. The pinned binary corresponds to the reported environment.

The 131,072-token limit includes input and output for each request. A 64-token
completion leaves 131,008 input tokens per request. Admission uses measured
process memory with an 88 GiB allocator cap, an 8 GiB host reserve and at least
1 GiB transient allowance.

Health reports `prefill_gpu_batch=true`, `prefill_batch_requests=4` and
`prefill_workspace_rows=4096`. Decode executes requests serially over private
Graph slots, reflected by `gpu_batch=false`. Prompt operators execute eagerly,
reflected by `prefill_cuda_graph=false`. A forward accepts one to four active
requests, excluding cancelled requests before shared execution. A device fence
completes state changes before native output publication.

Mean complete four-long-request time falls by 10.29%, with increased time to
first token for earlier requests. See the [latency table](ENGINE_V07_REPORT.md#每请求延迟).
Chat, queueing, cancellation, health and metrics follow the
[128K guide](ENGINE_V04_USAGE.en.md). Ordinary startup selects the
[mirrored/512-row profile](ENGINE_V05_USAGE.en.md), whose guide also describes
the explicit serial 1,024-row profile.
