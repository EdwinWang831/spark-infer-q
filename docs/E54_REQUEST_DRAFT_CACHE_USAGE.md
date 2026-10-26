# Request-owned rolling draft projections

E54 adds an explicit `request_projection_cache=True` option to the existing
rolling provider. For two serialized adaptive requests, prepare it with:

```python
provider, metadata = optimized_provider(
    session, history_window_rows=2040, max_request_context=262160,
    query_graphs=True, request_projection_cache=True,
)
```

Use the two-slot target and `serialized_postdraft=True` engine pattern from
[E53](E53_LONG_CONCURRENCY_USAGE.md). Target/prefill KV format and ring geometry
must match. Admission must include the three projection tensors per request
in addition to its feature window and target state. At this checkpoint's
max_context2048 they occupy50MiB/request; other geometries must be remeasured.
The shared padding/query buffers and model weights are not copied.

`DraftState` owns the cache. Before a draft call it binds that request's context
rows, K/V and rolling counters to the worker, then restores the shared worker
on normal return or failure. It preserves incremental computation when requests
alternate. A foreign owner, another worker, active nested bind or released cache
is rejected. Backend release fences before clearing the cache; a new handle
generation obtains a fresh cache. Do not carry cache objects across providers.

Default is off so historical rebuild experiments remain reproducible. This
option addresses request isolation and does not quantize weights, change the
predictor or cost model, enable GPU batching, or increase Graph slots beyond2.
Its populated256K acceptance and remaining limits are recorded in
[the report](E54_REQUEST_DRAFT_CACHE_REPORT.md). No eight-way capacity or serving
speed claim follows from cache isolation.
