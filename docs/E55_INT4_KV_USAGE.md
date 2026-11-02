# E55 INT4 KV research route

E55 adds explicit `kv_dtype='int4'` support to `LongTarget` and
`TensorFoldPrefillProvider`, including `index_ring_rows=2048`. Both sides must
use the same format and ring geometry. Existing defaults remain BF16.

The route retains TensorFold's existing H32/group32/FP16-scale quantizer and
attention arithmetic. Our adapter passes its bit width to the original prep
body and translates only raw-index addresses. Full K/V history, all scales,
BF16 pooled QSA history and recurrent state remain resident. This is a KV
change; target and draft weights are unchanged.

For a serialized single-request research run:

```python
target = SharedGraphTarget.from_session(
    session, capacity=262160, kv_dtype='int4', slots=1,
    index_ring_rows=2048,
)
provider, metadata = optimized_provider(
    session, history_window_rows=2040, max_request_context=262160,
    query_graphs=True, request_projection_cache=True,
)
prefill_provider = TensorFoldPrefillProvider(
    session.weights, session.torch, capacity=262160,
    expert_backend=expert, kv_dtype='int4', index_ring_rows=2048,
)
```

Use the existing native adaptive engine and close/release each component through
its own ownership API. `SharedGraphTarget` owns multiple inner targets: release
its leased states before closing the wrapper. No upstream monkey patch,
checkpoint download or model reload is needed. The current wrapper still has a
maximum of two private Graph slots; eight slots require further engine work.

This candidate failed the original short-screen raw-logit L2 gate. It remains
an explicit research option, with no quality/default promotion. One 256K
request, an allocation forecast and eight simultaneously populated requests
are separate validations. See [the report](E55_RESIDENCY_REPORT.md).

Reproduce only on an idle matching resident process through its serialized
queue. Stages are immutable; a correction receives a fresh version/job. Never
queue a duplicate while a job is running. The queue runner verifies source and
external-operator hashes. The archive/export commands and failure provenance
are recorded in the report; archives contain teacher logits, not model weights.
