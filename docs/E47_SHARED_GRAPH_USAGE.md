# E47 compressed KV and shared serial Graph slots

Prepared resident environment only. The [E47 report](E47_KV_DUAL_REPORT.md)
records capacity, precision and scheduling limits. The original BF16-relative
raw-logit precision gate failed; the INT8 option remains a research candidate.

```python
from spark_infer_q.shared_graph import SharedGraphTarget, shared_graph_engine
from spark_infer_q.prefill import PrefillExecutor
from spark_infer_q.prefill_tensorfold import TensorFoldPrefillProvider

# Existing loaded session, pinned expert backend, native NEXTN=0 context.
target = SharedGraphTarget.from_session(
    session, capacity=262160, kv_dtype="int8", slots=2,
)
prefill = PrefillExecutor(
    TensorFoldPrefillProvider(
        session.weights, session.torch, capacity=262160,
        expert_backend=experts, kv_dtype="int8",
    ),
    chunk_rows=1024,
)
try:
    target.warmup()
    with shared_graph_engine(
        native, target, eos_ids=session.eos, capacity=262144,
        token_budget=1024, tokenizer=session.tokenizer,
        prefill_executor=prefill,
    ) as engine:
        engine.submit_tokens("a", prompt_a_ids, max_tokens=64)
        engine.submit_tokens("b", prompt_b_ids, max_tokens=64)
        for event in engine.drain(max_steps=1200):
            consume(event)
finally:
    prefill.close()
    target.close()
```

The measured inputs each have262080 tokens. Logical prompt + requested output
must fit262144. Each request owns its KV, scale, QSA and GDN/PLE state and its
Graph captures; both slots share the eager Buffers, head and one prefill scratch.
All device execution is serialized, with a fence before shared scratch reuse.
`gpu_batch=False`, target-only greedy, no adaptive multi-request DFlash or HTTP.
The native scheduler retains its prompt priority; see the measured output gaps.

`LongTarget(..., kv_dtype="int8")`, `optimized_target(..., kv_dtype="int8")` and
`TensorFoldPrefillProvider(..., kv_dtype="int8")` expose the same explicit format.
The supported study formats are BF16, INT8 and FP8; target and prefill format must
match. INT8 scale rows are part of state equality and prefix hashing. QSA index
keys/pools remain BF16 and GDN recurrent banks remain FP32. Existing target and
prefill APIs default to BF16; the new shared-slot factory defaults to INT8.

The underlying KV packing, H32 quantization, FP8 conversion, QSA and attention
kernels remain pinned TensorFold dependencies. This module adds request/cache
ownership and scratch reuse; it does not introduce a new quantization algorithm.
