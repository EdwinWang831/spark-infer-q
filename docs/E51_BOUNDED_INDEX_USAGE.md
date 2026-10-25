# E51 bounded raw index (explicit research option)

Use the prepared resident environment and pinned TensorFold installation. The
option defaults to `None`; BF16 and FP8 are supported, text-only, native target
MTP absent, ratio4, maximum prepare1024. KV capacity remains262160, logical
prompt-plus-output capacity262144. This option does not add request slots.

```python
from spark_infer_q.shared_graph import SharedGraphTarget
from spark_infer_q.prefill import PrefillExecutor
from spark_infer_q.prefill_tensorfold import TensorFoldPrefillProvider
from spark_infer_q.kaifra_optimized import optimized_provider
from spark_infer_q.kaifra_speculative import speculative_engine

target = SharedGraphTarget.from_session(
    session, capacity=262160, kv_dtype="fp8", slots=1,
    index_ring_rows=2048,
)
prefill = PrefillExecutor(
    TensorFoldPrefillProvider(
        session.weights, session.torch, capacity=262160,
        expert_backend=experts, kv_dtype="fp8", index_ring_rows=2048,
    ),
    chunk_rows=1024,
)
provider, _ = optimized_provider(
    session, history_window_rows=2040, max_request_context=262160,
    query_graphs=True,
)
try:
    target.warmup()
    provider.worker.warmup()
    # The prepared native Scheduler context uses NEXTN=7. Prepare all target
    # head subset shapes before timing, as in scripts/e51_study.py.
    with speculative_engine(
        native, target, provider, eos_ids=session.eos,
        prior=session.calibration["conditional_prior"], mode="adaptive",
        capacity=262144, max_requests=1, token_budget=1024,
        tokenizer=session.tokenizer, prefill_executor=prefill,
        output_cache=True,
    ) as engine:
        engine.submit_tokens("request", prompt_ids, max_tokens=512)
        for event in engine.drain(max_steps=1200):
            consume(event)
finally:
    prefill.close()
    target.close()
    provider.worker.close()
```

Target and prefill must use the same KV format and ring layout. A mismatch is
rejected before launching the prompt. Existing formats/defaults remain intact.

Each request retains complete K/V history and BF16 pooled QSA keys. Only raw
index keys are circular:2048 rows per attention layer. Prep translates each
raw-store pointer using the device absolute position; pooling translates each
complete four-row block. Both call the original numerical JIT bodies; no
upstream global function, model weight or position is changed. This is a memory
layout adaptation, not a new quantization algorithm.

Snapshots save the prior unfinished raw block (zero to three rows per layer),
in addition to recurrent/PLE state. Copying requires matching ring geometry;
fixed-capacity resizing and prefix migration are rejected. State equality must
compare full committed KV and pools plus the unfinished raw tail, not discarded
raw history. `LongTarget.state_difference` follows that contract for ring states.

Eight-state memory figures are forecasts, not eight admitted requests. The
shared long-Graph wrapper still exposes one or two serialized slots and the
adaptive validation uses one. No multi-request DFlash/HTTP, paging or1M support
is introduced. FP8 same-format equality does not establish BF16-relative
quality; compressed-KV quality remains a separate gate.
