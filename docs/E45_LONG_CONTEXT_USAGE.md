# E45 opt-in 4K integration

This page records the E45 configuration and its original validation boundary.
For the subsequent direct256K run and updated physical capacity, see
[E46](E46_256K_REPORT.md).

This builds on the already prepared [resident workflow](RESIDENT_RESEARCH_WORKER.md)
and [E44 expert binary](E40_PREFILL_EXECUTOR_USAGE.md#optional-mia-shared-input-expert-tile-e44).
It borrows loaded weights and the pinned external operators. It is not a standalone
model loader or a declaration of 1M support. See the [report](E45_LONG_CONTEXT_REPORT.md).

```python
from spark_infer_q.kaifra_optimized import optimized_target, optimized_provider
from spark_infer_q.prefill import PrefillExecutor
from spark_infer_q.prefill_tensorfold import TensorFoldPrefillProvider
from spark_infer_q.prefill_experts import PrefillExpertTile
from spark_infer_q.kaifra_speculative import speculative_engine

# Use your already prepared resident session/native Scheduler context.
expert = PrefillExpertTile.from_file(
    expert_binary_path,
    expected_sha256="a5b15bc9ad18e7aea4008f3aba0210856b9b4a4bd764580cbd987b64659b0977",
)
target = optimized_target(session, capacity=4608)
prefill = PrefillExecutor(
    TensorFoldPrefillProvider(session.weights, session.torch,
                             capacity=4608, expert_backend=expert),
    chunk_rows=1024,
)
try:
    target.warmup()
    provider, profile = optimized_provider(
        session, history_window_rows=2040, max_request_context=4608,
    )
    with speculative_engine(
        native, target, provider, eos_ids=session.eos,
        prior=session.calibration["conditional_prior"], mode="adaptive",
        capacity=4352, max_requests=1, token_budget=1024,
        tokenizer=session.tokenizer, prefill_executor=prefill,
    ) as engine:
        # Validated pilot: at most4096 input tokens and64 output tokens.
        engine.submit_tokens("example", prompt_ids, max_tokens=64)
        for event in engine.drain(max_steps=128):
            consume(event)
finally:
    prefill.close()
    target.close()
```

Enter the native context with `max_drafts=7`, keep session inference mode active,
and preserve the pinned E17/E29/W8 artifacts. Prefix + output must fit the logical
capacity. The bounded feature option allocates99.61MiB even for a short request;
the ordinary default remains full history. Do not set `max_requests>1` for this
adaptive DFlash bridge: E45's dual check is a separate target-only isolation check.

The drafter uses an absolute-position2040-row suffix. The target retains complete
BF16 KV/GDN/QSA/PLE history; this is neither KV compression nor a bounded-memory
target. E17/E29 are frozen controls beyond2K, with no long-context recalibration.
8K/32K/1M, paged KV, sampling and arbitrary natural-document quality remain unvalidated.
