# Two serialized long-context adaptive requests (research opt-in)

**The E53 candidate fails the strict dynamic-trace gate.** Two populated256K
requests reproduce isolated outputs and full committed state, but request A
changes7 verification widths and144 predictor-rate rounds. Retain this adapter
for diagnosis; it is not promoted as a validated multi-request dynamic policy.
[E54](E54_REQUEST_DRAFT_CACHE_USAGE.md) supplies the explicit per-request cache
option that passes this strict two-request gate; E53's original failure remains.

E53 retains the existing private Graph slots and adds an explicit
`serialized_postdraft=True` flag. The existing default still rejects adaptive
multi-request use. This flag requires the shared serial scratch target, enough
private request slots and a deferred drafter. It does not enable GPU batching
or a calibrated cross-request cost objective. Maximum long Graph slots remains2.

Before creating/capturing the target, use a fresh memory snapshot and a measured
`LongCapacityBudget`. E53 uses73.822GiB common,3.606GiB/private slot and1GiB
sampled-overhead reserve, with the unchanged88GiB allocator cap and8GiB host
reserve. These numbers describe this resident checkpoint/runtime only; measure
again for other models or configurations. Do not add CUDA reserved slack to
Linux `MemAvailable` as if they were independent RAM pools.

The creation and warmup pattern is the same as [E51](E51_BOUNDED_INDEX_USAGE.md),
with these changes:

```python
target = SharedGraphTarget.from_session(
    session, capacity=262160, kv_dtype="fp8", slots=2, index_ring_rows=2048,
)
target.warmup()
# Prepare the same rolling provider, chunk1024 prefill and native NEXTN=7.
with speculative_engine(
    native, target, provider, eos_ids=session.eos,
    prior=session.calibration["conditional_prior"], mode="adaptive",
    capacity=262144, max_requests=2, token_budget=1024,
    tokenizer=session.tokenizer, prefill_executor=prefill,
    output_cache=True, serialized_postdraft=True,
) as engine:
    a = engine.submit_tokens("a", prompt_a, max_tokens=512)
    b = engine.submit_tokens("b", prompt_b, max_tokens=512)
    for event in engine.drain(max_steps=1600):
        consume(event)
```

Each request owns its target KV/recurrent/PLE/QSA state, mirrored feature suffix,
acceptance history, current proposals and chosen width. The shared drafter
invalidates its projection cache on request switches and reconstructs it from
that request's committed suffix. Graph scratch, screened head, prompt workspace
and draft weights remain shared and execute serially. Close/release use device
fences before slot reuse.

Full history is retained in the target; only the drafter uses the2040-row
suffix. FP8 precision remains experimental: same-format exactness does not
establish BF16-relative quality. This API has no HTTP/fairness/GPU-batch/8-way
performance guarantee. See [the report](E53_LONG_CONCURRENCY_REPORT.md) for the
actual populated-context validation and evidence boundaries.
