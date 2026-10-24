# Owned PrefillExecutor

The engine now has a reusable prompt-execution module instead of requiring the
E36 experimental private-forward clone. `spark_infer_q.prefill.PrefillExecutor`
owns chunking, a serial workspace, output ownership and failure quarantine.
`TensorFoldPrefillProvider` supplies E36 arithmetic using external operators.

The provider uses shallow weight views: MXFP8 prompt projections call the same
matrix's decode arithmetic. GDN uses a full-chunk scratch and the decode chain;
FP32 MoE output slots retain the existing shared-expert arithmetic. Original
weight objects and upstream module functions are not reassigned. Layer order
and GDN state transitions derive from TensorFold; see
[third-party notice](../THIRD_PARTY_NOTICES.md).

## Existing native runtime integration

With a loaded target and a compatible `NativeScheduler` context:

```python
from spark_infer_q.prefill import PrefillExecutor
from spark_infer_q.prefill_tensorfold import TensorFoldPrefillProvider
from spark_infer_q.kaifra_runtime import KaifraEngine

provider = TensorFoldPrefillProvider(target.w, target.torch,
                                   capacity=target.capacity)
with PrefillExecutor(provider, chunk_rows=512) as prefill:
    # Target-only: native must use max_drafts=0.
    with KaifraEngine.from_native(
        native, target, eos_ids=eos_ids, capacity=768,
        max_requests=1, token_budget=512, prefill_executor=prefill,
    ) as engine:
        engine.submit_tokens('request', prompt_ids, max_tokens=24)
        events = list(engine.drain())
```

For DFlash use `speculative_engine(..., prefill_executor=prefill)` and the
existing provider/budget setup, with a native `max_drafts=7` context. Target-only
and speculative engines require separate native contexts. The scheduler's
`token_budget` limits the prompt rows it sends to the executor: leaving it at
32 cannot produce a 512-row prompt batch. Larger budgets also change scheduling
fairness; compare both arms with the same budget.

The engine owns request states; the caller owns the PrefillExecutor and closes
it after its engines have finished. One workspace can serve independent states
serially. Concurrent/reentrant use and switching CUDA streams are rejected.
Returned full-vocabulary **last-row** logits and full five-layer DFlash features
remain valid after workspace reuse. Per-row prompt diagnostic observers require
the old rows8 path and are explicitly incompatible with this option.

On computation or completion-fence failure, do not retry the changed request
state. The executor is quarantined; retire the request through the runtime's
error path. close() fences before releasing workspace references and retains
them if that fence fails. No rollback of partially executed prompt layers is
promised.

## Scope

Initial provider: one GPU, the 48-layer NVFP4 Qwen target, BF16 KV, text only,
eager prompt execution, up to 512 rows per chunk and capacity up to 2048. The
E40 study uses capacity 1024, prefixes up to 512 and a 129-row continuation test.
Neither the constructor's capacity bound nor multi-chunk support establishes
4K+, vision, random sampling, cross-request GPU batching or HTTP performance.
The historical E40 request performance study uses the eager target and existing
K2 adaptive provider. [E41](E41_PREFILL_GRAPH_REPORT.md) separately validates
the unchanged module with optimized Graph/E33/W8 decode, prefixes up to 1,536
tokens, partial commits, cancel/reuse and loopback HTTP functional checks. Both
request arms use token_budget=512, physical capacity=2048 and logical capacity=1792.
Prefill stays eager; Graph covers decode. E41 measures lower TTFT and complete
request time against rows8, with no new residual decode gain or HTTP speed claim.

External dependencies remain: weight loading, State/Buffers, input staging,
attention/PLE/HC/MoE/commit and CUDA kernels. The first owned layer is execution
organization, not a new expert kernel or complete independence from TensorFold.
The provider targets the frozen Python operator interfaces used by E33/E36;
the latest MiaAI-Lab Zig engine is a separate source of implementation ideas,
not a drop-in dependency for this adapter.
