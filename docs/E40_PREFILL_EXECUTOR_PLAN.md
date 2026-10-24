# E40: owned prefill executor and request integration

Goal: turn E36 A/B/C into a reusable engine module with an injectable operator
provider, no cloned/executed upstream source and no global dispatch mutation.
This does not remove the external loader, State/Buffers or CUDA operators.

Initial boundary: one GPU, NVFP4 target, BF16 KV, text-only greedy, chunk size
at most 512, physical capacity at most 2048. Longer-context promotion is separate.
Caller owns request state. Executor owns one serial workspace and fences its
release; failures quarantine the executor and require retiring affected state.
Full five-layer DFlash features and final-row logits are owned outputs.

Use opt-in integration in both target-only and speculative backends. Per-row
diagnostic observers keep their existing contract and cannot be combined with
the final-row-only executor. Existing defaults remain rows8 pending validation.

Required validation before speed interpretation:

- CPU tests: preflight, chunk tails, ownership, stream mismatch, concurrent use,
  partial failure, fence failure and backend routing (not numeric evidence).
- Same resident model: seven E36 prefixes, rows8 vs provider logits, all feature
  rows, effective states, eight subsequent original decode steps per prefix.
- Forced multi-chunk continuation (including a short tail and nonzero start),
  workspace reuse, capture disabled, release and independent requests.
- Native request path: target-only and dynamic DFlash, output/position/acceptance
  traces, cancellation/reuse. No HTTP claim unless measured through HTTP.
- Only after correctness: matched warm-up, AB/BA timings of prompt and complete
  token API requests with identical budgets, tokens and timing boundaries.
- Final audit: source/functions unchanged, same resident objects/load count,
  native scheduler released and allocated memory returns to baseline.

No upstream benchmark ratio is a target prediction. Stop and diagnose numerical
or lifecycle failures; do not reload weights or mutate frozen experiments.
