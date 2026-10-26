# E53: long-context memory admission and serialized adaptive requests

Goal: establish the resource and engine gaps to eight independent 256K requests.
Do not raise the existing88GiB allocator cap, reload weights, clear the CUDA
allocator, run concurrent GPU processes, or compare fixed drafts/other engines.

1. Measure the actual FP8/ring2048 shared eager/prefill/screened-head/draft
   allocations, two independent full-capacity states and their private16-bucket
   Graphs, per-request mirrored2040-row feature windows, and sampled prefill,
   verification and full2040-row draft transient peaks. Weights stay resident.
   Record stage peaks, host available and allocation recovery. Two states with
   only a short populated prefix are not two populated256K requests.
2. Compute1..8 slot projections with an explicit transient reserve. Require
   both the process cap and conservative host reserve. Do not sum allocator
   slack and host available. Forecasts are admission estimates, not guarantees.
3. If two slots fit, enable an explicit serialized post-draft adapter over
   shared Graph scratch. Preserve single-request defaults and the unmeasured
   batched-cost rejection. Validate ownership, budgets, tokens, full committed
   state, cancellation/reuse against isolated adaptive execution, then independently
   populate two261120-token requests. No GPU batching or eight-way claim.

Report actual measured peaks and output/state checks separately from throughput.
FP8-vs-BF16 quality is not promoted by same-format equivalence. The frozen E29
per-request cost curve is not a calibrated multi-request throughput objective.
All failures remain in the immutable queue/evidence; an independent audit must
prove resident identities, original operators, and exact allocation recovery.
