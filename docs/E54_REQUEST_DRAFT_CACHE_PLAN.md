# E54: isolate request projections and diagnose E53's dynamic-trace failure

Keep the resident checkpoint and native queue, allocator cap88GiB and host
reserve8GiB. No reload, cache clearing, new weights, predictor fitting, fixed
draft arm, other-engine comparison, or K4 change.

The candidate stores context projections, keys/values and rolling counters per
request; weights, query Graphs/workspace, padding and target scratch stay shared.
Binding restores the worker even on exceptions. Request release fences first,
then clears its cache. Default remains off. The three tensors add50MiB/request
at max_context2048; include this measured increment in admission/eight-way
forecasts. This is a correctness/ownership change, not a memory-saving claim.

Validate CPU ownership/restoration/failure contracts, then run matched real
native requests: English3073/Chinese4097 short controls, combined, cancel/reuse,
and independent261120-token long controls followed by two populated long
requests,512 output tokens each. Require exact outputs, complete effective
target state, feature suffix and all decode rates/widths/acceptance traces.
Also compare current isolated long controls against E53's archived controls.

On isolated long A windows30/31/32/36 (one-based), replay the identical committed
suffix/anchor with a fresh projection cache. Record incremental/rebuild cache,
proposal, rates and width differences without feeding diagnostics to the target
or changing the committed context. These are paired arithmetic probes, not a
new generation arm. They must leave the real request cache unchanged. This
tests the rebuild hypothesis; do not claim it proven in advance or uniquely
explanatory if other differences remain.

All timing includes guards, progress and retirement state hashes; diagnostic
probes are explicitly recorded. No new serving/decode throughput claim. Save
every result before strict assertions, preserve failures, independently audit
resident identity, source/result digests, original operators and exact recovery.
FP8-versus-BF16 quality, HTTP fairness, batching and eight-way capacity remain
outside this gate. Promote only this explicit option if all strict gates pass.
