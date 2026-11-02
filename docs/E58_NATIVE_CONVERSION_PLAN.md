# E58: native conversion identity and mixed INT4/FP8 after BF16 prefill

2026-10-28. Same resident worker,88GiB allocator cap,8GiB host reserve; no
reload/download/empty_cache/parallel GPU process. Existing defaults and failed
E55–E57 quality decisions remain unchanged. Serialize through the existing queue.

Add an explicit BF16-row converter invoking the same native JIT quant_groups_4
or FP8 store_kv used inside prep. First compare native conversion versus actual
BF16 and INT4/FP8 prep on identical projections at0/1/2047/261120 positions and
1/3/8/1024 rows. Require byte/scale equality, offset guard preservation and
changed-input codec Graph replay. Also compare Torch reference packing, but
preserve disagreements instead of assuming reference/native equivalence.

For the six E57 cases (4K/8K/16K), do BF16 prefill once per case, not once per
format. Preserve its prefix; after baseline generation restore its compact seed
and clone a private state for conversion/decode. Account for the source clone
in peak checks. Do not copy or reload weights. This short-state clone strategy
is not an admission plan for256K or eight-way prefill.

Compare bf16, deferred_torch4, deferred_native4, mixed2 (FP8 on zero-based layers
43/3; INT4 elsewhere), mixed4 (FP8 on43/3/15/47), and deferred_fp8. Masks come from
E56's frozen short calibration ranking, not the current outputs. Convert each
layer once; record all actual formats and real unique state storage. Verify
non-KV state unchanged and no old Graph reused. On real BF16 prefix rows compare
native versus Torch codes/scales for each INT4/FP8 cache; save discrepancy counts.

Use matched BF16 teacher tokens up to32 positions and complete64/128 greedy
outputs. Exclude each uncompressed prefill logit from the main decode metrics.
Maintain KL<=.02, raw L2<=.03, top1>=.95, no lost BF16 objectives. Record E57
baseline/torch-control reproduction where exact inputs match. Save CPU BF16
teacher artifacts for subsequent studies so reference prefill need not repeat.

Choose passing non-BF16 candidate by actual state storage, then KL, then name.
No default/quality/Graph/speculation/256K/eight-way promotion from this bounded
screen. Capacity forecasts replace only measured KV-format storage relative to
E55, preserving its shared area/reserve; additional conversion/prefill admission
peaks remain separate. If none passes, retain negative evidence and do not
implement an unqualified serving option.
