# E55: reduce actual resident bytes toward eight256K requests

2026-10-27. Keep the E54 request cache and frozen predictor/costs as the control.
First inventory unique CUDA storages, including aliases across the loaded target,
legacy target buffers, draft original/fused weights, projection and optimized
provider. Report exclusive storage, not sums of aliased tensor views. Do not
modify or free session-owned weights during the inventory.

Select a measurable reduction from the inventory. Lossless removal/reuse of
unused owned buffers takes precedence when it can contribute meaningfully. If
the eight-way deficit cannot be addressed with those allocations, explicitly
measure the low-bit KV candidate and require separate arithmetic/quality gates;
an INT4 allocation is not a quality or concurrency result.

Inventory-v2 confirms target unique storage75002047104 bytes. Removing all
non-target roots cannot put eight FP8 states under88GiB. Extend the explicit
bounded-index adapter to existing H32/group32/FP16-scale INT4; keep defaults and
all quantizer/dequantizer arithmetic unchanged. Differential kernel checks must
cover BF16/FP8 regression plus INT4 wrap/odd starts/near256K, all scales,
pool/selectors and changed-position Graph replay. Then require full-model
same-format eager/prefill/Graph/partial-commit equivalence before quality work.

Use E47's existing eight bilingual/objective cases, BF16/FP8/INT4, matched BF16
teacher tokens and <=64 greedy outputs. Preserve original E47 thresholds:
no lost BF16-passed objectives, mean KL<=.02, raw relative L2<=.03, teacher
top1>=.95. Failures do not promote a cache format. Add an actual single256K
INT4 dynamic request and a matched BF16 teacher-forced comparison as an
explicit research diagnostic even if this conservative short gate fails;
never describe it as accepted quality or eight-request execution. Keep native
dynamic budgets and request-owned draft projections, no fixed-draft arm.

Use the resident serialized queue, unchanged88GiB allocator cap and8GiB host
reserve. No reload, downloads, empty_cache, parallel GPU processes or changes to
the separate K4 worktree. Seal sources, preserve failures and independently
audit original operators, resident identities and exact allocation recovery.
Do not infer8-way execution,1M quality or serving speed from a storage forecast.
