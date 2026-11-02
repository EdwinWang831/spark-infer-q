# E44: adapt Mia's complete shared-input expert tile

2026-10-24. E43's single-tile instruction scheduling was exact but failed its
component cost gate. Preserve that negative result. This is a distinct candidate:
adapt the pinned MIT TensorFold/Mia `fn_experts_tile.cu` (Mia repo commit
656879fd752749bbd11a6e1a065a326c00940911) into our owned prefill provider.
Use its existing default policy: plan64, gate/up RT4/S4, down RT2/S3, four warps
handling different32-column output blocks and sharing staged input fragments.
Enable at rows>=1024; shorter calls use the original MoE. No configuration sweep.

Unlike E38/E39, warps share X across output columns rather than sharing the same
weight block across row groups. Keep original BF16 activations, NVFP4 per-K16
scale-FMA order, F32 down and identical epilogue/slot order. Dynamic shared memory
and resources must be archived. No upstream global dispatch or weight mutation.

Use E43's boundary/stream/Graph protocol, adding rows48/63/65 around the plan64
boundary. Complete-MoE real inputs: English64, code512, code1024, mixed1024,
layers0/23/47; all buffers/router/picks/weights exact. Warm/128MiB-perturbed cost
screen: six AB/BA repetitions. Primary adoption gate applies to the six eligible
1024-row cases: summed per-case median cost >=5% lower, >=5/6 repetition wins,
no eligible case >2% slower. Retain shorter calls as same-path controls and report
all12 cases separately; do not let noisy unchanged controls select the candidate.

Only on that gate pass, run complete owned chunk1024 prompt and full optimized
Graph/E33/W8/E17/E29 requests, with same budget1024/capacity2048/logical1792,
greedy cap64, warm then four AB/BA pairs. Require full logits/features/valid
state and partial-commit handoff, output/position/decode-trace equality, no new
capture/JIT/fallback during measured requests. Report TTFT/whole wall/residual
decode separately, plus cancel/reuse and loopback SSE functional smoke.

Same current resident process throughout; independent cleanup audit, no reload,
cache clearing, concurrent GPU work, default promotion or claimed Mia speed lead.
The complete kernel adaptation carries MIT attribution in THIRD_PARTY_NOTICES.
