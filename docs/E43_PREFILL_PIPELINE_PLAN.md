# E43: single-tile expert instruction pipeline

2026-10-24. Learn the MMA/FMA software-pipeline idea from MiaAI-Lab's pinned
`fn_experts_tile.cu`, without porting its Zig runtime or re-running E38/E39's
row/CTA weight-reuse design. One candidate: within each original K32 group,
issue the next independent fresh-zero MMA before consuming the previous result
with scale FMA. Preserve each output's (group,half) FMA order, tile16, original
plan/grid/4-warps/staging-depth2, BF16 activations, NVFP4 weights and F32 down.
No global dispatch patch, weight mutation, new precision, reload or service stop.

The currently resident worker differs from E42's historical process. Capture its
current identity/baseline, and reuse it throughout E43; do not require equality
with old E38/E42 object IDs. Shared GPU queue stays serial.

1. Seal/compile one isolated SM121 extension; preserve source transformation,
   pinned source/header and binary hashes plus ptxas resources. No parameter sweep.
2. Boundary checks: rows1/15/16/17/31/32/33/64/128/512/1024, contiguous/strided X,
   BF16 and F32 down, guards, repeated Graph replay and a separate caller stream.
   Require exact complete gate/up and down outputs versus existing operators.
3. Real complete-MoE screen on four inputs English64/code512/code1024/mixed1024,
   layers0/23/47. Capture actual provider inputs without globally replacing
   operators. Compare router logits/pick/wts/act/y exactly. Six AB/BA repetitions
   in warm and128MiB cache-perturbed regimes; timed scope includes router/select/
   plan, routed and shared experts and output slots. Perturbation/checks outside.
   Gate: perturbed summed per-case median cost at least5% lower, >=5/6 aggregate
   repetition wins, no real case >2% slower. Short-control uses original path.
   Stop before full prompt/request timing if this gate fails.
4. Only on cost pass: owned chunk1024 complete prompt and Graph/E33/W8 request
   comparisons with equal native budget1024, physical2048/logical1792, E17/E29,
   output cap64, one warm and four AB/BA pairs. Full logits/features/valid state,
   partial-commit handoff, tokens and decode traces must match. Report TTFT,
   whole-request wall and residual decode separately. HTTP/cancel smoke if promoted.
5. Independent post-return audit: same current worker/objects, source/no_grad
   wrapper identity, native group/env restoration, no retained study modules or
   observers and allocated bytes <= captured baseline; no empty_cache/reload.

Raw failures remain evidence. A component gain is not a TTFT or serving gain;
there is no fresh matched Mia engine ranking or4K/quality/general-traffic claim.
