# E41: PrefillExecutor with Graph/E33/W8 and longer input

2026-10-24. Continue E40 on the same resident weights. No new kernels, default
promotion, model reload or 4K claim. Test the current committed close guard too.

1. Full-logit Graph handoff: compare rows8 vs owned 512-row prefill on English64,
   code512, code513, mixed1024 and Chinese1536. Require full last logits, all five
   feature rows and valid state to match. Follow with widths1..8 and alternating
   full/partial commits, covering all width/parity buckets without fallback.
2. Full optimized request configuration: Graph + E33 original-BF16 decision head,
   W8 draft MLP/Q4 draft head, E17 NumPy and frozen E29 W8 costs. Compare only
   prefill mode; both use logical capacity1792, physical2048, budget512, greedy
   output cap64. Natural English/Chinese, controlled code513 and mixed1536.
3. Warm each arm per case, then four AB/BA repetitions: 8 warm +32 timed requests.
   Compare output/position/acceptance/width/rate traces, full committed feature
   context and final valid state outside timing. No new Graph capture or screened
   head JIT shapes inside measured requests. Report TTFT, decode and complete
   token-API wall separately; not an HTTP or native-TensorFold ranking.
4. Native waiting/prepared-prompt/prepared-decode cancellation and ID reuse;
   loopback SSE stream/cancel smoke tests (functional only, no HTTP speed claim).
5. Independent post-return audit: pinned sources/code, original no_grad wrapper,
   same worker/models, native group/environment cleanup, no owned stage modules,
   and allocated bytes no higher than pre-study baseline. Do not clear caches or
   mutate other workspaces to meet this condition.

Stop on numerical/state/JIT/lifecycle failure, preserve the partial report and
diagnose. A faster rows8 comparison is not an incremental CUDA-kernel gain.
