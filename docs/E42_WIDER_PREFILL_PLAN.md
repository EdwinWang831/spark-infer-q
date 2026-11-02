# E42: 1,024-row owned prefill screen

2026-10-24. Continue E41 with the same resident weights and external source
seal. One variable: owned chunk_rows=512 versus an experimental provider subclass
allowing chunk_rows=1024. No production bound/default change before evidence.
Keep physical capacity2048, logical1792 and (in request tests) scheduler budget1024
on BOTH arms. Decode, E17/E29 costs, Graph/E33/W8 and greedy output cap64 stay fixed.
No kernel, precision, model or predictor change; no 4K or TensorFold ranking claim.

1. Numeric first: English64, code513, code1024, mixed1025, Chinese1536. Compare
   full last-row logits, all five-layer prompt features, valid GDN/QSA/PLE state.
   Then widths1..8, odd widths full commit and even widths half commit, with
   actual Graph replays. Cover both parity buckets for each width. Stop before
   timings on any mismatch; record mismatch counts/max absolute differences.
2. Warm both arms, then six alternating paired prompt timings on code1024 and
   mixed1536 and four on English64. Include fresh state, stage, layers, features,
   head, commit, completion fence; exclude provider/workspace setup and checks.
   Require exact outputs/state for every measured run. Performance gate: long
   prompt summed wall at least3% lower, at least4/6 repetition aggregates lower,
   neither long case >2% slower, short input <=3% slower. Stop if gate fails.
3. Only after both gates: matched full optimized requests on natural English and
   Chinese plus controlled code1024/mixed1536, one warm per arm then four AB/BA
   pairs. Require equal tokens, positions, complete committed features/state and
   decode acceptance/width/rate/head traces, without capture/JIT/fallback during
   measurement. Compare prompt plan trace separately: wider chunks intentionally
   change internal prompt segmentation. Report TTFT, whole token-API wall and
   request-minus-native-TTFT residual decode separately, with no assumed gain.
4. If requests execute, cancel/reuse and loopback SSE functional checks follow.
   Audit independently after return, including failed screens: same resident
   objects/load_count1, sealed operator code/no_grad wrapper, native group/env
   cleanup, removed stage modules and allocated bytes <= captured baseline.

No cache clearing, reload, concurrent GPU job or upstream dispatch mutation.
Raw failures and source hashes remain archived. A failed numerical screen is a
valid negative result; do not relax exactness or vary another knob to hide it.
