# E47: compressed KV and two resident256K requests

User authorized KV compression and multi-request memory reuse after E46.
Keep weights resident, original88GiB allocation cap, shared serial queue, no
downloads, source patching, empty_cache, model unload or other-workspace edits.

1. Reuse pinned TensorFold INT8 (H32,32-element absmax FP16 scales) and FP8
   (e4m3,row scales) KV. QSA index/pools and GDN remain BF16/FP32 as before.
   Explicit format match in prefill and target; BF16 defaults stay unchanged.
   Compare complete numeric/prefill and Graph paths within each compressed
   format including scale storage. Cross-format BF16 equivalence is not expected.
2. Eight prespecified English/Chinese/structured/arithmetic/code/retrieval inputs,
   same-prefix teacher forcing at up to8 output positions, full-logit KL/L2/top1,
   plus independent greedy answers. Pilot gate: no lost objective fixture passed
   by BF16, mean KL<=0.02 nats, mean relative-logit L2<=0.03, teacher-forced top1
   agreement>=0.95. Prefer INT8 if it passes; FP8 is the fallback if it passes.
   This bounded engineering gate is not general long-document quality evidence.
3. Two stable Graph state slots sharing one eager buffer set, screened head and
   one owned prefill workspace; per-slot captures, serialized GPU dispatch.
   Do not duplicate K4 joint/batched DFlash or remove its admission restrictions.
4. With selected format, admit two different262080-token inputs together under
   native Scheduler, prefill/decode interleaving, max64 outputs each. Distinct
   retrieval keys must be preserved. Per-request target-only continuation oracle,
   unchanged committed-prefix digest, long QSA and Graph checks, cancellation /
   same-slot reuse and full cleanup. Guard8GiB system reserve. Report exact live
   KV/cache bytes and peak allocated, shared/private storage, first-token delays
   and output gaps. This is two-admitted-request capacity/isolation, not joint
   GPU batching, concurrent DFlash or a throughput-lead claim.
5. Independent post-return source/operator/group/environment/allocation audit;
   keep failed jobs and stop on numerical or resource failure without redefining
   shorter work as two256K success. Frozen E17/E29 are not retrained.

Amendment after immutable pilot-v1: neither format passed the original raw-logit
L2<=3% gate (INT8 6.55%, FP8 8.42%), although INT8 greedy outputs matched8/8,
teacher top1 matched54/54 and mean KL was0.000380. Preserve STOP_E47_QUALITY.
Continue INT8 as a capacity-only research candidate, without quality promotion
or changing the original gate. Add one full256K BF16 control after the two-state
run, measuring raw/centered logit drift and probability KL on matched prefixes.
The two-slot run records the actual native scheduling waits; fairness and
interleaving are observations to test, not assumed guarantees.
