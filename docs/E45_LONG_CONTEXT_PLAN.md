# E45: bounded draft history and the first complete4K gate

2026-10-25. Long-term objective: each request1M, multiple concurrent requests.
Current checkpoint max_position_embeddings262144; this pilot does not extend
RoPE/model quality limits or claim1M support. Current physical4608/logical4352,
prompt chunks1024, output cap64, BF16 KV, E44 opt-in expert tile.

Implement a per-request mirrored feature ring (capacity2040, two copies for a
contiguous zero-copy suffix) and a positioned rolling draft projection cache.
Only accepted target rows enter it. Target KV/GDN/PLE/QSA history stays complete.
Absolute context_start + retained_rows equals committed target position; predictor
and frozen cost controller receive the full global position. No new fitting.
The drafter sees <=2048-block8=2040 suffix rows, with absolute RoPE positions:
this conservative proposal variant is not asserted identical to SGLang's full
query-specific sliding mask. Final greedy target outputs must remain exact.

1. CPU ownership/shape/ring/rewind/absolute-position checks, including a logical
   1M-row bounded-storage simulation (not a1M model run).
2. Resident pilot, same weights/no reload: full logits/features/valid-state
   comparison of native rows8 target versus owned prefill on boundary2049,
   mixed3073, code4096 (plus short controls). QSA must be active for4K scratch.
   Graph widths1..8/full and partial commits, actual replay without fallback.
3. Real target features: ring suffix equals full-history suffix at2040/wrap/tail
   boundaries; rolling projected keys/values, hidden/proposals/signals compared
   to a full-rebuild absolute-position suffix reference. Stop on mismatch and
   preserve failed evidence before changing policy. No numeric tolerance waiver.
4. Only after pilot passes: native full optimized single-request generation at
   <=4K with bounded ring versus full-history/same-suffix reference, warmed AB/BA,
   equal inputs/output work/traces, complete target-only greedy oracle, explicit
   history bytes/start/end, TTFT and decode timings. Old E17/E29 beyond2K are
   frozen engineering controls, not calibrated long-context optimization claims.
5. Lifecycle/cancel/reuse and loopback SSE smoke; two admitted long target-only
   requests with interleaved state checks, explicitly not batched DFlash or a
   concurrency-throughput win. K4 batching remains the other workspace's work.
6. Independent post-return audit against the current stable resident baseline:
   source/no_grad wrapper, same current worker/objects/load_count1, group/env,
   removed study modules and no owned allocation growth. No empty_cache/reload.

Physical constructor policy may allow up to8192 for future experiments; actual
validated input bound remains only what this report measures. No paged/quantized
KV,8K/32K/1M model, multi-request DFlash or new checkpoint is implemented here.
