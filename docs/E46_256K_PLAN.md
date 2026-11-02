# E46: direct256K capacity and execution gate

User explicitly requests proceeding directly to256K. No staged8K/32K milestone.
Use existing resident weights, shared serial queue and unchanged BF16 target KV.
Physical262160 reserves native lookahead, logical262144, prompt262080, output64.
Committed positions stay within checkpoint max_position_embeddings262144.

Avoid full-KV Graph warmup cloning above8192: use the existing non-cache State
snapshot, preserving the one committed KV row while replaying only later rows.
Keep source/operator fingerprints, old bounded draft2040 and frozen E17/E29.

One actual native single-request prefill and dynamic decode, with per1024-chunk
wall/position/memory reporting. Insert diagnostic pause after first token:
check the real >128K tiled QSA selector against CPU stable selection of its
scores; all16width/parity Graph buckets with full logits/features, kept-state
and tail-cache exactness; target-only greedy continuation versus dynamic output.
Check base-prefix cache digests unchanged after all diagnostic/decoding work.
No full256K oracle prefill rerun or universal long-document quality claim.
Use a seeded bilingual synthetic document with early/middle/late retrieval keys,
report retrieval result separately from execution/state correctness.

The active prefill/decode times exclude inserted diagnostics; this is a capacity
and correctness run, not an uninterrupted end-to-end serving benchmark or a
multi-request/1M claim. Source-sealed separate post-return audit must recover
the current resident baseline without unloading models/clearing shared caches.
Abort cleanly before resource exhaustion if host available memory falls below
8GiB. Keep all failed jobs and progress rather than substituting shorter work.
