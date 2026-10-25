# E51: preserve FP8 with a bounded raw QSA index

Use the existing serialized resident queue (PID46235, load_count1), no reload,
allocator-limit increase, cache clearing, concurrent GPU experiment or defaults
promotion. Fixed target/weight arithmetic, E17/E29 dynamic budget and E50 draft
query Graph/output cache. No cross-engine or fixed-draft comparison.

The explicit 2048-row ring retains the new prepare (at most1024) and prior
unfinished ratio4 block. Full KV and full BF16 pooled index remain. Existing
TensorFold prep/pool JIT bodies receive translated raw pointers; absolute RoPE,
KV positions, pooled addresses and downstream operators are unchanged.

Gates:
- GPU prep/pool equivalence for BF16 and FP8, prompt/decode windows, wraps,
  odd starts and positions near256K; exact queries, codes/scales, index queries,
  current raw rows, pools, scores and selector IDs. Capture/replay at a new
  absolute position proves the offset is device-dependent.
- Actual complete-state allocation delta, fixed capacity, snapshot/restore
  unfinished tails, no unsupported native MTP or prefix migration.
- Sequential full-prefix baseline/ring native adaptive requests in each KV
  format,261120 inputs/512 outputs. Compare all output IDs, budget/acceptance
  traces, complete committed KV/pooled/GDN/PLE, live raw tail, features and
  selected keys. Warmup and JIT before timing, no timed captures/fallbacks.
- Short cancellation/reuse and post-return original-operator/resident/memory
  audit. Report decode and full wall separately; capacity is not an8-way run.

Do not call same-format FP8 equivalence BF16 quality parity. Removing discarded
raw history changes the equivalence contract; hash all semantically live state
instead. Measure retained-state memory, not reserved bytes or hardware capacity.
Do not promise speed from a memory-saving change; retain negative results.

After the first full FP8 pair unexpectedly showed27.09→35.15 tok/s, add a
discriminating repeat before claiming a stable gain: prepare one real FP8
prefix, let full/ring Graph owners share exactly the same K/V/pooled addresses,
eager buffers and screened head, retaining independent raw index/recurrent
states. Warm both before prefill, then four512-token AB/BA seeded decode pairs
and16-round kernel profiles. Reset only small recurrent/PLE/raw-tail snapshots;
never clone/reload weights or full context. Compare each output/live-state trace
against the full FP8 request. This excludes native Scheduler/event work and
does not replace the full-request pair; it isolates raw-ring/dispatch from KV
allocation address changes and checks order effects.
