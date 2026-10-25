# E48: decode opportunities between long prefill chunks

User requested pushing E45–E47 then continuing. Remote branch verifiedc0a99d0.
Keep existing resident weights, INT8 research cache,32 private-state Graph buckets,
shared serial scratch,1024-row prefill, two target-only slots. E47's failed raw
precision gate remains failed. No default KV/policy change, MTP or K4 batch edits.

Add an opt-in single-rank/NEXTN=0/two-slot adapter policy. Use the pinned native
Scheduler's existing prefill polling interval2 and exposed decode_step phase:
when both waiting prefill and runnable decode exist, allow at most one prompt
plan before a decode opportunity. With no runnable decode, continue prefill.
Native C++ still chooses request plans and owns queues, blocks and output stops;
no upstream monkeypatch or manipulation of MTP schedule/process counters.

1. Actual pinned native Scheduler with deterministic host backend: native control
   vs interleave for mixed chunks, unequal lengths, late admission, cancellation,
   EOS, length cap, reuse and block recovery. Assert per-request tokens identical,
   no empty device plans, bounded prompt runs while decode is ready.
2. Real short two-request controls and cancel/reuse on the same shared Graphs.
   Inputs include valid English/Chinese chat and controlled longer prompts.
   Preserve full committed cache/scale/GDN/PLE fingerprints across policy arms.
3. One contemporary full256K matched pair: same two E47 inputs, cap64, common
   preheated target/head, control then interleave. No diagnostics inserted inside
   timed requests. Serialize API submit/step, stream events, post-run fingerprint
   checks. Capture both TTFTs, each output gap, completion times, total wall,
   actual native prompt/decode traces and work, Graph counters and memory.
   This is a bounded engineering comparison, not HTTP, universal throughput or
   adaptive multi-request DFlash evidence. No historical E47 timing relabeling.
4. Require same output tokens/final states, no Graph/head fallback or timed JIT,
   >=90% less A maximum output gap, A completion earlier, and no more than5%
   combined completion/B-first-token regression. Preserve any failed gate.
5. Waiting/prepared prompt/prepared decode cancel, late admission and reuse;
   independent post-return source/operator/group/env/allocation audit against
   current stable baseline. No model reload, empty_cache or allocation-cap raise.

The first study stopped on its first warmup because head subset hashes are
registered lazily, separately from screen.prepare(rows). Preserve that failed
job/audit; prewarm exact subset counts1..128 before all timers. Warmup records
may populate kernels, but every formal short/long timer still requires unchanged
kernel hashes and zero fallback. No full256K run had started at that stop.
