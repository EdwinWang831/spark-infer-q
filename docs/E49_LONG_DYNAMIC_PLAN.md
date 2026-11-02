# E49: long-output adaptive DFlash and four-request capacity

2026-10-25. User explicitly removed fixed-draft comparisons and requested starting
target-only versus DFlash + current dynamic budgeting, with four concurrent256K
requests as a goal. Do not claim four long requests from the separate K4 short
context result (capacity1536 / physical2048).

Reuse resident PID46235/load_count1 through its serialized queue. Keep the88GiB
allocation limit and8GiB host-available reserve, existing weights and external
operators. No reload/download/cache clearing or changes in the other worktree.

1. Measure actual262160-capacity INT8 state storage and common eager/prefill
   allocations. Calculate the four-state lower bound before allocating further
   slots. If even that exceeds the allocation limit or host reserve, preserve an
   explicit capacity stop. An allocation forecast is not four-request execution.
   E47 failed INT8 precision gate remains failed.
2. Same-resident single-request long-output comparison: target-only and adaptive
   DFlash/E17/E29/Graph, with no fixed-draft arm. Prompt261120 and cap512 leave
   lookahead room within logical262144 / physical262160. Actual EOS still stops.
   Use a valid chat with bilingual long background and a substantive English
   technical answer request; record exact input and output token identities.
3. Compare within each KV format (BF16 and INT8) under the same input/timer.
   Prewarm Graph/head subsets before timing, keep all drafting/predicting and
   verification inside request timing. Preserve exact greedy outputs and full
   committed KV/scale/index/GDN/PLE state across target-only/adaptive arms.
   Report actual output length, residual decode tok/s, accepted/observed drafts,
   width histogram, verify rounds, draft/verify/commit costs, TTFT and full wall.
   Do not force512 tokens by suppressing EOS or relabel a short answer as long.
4. Across formats report observed output differences and task checks separately;
   this bounded case does not pass the original general KV quality gate.
5. Four-request adaptive integration requires per-request draft caches, native
   NEXTN counters/commit, Graph isolation and fair scheduling, not merely removing
   the current two-slot/NEXTN0 guards. Capacity failure blocks the four256K run;
   continue the independently useful single-request comparison without claiming
   multi-request completion. Perform independent post-return allocation/source/
   operator/group/environment audits and retain all failed stages.

Capacity-v1 measured one full INT8 state allocation delta4665401856 bytes and
common allocated78121510400 bytes. Four-state lower bound96783117824 bytes
(90.14GiB) exceeds88GiB before decode Graphs/head auxiliaries/draft caches.
The stop is an allocated-byte forecast from one real layout, not an OOM test
or four-request execution. Host-available extrapolation ignores reusable
reserved allocator slack and must not be presented as measured four-slot RAM.
Independent capacity audit522 recovered77393408000 bytes on the same worker.

Study-v1 stopped after the first adaptive long request at the formal head-kernel
registration gate. Prewarming candidate counts only for one row did not cover
subset keys indexed by (verify_rows,candidate_count). Its valid control is kept
as a diagnostic, not substituted into the new matched pair. Study-v2 prewarms
rows1..8 x candidate counts1..128 before timers and preserves a complete last-run
record before rejecting any future timing/fallback gate. Library/model arithmetic
is unchanged. Audit526 independently recovered the stable baseline after failure.
