# E50: optimize our long-context path, then account for eight full states

2026-10-25. User paused cross-engine comparisons and asked to improve our engine
first, then work out why eight concurrent256K requests do not fit. No Mia run,
fixed-draft comparison, model reload, service stop or cache clearing.

Use resident PID46235/load_count1, unchanged target arithmetic/BF16 KV, E17/E29
dynamic budgets and E49's261120-token English-tutorial input/output cap512.
The first candidate captures the stable five-layer DFlash query front/tail in
ten islands, reusing K4's island design with absolute-position staging. Original
context projections, eager attention, Q4 selector and CPU features stay intact.
Default off; reject unsafe short/nonrolling combinations. Dynamic budgets stay.

One owned chunk1024 prefill builds a common real prefix. Preserve active small
GDN/PLE snapshot and2040 draft-feature suffix, never clone full KV. Re-lease the
same Graph state, restore the small snapshot and reuse its immutable KV prefix
for component/decode diagnosis. These seeded timings exclude prefill and native
Scheduler/event overhead and must never be labeled HTTP/full-request results.

- Exact draft hidden/proposals/signals at the real absolute offset; complete
  baseline/candidate512-token greedy outputs, cumulative rates/widths/acceptance,
  committed KV/scale/index/GDN/PLE fingerprints; reference E49 BF16 tokens.
- Normal/profile/normal real256K baseline prefix,16 adaptive rounds each. CPU
  and CUDA trace phase ranges; analyze actual kernel events, never double count
  CUDA annotation parents or call host gaps Python/GPU utilization.
- Four AB/BA seeded512-token pairs. Require >=1% aggregate decode gain and every
  repetition positive, exact traces/state and no target/head fallback or formal
  JIT/capture. Failure remains failed; no chasing micro-timings as service wins.
- Only if that gate passes, run a contemporary uninterrupted native token-API
  baseline/candidate pair with real prefill, cap512 and all drafting costs inside
  timing. Preserve output/state/trace and independent cleanup audit. One pair is
  bounded engineering evidence, not universal TPS/default promotion.
- Inspect resident CUDA/host allocation accounting and use E49's measured INT8
  private-state layout to forecast8 x256K. Report current88GiB cap separately
  from hardware/system available memory, uncounted Graph/draft storage, reserved
  allocator slack and uncertainty. No eight-state allocation or support claim.

Always independent post-return operator/group/environment/module/source and
allocation audit against77393408000 bytes; preserve failed stages. No edits to
the separate K4 worktree or asynchronous messages to another chat.

The native CPU binding probe confirmed get_token_ids() materializes the full
std::vector before Python slices it (261120 prompt +512 outputs: about3.71ms
per read versus0.00045ms for an output-only tuple). Add default-off output_cache:
after native process_outputs, read bounded native last_accepted_token and
last_output_token fields, validate count/length/tail, append to the output ledger,
and compare the entire native output prefix once at retirement. Lifecycle pilot
548 covers twelve real native cases with cache off/on. Full study compares the
original chain against query islands + this output cache; the component probe
is graph-only, so its small gain is not relabeled as the combined service gain.

After the speed study and audit, measure one empty full-capacity INT8 and one
INT4 State using already-present external storage. No forward pass, eight-state
allocation, precision promotion or new production KV option. This distinguishes
an existing lower-bit memory route from a validated quality/serving result.

User then asked explicitly whether eight256K with FP8 is impossible. Keep FP8
as a target, not a ruled-out format. Layout-v2 adds an actual empty FP8 state
alongside INT8/INT4; the resulting eight-state figures remain forecasts and
exclude execution workspaces, physical host ownership and quality validation.

Static eight-FP8 follow-up: inspect the pinned raw IKC pool-reader liveness.
A CPU-only model checks new1024-row writes plus up to3 prior unfinished rows,
including zero/partial retention and future rewrites. This is a proposal to bound
raw index storage while retaining full BF16 pooled history and FP8 KV, not a
GPU implementation, measured saving or eight-request admission claim.
