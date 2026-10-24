# Experiment plan

**Rule: use small probes to isolate initial blockers, then combine verified components into meaningful integration milestones.** Keep internal diagnostics, but do not stop after every small substep once the foundational gates pass. Do not download the full DeepSeek V4.1 checkpoint to test compatibility. Do not overwrite a working local inference environment.

## Current priorities

2026-10-24 prefill evidence: [E36](E36_PREFILL_NUMERICS_REPORT.md) passed exact large-chunk prompt and decode-handoff gates. [E37](E37_PREFILL_PROFILE_REPORT.md) rejected GDN scratch reuse (matched wall +1.02%) and identified routed experts as 52.5–54.9% of cumulative GPU kernel time on two 512-token probes. Prioritize a measured prefill expert-compute candidate, preserving full logits/features/state checks; E36 default integration and complete-request timing remain separate gates. The other workspace owns K4 dynamic DFlash and service work.

2026-10-23 engine milestone: [K1 target-only runtime](K1_RUNTIME_REPORT.md) builds on E5's real C++ Scheduler, adding a reusable request API, output events, lifecycle/cleanup and loopback SSE. Seven native/GPU groups pass (233 full-logit rows, 91 live-state comparisons); DFlash2 integration and actual cross-request GPU batching remain subsequent work. See [usage](K1_RUNTIME_USAGE.md).

Subsequent [K2 integration](K2_DFLASH_SCHEDULER_REPORT.md) connects Q4 DFlash2, native lagged/E16 confidence and the current E17 post-draft BudgetProbe to that API. The latter is checked against the existing loop at every selected width. Cross-request GPU batching remains subsequent work; see [K2 usage](K2_DFLASH_SCHEDULER_USAGE.md).

Evaluate and port useful DeepInfer components for Qwen on a single DGX Spark. MTP is an important workstream alongside the engine, scheduling, memory/cache management, GPU operators and Qwen executor. Its importance does not make the other components mere dependencies of MTP, and it does not require choosing a complete port over selective reuse before evidence is available. Keep the E0–E4 identifiers and prioritize work by dependencies, correctness evidence, likely benefit and porting cost.

| Workstream | Next evidence needed |
| --- | --- |
| SM121 operators and quantization | Complete layer-0 MoE passed with real routing/aggregation, BF16-input compressed experts and MXFP8 shared expert; selected GDN/QSA blocks and state checks subsequently met gates, with CPU full-path route divergence recorded; minimal executor lifecycle and matched GPU reference subsequently tested; next remaining formats and full-model integration |
| Qwen model and state adapter | Selected real GDN/QSA block, BF16 state and FP8 KV partition checks ran with MTP off; minimal request metadata and independent GPU reference subsequently tested; next full-model logits and loading |
| Engine, scheduling and memory | Test core scheduling/sequence lifecycle beyond the existing CPU sequence tests; assess async scheduling, graph integration and allocation/cache contracts independently of MTP |
| Drafts and verification budgets | Real Qwen DFlash2 plus original DeepInfer MTPVarlen passed bounded greedy/state integration; next full draft-forward reference, complete-answer/repeated comparisons and independent calibration; native Qwen MTP remains separate |
| Performance and integration choice | Attribute each component's effect using controlled comparisons against the existing TensorFold baseline; assess combinations after individual correctness gates pass |

The initial complete expert check found 14.6%–15.4% output L2 differences from FP4 activation quantization on synthetic inputs. Subsequent [native packed-weight/BF16-input expert tests](E2_NATIVE_REPORT.md) passed 11 cases on SM121 without GPU weight expansion. The subsequent [complete MoE milestone](E2_MOE_REPORT.md) passed 8 joint tests including routing, all 512 expert execution coverage, MXFP8 shared expert and graph replay. The subsequent [GDN/QSA decoder/state milestone](E2_BLOCK_REPORT.md) met 16 selected gates; one GDN Top-10 set divergence versus CPU full path remains recorded, and exact model equivalence is unproven; total runtime memory remains to be budgeted beyond the 63.28 GiB metadata estimate for main routed expert packs. Continue the engine/cache and MTP contracts in the [interface map](ADAPTER_INTERFACES.md); neither requires declaring MTP the sole objective. Real-model MTP validation depends on correct target-only execution. Candidate optimizations should also be measured with MTP disabled to distinguish their standalone value from interactions with speculation.

### MTP track — planned sequence

1. **Trace and specify the mechanism.** At pinned DeepInfer revision `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`, map proposal token IDs/distributions, target verification, accepted-prefix length, corrective/bonus token, and committed target/draft state. Treat DFlash proposal generation, confidence scheduling, sampling and state management as separate interfaces.
2. **Build a model-independent reference controller.** Use controlled logits and a tiny stateful executor to exercise K=0, all accepted, first/middle/last rejection, EOS, cancellation and request remapping. Check that rejected drafts never enter committed GDN/QSA state and that accepted-prefix state matches sequential execution. This can progress without porting native SM100 GEMMs; toy-state success is not proof of real Qwen state semantics.
3. **Connect native Qwen MTP at fixed K.** Independently validate the actual proposal generator and target-only logits/state. Start from small supported draft lengths. The observed single MTP layer and configured draft depth six are different quantities; neither proves DFlash's block-generation behavior is available. Reuse the Qwen executor and quantization adapters validated by the broader model-porting work.
4. **Add confidence-based verification budgets.** DeepInfer's DFlash has learned confidence-head weights. Do not assume Qwen supplies an equivalent head, or treat a draft token's maximum softmax probability as a calibrated acceptance probability. First use fixed K; then evaluate position/context-specific acceptance estimates, calibrate on separate data, and consider a learned head only if justified. Re-measure Spark's costs rather than reuse SM100 performance curves.
5. **Compare target-only, existing TensorFold MTP, fixed-K candidate and adaptive candidate.** Hold weights, workload and sampling policy constant. Measure actual emitted tokens, total latency, draft/verification/restore overhead, acceptance and state memory. Acceptance rate alone is not evidence of speedup.

Important transfer limits from source:

- `csrc/engine/mtp_varlen.hpp` selects a total verification budget using estimated expected output divided by measured/estimated step time; the TileLang allocator distributes the budget across sequences using cumulative confidence. Selecting the budget and distributing it are distinct decisions. The CPU scheduler deliberately uses lagged confidence to avoid later draft positions influencing earlier verification eligibility; preserve or independently prove the relevant sampling invariant.
- `models/mtp/dflash.py` implements a DFlash-specific proposer and learned confidence head. Transfer the scheduling/verification ideas through an adapter; a native Qwen MTP layer is not a drop-in replacement for those weights or forward semantics.
- V4.1 decoder SWA bounded replay is a separate cache optimization. A finite suffix must not be assumed to reconstruct Qwen's GDN recurrent state. Establish exact snapshots/accepted-prefix commit or exact replay from a valid checkpoint first.
- The upstream sampler's `DRAFT_PROB_PENALTY=5e-5` perturbs residual sampling weights. Keep an exact reference and explicitly identify/test this approximation if retained; passing upstream tests does not establish mathematically unbiased sampling.

For SM121, evaluate operator ports across all workstreams, including matrix operations, routing/TopK, attention, state/cache movement, sampling and MTP budget allocation. Reuse verified TileLang/cuBLASLt routes and evaluate existing TensorFold operators where appropriate. Do not exclude a component solely because it is unrelated to MTP; classify its Qwen/single-GPU applicability and validate its value before committing to a substantial port.

## E0 — Environment fingerprint and source constraints

1. Run `python3 scripts/probe_environment.py --static-only` on the DGX Spark and, separately, in the existing inference container. This mode does not import PyTorch or call `torch.cuda`. Query compute capability with `nvidia-smi`/NVML, which does not create a CUDA context. Keep raw logs local.
2. Run `python3 scripts/inspect_hf_config.py /path/to/config.json` on an existing local Qwen checkpoint.
3. Record the precise DeepInfer commit, Qwen config/weight revision, Python/CUDA/PyTorch versions and observed GB10 capability.
4. Classify source constraints in [PORTING_MATRIX.md](PORTING_MATRIX.md).

**Pass:** reproducible hardware and model identity. **Fail:** missing or contradictory environment information. **No speed measurements yet.**

Completed as `PASS_STATIC` on 2026-10-20; see [E0_REPORT.md](E0_REPORT.md). Static success does not establish CUDA/kernel execution compatibility. Raw reports remain outside tracked files; only sanitized observations are published in the local working tree.

## E1 — Core build and kernel compatibility

Use a **separate** Python environment, matching DeepInfer's documented toolchain where possible.

Build/test independently:
- DeepInfer C++ extension and imports (without weights).
- DeepGEMM: isolated matrix operation + reference comparison.
- FlashMLA, DeepEP, DeepSelect, TileLang: test only code paths relevant to Qwen and single-GPU operation.
- SM121-specific execution, memory usage, optional CUDA Graph capture.

For each component log one outcome:
`NOT_RUN` / `BUILD_FAIL` / `IMPORT_FAIL` / `KERNEL_FAIL` / `NUMERIC_FAIL` / `PASS` / `NOT_APPLICABLE`.

**Pass:** at least a minimal independent kernel and extension can execute correctly. **Fail:** mark and investigate the failing component, not the whole architecture.

2026-10-20 progress: core extension build/import and the same 14 upstream CPU sequence tests passed in two environments. Subsequent SM121 probes passed 12 RMSNorm cases, one CUDA Graph replay, cuBLASLt BF16, and NVFP4 synthetic/actual-weight-slice comparisons. Native DeepGEMM BF16 dispatch rejects architecture 12. The minimal GPU gate is satisfied; full backend compatibility remains incomplete. See [core results](E1_BUILD_REPORT.md) and [GPU results](E1_GPU_REPORT.md).

The subsequent [complete expert experiment](E2_EXPERT_REPORT.md) independently matched CSF decoding and FP32 weight dequantization to executed TensorFold reference source, then separately measured arithmetic agreement and activation quantization differences. Agreement with the same stored quantized representation does not establish model quantization quality.

## E2 — Model executor and cache adapter

1. Introduce new architecture registration and a Qwen-specific config adapter; do not change DeepSeek V4.1 behavior.
2. Create a tiny synthetic checkpoint for shape and scheduling integration (no quality claims).
3. Test single token, prefill then decode, different prompt lengths, and memory lifecycle.
4. Compare logits to a known-correct implementation at matched precision and input.

**Pass:** numerically justified agreement on supported paths, no invalid memory accesses.

2026-10-20 partial result: three full real expert projections plus SwiGLU passed selected SM121 arithmetic checks at batch 1/8/32; an FP32-dequantized-weight control preserving BF16 activations also passed. FP4 activation fidelity is unresolved, and that initial experiment did not validate a model adapter, routed MoE, full layer, prefill/decode state or logits. Later MoE evidence is recorded below. See [E2_EXPERT_REPORT.md](E2_EXPERT_REPORT.md).

Subsequent native result: the original TensorFold expert CUDA source compiled for SM121 and passed 11 cases using compressed GPU weights and BF16 activations, including batch 17/tails/strides, BF16 and FP32 down output, skip and graph replay. The maximum selected complete-output relative L2 error was about 0.0378%. A metadata scan budgeted one native pack of main routed expert weights/scales at 63.2815 GiB, excluding all other model/runtime components; full-model capacity is not confirmed. See [E2_NATIVE_REPORT.md](E2_NATIVE_REPORT.md).

Subsequent integrated result: complete layer-0 MoE passed 8 joint SM121 checks with synthetic BF16 hidden states, actual router/top-10, device plans, compressed expert weights, MXFP8 shared expert and weighted combine. The maximum selected output L2 error was 0.02821%; all 512 experts were covered using an explicitly injected-logit stress case. The initial 7/8 result and CPU/GPU router BF16-boundary diagnosis are retained. No decoder block, model state, logits or performance validation is implied. See [E2_MOE_REPORT.md](E2_MOE_REPORT.md).

Subsequent decoder/state result: real layers 0/GDN and 3/QSA ran independently with four-stream hyper-connections, MXFP8 projections and complete MoE. Sixteen selected checks met preset gates, including prefill/decode, snapshot/restore, reset/growth, two-request layouts, fixed-prefix graph forward, real-budget sparse QSA on primed caches and FP8 KV partitioning. CPU output L2 errors were 0.11932%/0.12926%; one GDN row changed its selected expert set against CPU full path, while same-logit Top-10 and native partition expert sets agreed. This is not full model or DeepInfer executor validation. See [E2_BLOCK_REPORT.md](E2_BLOCK_REPORT.md).

Subsequent [minimal executor milestone](E2_EXECUTOR_REPORT.md): a bounded, serialized Python host executor now runs real GDN/QSA block backends. Sixteen lifecycle/state GPU checks passed, including cancel/drain/generation, interleaved positions/padding, complete prefix restore, growth invalidation, continuous graph reuse, cross-stream tensor dependencies and host-exception rollback. Two independent FP32 PyTorch GPU diagnostics (TF32 disabled) report output L2 differences of 0.09776%/0.13366%, with 0/17 route-set differences per layer. Two local timing comparisons show graph/eager output identity and eager/graph median ratios 1.327×/1.388×. These are one-block measurements; the reference is a numerical control, not a quality ceiling. Full model loading, token/logits, quality, long-context memory and the DeepInfer C++ scheduler bridge remain open. No MTP was enabled.

Subsequent [complete loading milestone](E2_LOAD_REPORT.md): all 48 target layers and 24,576 routed experts loaded together, with 69.8511 GiB unique CUDA weight storage, 26.8222 GiB read-only n-gram mappings and separately retained raw MTP/vision tensors. All 375,487 source tensors have a storage/representation disposition; no full-table residency or whole-checkpoint payload hash is implied. Selected complete-target token/logit execution and restore/replay ran with MTP disabled. Prefill versus serial final logits differ by about 10.68%, triggering numerical review; this is not a task-quality percentage. The user explicitly authorized stopping the existing service for this milestone. Full-model reference/quality, full-context capacity, performance and DeepInfer scheduler integration remain open.

## E3 — Speculative decode and recurrent state

Test `K=0` before enabling speculative decoding. Then test each supported native MTP length and verify:
- All drafts accepted; first rejected; rejection in the middle; EOS and stopping.
- GDN state and QSA/KV state remain correct after accepted-prefix commit/replay.
- Greedy results match target-only when the same model and settings are used.
- Stochastic sampling distribution is evaluated statistically; do not demand identical samples merely because execution paths differ.
- Only after fixed verification passes, try variable-length budget allocation.

**Pass:** validated state invariants and unbiased speculative sampling under the chosen supported sampling scheme.

Subsequent [DFlash2 integration](E3_DFLASH2_REPORT.md): real 603M BF16 drafter weights and the complete target passed six forced state cases and three 16-token greedy prompts in fixed/dynamic modes. A thin binding calls original DeepInfer MTPVarlen, with lagged empirical rates and Spark costs. Outputs/logits/live state matched same-path serial execution. Aggregate exploratory timing ratios were 1.90×/2.03×, one sample favored fixed; only thinking prefixes and one run per case were measured. This is a bounded greedy milestone, not the full stochastic/engine E3 acceptance. Independent draft/model forward parity, target prefill review, full answers, stable performance, long context and full scheduler integration remain open.

Followup [official draft/complete-generation study](E3_FOLLOWUP_REPORT.md): eight requests completed to EOS, with 72 balanced timing runs. Tokens/live state matched serial, with separate exact-logit checks. Fixed/dynamic aggregate ratios were 1.781×/1.789×; all dynamic decisions chose width4 and one short task slowed down. FP32 draft mathematical controls passed; BF16 draft outputs differ, and a target MoE precision control did not fix prefill. These are bounded decode observations, not E4 service comparisons or general quality/long-context proof.

Subsequent [five-mode budget audit](E3_BUDGET_AUDIT_REPORT.md): 200 balanced timings on the eight reused regression tasks and six separate calibration prompts. Fixed1/2/3 aggregate decode ratios were 1.278×/1.597×/1.794×; dynamic was 1.799× and chose width4 at all 475 decisions. Three tasks were replayed at all 166 serial positions (1992 timed actions, 664 live-state checks). A future-aware path reduces modeled cost by 20.3% on the mixed-acceptance task, versus 3.5% on the high-acceptance code task. This exposes local prediction headroom, not an online speedup. Next evaluate independently calibrated pre-draft gating and verification choices on new inputs; full-engine/multi-request and numerical reference work remain separate.

Subsequent [frozen local-policy study](E3_LOCAL_POLICY_REPORT.md): 16 training / 6 independent calibration / 8 new test prompts, 392 balanced runs. A pre-draft gate plus calibrated post-draft progress predictor improved aggregate decode speed 1.95% over same-run fixed3 (38.55→39.30 tok/s); gate alone was slower. Eight tests completed to EOS with exact outputs/live states and separate kept-logit checks. A separate [immutable-math/EOS study](E3_DRAFT_FASTPATH_REPORT.md) ran 128 balanced measurements on the reused eight prompts: hidden/proposals stayed exact; cache alone gave only 0.21%, cache+EOS gave 1.26% over its own fixed3 baseline. No combined result or statistical significance is established. Eight-row execution was not yet validated at those milestones.

The subsequent [eight-row study](E3_WIDE_BLOCK_REPORT.md), following its [fixed design](E3_WIDE_BLOCK_PLAN.md), passed 44 full-target window/prefix checks, legacy four-row identity and full official eight-row FP32 controls using unchanged external SM121 kernels. On eight reused regression tasks, 200 balanced timings measured fixed3/fixed7/dynamic7 at 38.10/41.50/43.10 decode tok/s. Dynamic selected widths3..8, beating fixed7 by 3.86%; 56 separate non-serial oracle runs and all 272 main/secondary timings matched tokens/live state. Crucially, the separate 72-run matched control found dynamic effectively tied with a calibration-selected fixed width7 (six drafts): 42.21 vs 42.20 tok/s. Wider drafting is promising on this workload; online adaptation over a suitable static length is not established. Initial unsupported commit(keep=0) test-driver failure, official BF16 differences and target prefill limits are retained. No full scheduler, stochastic-sampling, native Qwen MTP, long-context or E4 engine claim follows.

## E4 — Performance, cost and selection

Compare DeepInfer-adapted, the existing TensorFold service, vLLM and SGLang under controlled conditions (first verify each candidate is available and correct):
- same model revision, weights/quantization, context, generated-token cap and sampling settings;
- concurrency 1, 4, 8; cold and warm runs;
- 512 / 4k / 16k input lengths where the hardware permits;
- tasks: ordinary conversation, code completion, reasoning.

Collect `TTFT_ms`, `TPOT_ms`, `output_tokens_per_second`, `total_output_tokens_per_second`, `draft_acceptance_rate`, `peak_memory_bytes`, power/thermal notes and failure count. Report token accounting explicitly and never use theoretical full-acceptance output as measured throughput.

**Pass:** reproducible improvement on at least one target workload without an unacceptable correctness/memory regression. Document negative results.

2026-10-20 bounded E4 result: [matched plan](E4_MATCHED_PLAN.md), [report](E4_MATCHED_REPORT.md), [public main data](../results/e4/2026-10-20-matched.json). Four ABBA loading sessions produced 336 timed runs over eight new English texts / fourteen settings. Dynamic measured 47.37 versus fixed6/fixed7 46.39/46.30 decode tok/s (+2.12%/+2.31%) within the candidate path. Candidate timed output IDs match own serial; 84 separate nonserial oracles check committed logits/final live state. All 240 objective main answer checks pass; prose/capped prefixes have no full quality claim. Cross-engine hashes differ on six settings, and native original first repetitions are slower under asymmetric per-case warmup; a separate post-hoc native warmup control is preserved alongside the main data. Original native MTP 48.82 tok/s is not a fair warmed leaderboard comparison; post-hoc per-case warmup measured 55.99 tok/s over 112 timings (28 additional warmups), with native hashes/counts/text unchanged. Cold prompt latency and target verification dominate engineering priorities. This completes bounded TensorFold measurements, not the broad E4 multi-engine/concurrency/long-context gate.

## E5 — Bounded native engine bridge

2026-10-21: [plan](E5_ENGINE_BRIDGE_PLAN.md), [report](E5_ENGINE_BRIDGE_REPORT.md), [data](../results/e5/2026-10-21-engine-bridge.json). The real DeepInfer C++ Scheduler now drives the complete Qwen target-only execution through a synchronous adapter. CPU toy and real-token replay checks passed the same seven-group matrix; GPU checks covered 364 full-logit rows and 79 live-state comparisons, with replay-equivalent dispatch/output ledgers and complete native logical-block return. Two GPU preflight harness failures (special-token vocabulary and tokenizer return type) were corrected and retained. This is not full LLMEngine/Worker, GPU cache-page mapping, async/ragged batch or a performance result. Target prefill numerical review remains open; DFlash2 draft-head profiling/compression is next.

## E6 — Optional draft-head compression

2026-10-21: [frozen plan](E6_DRAFT_HEAD_PLAN.md), [report](E6_DRAFT_HEAD_REPORT.md), [data](../results/e6/2026-10-21-draft-head.json), [reusable API and device boundaries](DRAFT_HEAD_USAGE.md). Four heads × fixed7/dynamic7 × eight new tasks × four repetitions produced 256 timings; 64 independent oracles checked 2,616 committed logit rows and live state, with timed outputs/state also exact. All tasks reached EOS. Full Q4 dynamic measured 45.54 versus BF16-full 41.70 decode tok/s (+9.23%); fixed subset coverage loss in Chinese is retained. Twelve stored-weight Q4 arithmetic cases and unchanged target-head SHA passed. Original target BF16 head remains resident; no model-wide memory saving, full BF16-target equivalence, cross-GPU, random-sampling, batching or cross-study native-MTP speed claim follows. BF16 remains the API default; an explicit optional provider and observable fallback support reuse. Q8/mixed precision is a future comparison, not measured by E6.

## Experiment record template

```yaml
experiment: E1-deepgemm-sm121
date_utc: null
hardware: null
driver: null
cuda: null
torch: null
upstream_sha: null
model_revision: null
quantization: null
command: null
status: NOT_RUN
correctness_reference: null
metrics: {}
notes: null
```

Never commit raw user prompts, API keys, identifying machine reports, model weights or binary traces.

## E7 — Warmed full-Q4 / native matched comparison

2026-10-21: [plan](E7_TENSORFOLD_MATCH_PLAN.md), [report](E7_TENSORFOLD_MATCH_REPORT.md), [data](../results/e7/2026-10-21-tensorfold-match.json). All 200 formal timings, 100 full warmups (including 40 candidate nonserial oracles) completed across native/ours/ours/native loads. Full Q4 dynamic 45.62 versus native MTP 49.18 aggregate decode tok/s; natural tasks 45.57 versus 43.58. Complete-request sums 38.14 versus 18.79 s, summed TTFT 19.77 versus 1.65 s. Native and candidate outputs differ on two tasks; no overall speed lead. Next validate full-prompt chunking through unchanged decode arithmetic before timing; do not mix INT8 or budget recalibration.

## E8 — Decode-arithmetic prompt chunking

2026-10-21: [plan](E8_PREFILL_PLAN.md), [report](E8_PREFILL_REPORT.md), [data](../results/e8/2026-10-21-chunked-prompt.json). 81 full prompt/tail comparisons (every-row logits/features, boundary live state and eight-step continuation) passed; 160 warmed timings. Eight-row chunks: 109.23 versus serial 23.43 prompt tok/s, 4.66×. Includes state init/capture/concat, excludes tokenization/network. No drafter resident or full speculative generation; serving default unchanged. Next integrate rows8 into complete Q4 dynamic requests and measure actual wire TTFT/complete wall/decode.

### E8 complete-generation integration

2026-10-21: [plan](E8_INTEGRATION_PLAN.md), [report](E8_INTEGRATION_REPORT.md), [usage](PROMPT_USAGE.md), [data](../results/e8/2026-10-21-integrated-prompt.json). New Q4 SSE executor defaults to rows8. All 40 full-generation oracle comparisons (3,348 committed logit rows) and 80 timed outputs/separate post-request live-state checks passed against serial rows1 and E7 IDs. Paired TTFT 19.69→4.23 s; complete-request sums 37.92→22.67 s (1.67×). Decode 45.91→45.39 tok/s, a retained 1.14% regression. One load / reused ten tasks; native was not rerun, so no new native ranking. Next remaining prompt costs, target verification/decode and budget feedback; INT8 remains a separate comparison.

## E9 — Decode time distribution

2026-10-21: [plan](E9_PROFILE_PLAN.md), [report](E9_DECODE_PROFILE_REPORT.md), [data](../results/e9/2026-10-21-decode-profile.json). E8 untraced target83.81% / draft14.39%. Three valid Nsight decode ranges cover202,995 kernels; GPU activity91.76%, not SM utilization. Expert/qmmf/BF16 families78.6% cumulative kernel time. Twelve independent generation oracles and15 post-request output/state checks pass. Initial repeated collection failed (one valid trace retained separately); single recording succeeds. Four controls are genuinely CUDA-untraced, eight intermediate controls CUDA-recorded/NVTX-inactive; correction retained. Next measure target kernels/verification effectiveness and draft backbone; no new speedup or native ranking.

## E10 — Target expert-kernel launch and staging

2026-10-21: [plan](E10_TARGET_KERNEL_PLAN.md), [report](E10_TARGET_KERNEL_REPORT.md), [data](../results/e10/2026-10-21-target-expert.json). Varied only warps per block (1/2/4/8) and register staging depth (1/2) of the original TensorFold NVFP4 expert kernel on all 512 real layer-0 experts with seeded BF16 inputs and hand-specified routing; weight format, activation precision and MMA/FMA order unchanged. All 198 exact variant checks pass (outputs, guards, skipped shared slot). Screening 15 shapes / 8,100 CUDA-event intervals; the frozen 2-warp/depth-1 shortlist on an independent-seed holdout (12 shapes / 2,160 intervals) was only +0.08% equal-weight geomean under a 128 MiB cache sweep (+0.53% with reuse), so no full-generation A/B ran and the default is unchanged. Nsight Compute counters returned ERR_NVGPUCTRPERM; no bandwidth or SM-utilization claim. Status PASS_EXACT_MICRO_NO_PROMOTED_OPTIMIZATION: launch/staging tuning has little room; structural changes to bytes read remain open.

## E11 — Target projection execution

2026-10-21: [plan](E11_PROJECTION_PLAN.md), [report](E11_PROJECTION_REPORT.md), [usage](E11_PROJECTION_USAGE.md), [micro data](../results/e11/2026-10-21-projection-micro.json), [generation data](../results/e11/2026-10-21-projection-integrated.json). Screened BF16 tiling (output head, hyperconnection down/up) and MXFP8 split-K reduction (GDN in/out, shared expert) on real weights: 207 exact variant checks and 7,452 event intervals, all outputs exact. BF16 showed no stable gain. Explicit MXFP8 partials on GDN out were 2.07× locally with cache reuse but 1.06× under a cache sweep (shared gate/up 1.39×/1.01×), and in paired full generation (one load, ten seen tasks, 80 timings) decode moved only 45.445→45.727 tok/s (+0.62%), complete requests −0.64%; one of four repeats was slower and some tasks regressed. 60 generation oracles (5,022 committed logit rows) and all 80 timed outputs/live states pass; the budget's verify widths shifted, so work changed as well as kernel cost. Kept as an opt-in `ProjectionRoute` (800 KiB scratch, existing `qmmf` binary); default unchanged. No native TensorFold ranking, multi-request, long-context or sampling claim.

## E12 — Expert-aware verification

2026-10-21: [plan](E12_EXPERT_VERIFICATION_PLAN.md), [report](E12_EXPERT_VERIFICATION_REPORT.md), [data](../results/e12/2026-10-21-expert-verification.json), [resident worker](RESIDENT_RESEARCH_WORKER.md). Reused PR #1 helpers. All 72 prefixes / 576 width actions / 1,728 timings passed selected logits/features/live-state checks. True expert union reduced held-out cost MAE 26.17%; previous width4 union only 2.42%. Fixed-prior same-prefix budget diagnostic improved 0.563% / 0.092%, with true measured costs 1.277%; none is online decode speedup. Context39–116 only. No neural predictor training, Target routing change or default policy change. Next test cheap pre-verification signals and actual budget opportunity before online A/B; keep Target/DFlash resident between experiments.

## E13 — Resident draft context cache and confidence signals

2026-10-21: [plan](E13_DRAFT_CACHE_CONFIDENCE_PLAN.md), [report](E13_DRAFT_CACHE_CONFIDENCE_REPORT.md), [usage](DRAFT_CONTEXT_CACHE_USAGE.md). Gated cache preserves selected FC/K/V/hidden/proposal checks after retaining an initial failed BF16 shape gate. Twelve short tasks /96 timings: 41.14→41.02 tok/s, −0.31%, no benefit proven. Three 1024/1536/1920-token inputs /24 timings: 39.73→42.22 tok/s, +6.27%; four paired repetitions positive, output/committed-logit/state checks pass. Includes build/append, excludes prompt/network; synthetic context cases and96-token cap, no4K+ or new native ranking. Held-out selector-margin Brier improves18.64% vs depth prior,16.28% vs adaptive prior; no online confidence policy tested. Source/data/failure records retained. Same resident worker, load_count1, default unchanged.

## E14 — Online confidence budget

2026-10-21: [plan](E14_CONFIDENCE_BUDGET_PLAN.md), [report](E14_CONFIDENCE_BUDGET_REPORT.md), [data](../results/e14/2026-10-21-online-budget.json). Nine new tasks /108 timings /27 oracles (3411 committed logit rows) pass. Cached native reference38.045, post-draft depth37.765, margin37.790 tok/s; margin versus depth+0.065%, time-reduction95% task-bootstrap−0.675% to+0.730%, no confirmed gain. Post-draft action sequences identical in all repeats. Native32-corner check always selects anchor+5 on frozen costs; fresh384/1024/1792-context costs select it in31/32/30 of32 corners. New deep acceptance is overestimated (depth5 mean prediction89.10%, observed57/81=70.37%, counted once per task). Fresh calibration is diagnostic, not a retuned online test. Same resident PID/weight objects, load_count1, READY;101 CPU checks,99 pass/2 skip. Next broaden depth calibration and require action sensitivity before another GPU trial; no native ranking or default change.

## E15 — Trained seven-depth confidence budget

2026-10-21: [plan](E15_CONFIDENCE_TRAINING_PLAN.md), [report](E15_CONFIDENCE_TRAINING_REPORT.md), [data](../results/e15/2026-10-21-confidence-training.json), [model](../results/e15/2026-10-21-selected-model.json). Corrected PR #6 throughput-oracle accounting.104 new synthetic prompts /8344 unique serial positions,60/24/12 prompt splits by semantic group plus8 SQL family holdouts; all proposal-path comparisons pass. Both F0/F1 CPU heads trained, F1 chosen on calibration before test. Held-out uniform-prefix estimated+8.59% (group interval+3.89% to+13.75%), oracle-gap capture40.32%; SQL+4.67%; offline is not online speed. Frozen six-task online90 timings /18 oracles /2286 committed logit rows pass: predictor44.304, lagged control40.276, fixed-seven39.956 decode tok/s (+10.00%/+10.88%). All task medians and five repeats positive; lagged time reduction9.09%, task-bootstrap5.45% to13.97%. Prediction/features/transfers included; prompt/clones/oracle excluded. Source frozen a3d0534, weights remain resident/load_count1/READY;111 CPU tests108 pass/3 skip, plus7 Spark CPU tests pass. Pilot capped generation, no natural EOS completion distribution, F0/F1 online ablation, default switch or new native TensorFold ranking.

## E16 — Bilingual expansion and old/new predictor comparison

Verified 2026-10-22; executed 2026-10-21: [frozen plan](E16_BILINGUAL_EXPANSION_PLAN.md), [report](E16_BILINGUAL_EXPANSION_REPORT.md), [evidence](../results/e16/2026-10-21-bilingual-expansion.json), [model](../results/e16/e16-selected-model.json). 600 authored synthetic prompts (240 en/240 zh/120 mixed), 120 grouped semantic scenarios, 36,846 sampled serial positions, 463 natural EOS; train/cal/test/OOD 300/100/100/100. Both small heads fitted; F1 selected on calibration. Independent CPU replay reproduces test/OOD metrics and verifies all source/data/model hashes. Final 20-task online300 timings/100 oracles/31,540 committed logit rows pass: F1 44.078, lagged40.215, old F1 42.721, new F0 43.946 tok/s. F1 +9.61% vs lagged, +3.18% vs old F1; time-reduction family-bootstrap intervals7.38%–10.46% /1.36%–4.32%. English +7.60%, Chinese +10.75% vs lagged. 15 natural EOS; selected 1K/1.5K inputs and up-to-1K outputs. Two old-predictor regressions retained; F1/F0 +0.30% inconclusive. Preceding 12-task144-timing ablation also passes. Same resident PID/load_count1/READY, failures unchanged; default remains opt-in. No 4K+, sampling, GPU batching, real-user generalization or new native TensorFold ranking.

## E17 — Targeted retraining preparation

2026-10-22: [plan](E17_TARGETED_RETRAINING_PLAN.md), [preparation report](E17_TARGETED_RETRAINING_REPORT.md). Training-only regret weights over18,567 E16 train positions and512 diverse hard positions prepared; no old test/cal/OOD rows enter the pool. Frozen approximate-cost Dinkelbach action regret is not recoverable wall time. OASST1 pinned Apache-2.0 source:480 reviewed human roots plus120 grouped mixed derivatives; en240/zh240/mixed120, new420/90/90 split, actual tokenizer capacity maximum970. At preparation time, new Target labels and four-way unweighted/weighted F0/F1 fit had not started; selection/gates frozen in recipe. Observed width-cost bias motivates paired training-prefix cost checks before attributing errors.

## E18 — True short-draft and skip-draft feasibility

2026-10-22: [plan](E18_PREDRAFT_COMPUTE_PLAN.md), [report](E18_PREDRAFT_COMPUTE_REPORT.md), [data](../results/e18/2026-10-22-draft-probe.json). Same resident weights; true block4 backbone/head3 versus block8/head7 plus no-draft action. 12 train tasks /48 prefixes /1,872 formal action checks /624 warmups /4,449 formal committed logit rows pass, new Q4 M3 arithmetic passes. Warm draft summed time reduction only0.74%; first-three proposals match38/48. Hindsight aggregate prefix oracle skip-or8 +3.06% vs only8; adding4 gives only0.38% marginal, no online throughput claim. Prefix-only, simulated append8 cache, block4-before8 order limits retained. Same PID/load_count1/READY, failures unchanged; do not switch default to three drafts. Next test cheap causal skip/full gate and independently finish E17 cost-aware retraining preparation.

### E17 formal execution — completed

2026-10-22: [final report](E17_COMPLETED_TRAINING_REPORT.md), [completed evidence](../results/e17/completed/2026-10-22-targeted-training.json). All 512 cost positions, 510 learning prompts, calibration selection and 90 fresh test prompts complete. Selected ordinary F1 passes 360 formal online timings /120 oracle checks on20 roots:36.828 tok/s, +11.02% vs fixed, +0.09% vs E16 with an interval crossing zero. No confirmed added speed over the prior predictor; default unchanged. The startup snapshot above is historical.

## E19 — MLP capacity comparison

2026-10-22: [plan](E19_MLP_CAPACITY_PLAN.md), [final report](E19_MLP_CAPACITY_REPORT.md), [evidence](../results/e19/completed/2026-10-22-mlp-capacity.json). Same-data137→64→32→7 MLP,11,157 parameters; calibration-selected unweighted model has −0.55% offline efficiency vs matched linear, fails promotion. No online MLP TPS.

## E20/E21 — Exact accelerated label collection

2026-10-22: [fixed-width report](E20_FAST_COLLECTOR_REPORT.md), [adaptive report](E21_ADAPTIVE_COLLECTOR_REPORT.md), [usage](FAST_COLLECTOR_USAGE.md). Adaptive collection on6 roots/36 timed runs reduces summed task-median collection computation83.47→65.83s (21.13%); four fresh confirmation roots1.286×. All output/full committed hidden feature/logit/state/label checks pass, including actual entry smoke; known Chinese case still+2.15% time. Collection speed, not serving decode TPS.

## E22 — Predictor diagnosis and optional CPU runtime

2026-10-23 UTC: [report](E22_PREDICTOR_DIAGNOSIS_REPORT.md), [diagnosis](../results/e22/2026-10-23-predictor-diagnosis.json), [CPU results](../results/e22/2026-10-23-cpu-runtime.json). Post-test counterfactual attributes about77% of observed MLP deficit to extra scalar execution fees along the stated decomposition. NumPy scalar runtime0.3353→0.0118ms,28.38× head-only;56,227 positions match within8e−16 probability and unchanged actions at matched costs. Replayed new-fee model remains−0.06% vs linear; no online generation or default replacement.

## E23/E24 — Sealed objective and coverage workflows

2026-10-23 UTC: [dated execution snapshot](E23_E24_EXECUTION_STATUS.md), [objective plan](E23_OBJECTIVE_COMPARISON_PLAN.md), [coverage plan](E24_COVERAGE_EXPANSION_PLAN.md). E23 changes only the training objective on identical architecture/data/backend; calibration gain+0.040%, followed by72 fresh prompts/60 roots. The [fresh test report](E23_OBJECTIVE_COMPARISON_REPORT.md) finds +0.0128% vs matched BCE and +0.0195% vs linear with intervals crossing zero; the new gate failed, no online run. E24 fixes the calibration-selected recipe; all504 tokenization checks and six GPU producer scenarios passed. Learning collection is running before expanded-data fit and a conditional72-root test.504 new prompts cover QA/code/math/real roles/long inputs. No default switch or new online speed claim from startup evidence.

## Proposals without results yet

These five plans are in main as proposals, not results. None has its own GPU run, and no conclusion should be cited from them. Each file opens with a PROPOSED status line. When one is scheduled, give it an E number, freeze its inputs and thresholds, record the run as a normal plan/report pair, and update its row here.

| Plan | From | Status | Reference code |
| --- | --- | --- | --- |
| [Prefill numerics reference](PREFILL_REFERENCE_PLAN.md): judge serial-decode and native-prefill prompt arithmetic against an FP32 dequantized reference instead of against each other, with a pre-registered rule for switching the prompt path | #1 | Not run | — |
| [Expert-footprint accounting](ACHIEVED_BANDWIDTH_PLAN.md): turn per-step routed expert unions and E9 kernel times into a per-family expert-footprint rate | #1 | Not run. Footprint is not measured DRAM traffic (see E12), so without hardware counters the result would only indicate kernel headroom; [E10](#e10--target-expert-kernel-launch-and-staging)/[E11](#e11--target-projection-execution) tuning has meanwhile measured +0.08%/+0.62% | `spark_infer_q/moe_cost.py` |
| [Expert-union-aware verification budget](MOE_AWARE_BUDGET_PLAN.md): predict step time from rows plus routed-expert union and swap only that model into MTPVarlen | #1 | Partly run. E12 ran the offline cost stage: the current step's true union cut held-out cost MAE 26.17%, a history-only union only 2.42%, and same-prefix budget diagnostics stayed at or below 1.28%. The online stages are on hold under the plan's stop rule | `spark_infer_q/moe_cost.py` |
| [Draft-fidelity acceptance](DRAFT_FIDELITY_ACCEPTANCE_PLAN.md): check whether the BF16 drafter's arithmetic lowers acceptance compared with official stepwise BF16 and FP32 drafters on the same serial prefixes | #4 | Not run | `spark_infer_q/draft_fidelity.py` (`step_outcome`, `paired_ratio`; `emitted_by_width` there is also used by E15/E16) |
| [Exact greedy head screening](EXACT_HEAD_SCREEN_PLAN.md): screen the target head with a Q8 copy and rigorous bounds, then compute exact BF16 logits only for candidates, keeping the greedy argmax | #7 | Not run; only a synthetic bound check (`scripts/head_screen_bound_sim.py`) | `spark_infer_q/head_screen.py` |
