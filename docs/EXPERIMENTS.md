# Experiment plan

**Rule: use small probes to isolate initial blockers, then combine verified components into meaningful integration milestones.** Keep internal diagnostics, but do not stop after every small substep once the foundational gates pass. Do not download the full DeepSeek V4.1 checkpoint to test compatibility. Do not overwrite a working local inference environment.

## Current priorities

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

## E4 — Performance, cost and selection

Compare DeepInfer-adapted, the existing TensorFold service, vLLM and SGLang under controlled conditions (first verify each candidate is available and correct):
- same model revision, weights/quantization, context, generated-token cap and sampling settings;
- concurrency 1, 4, 8; cold and warm runs;
- 512 / 4k / 16k input lengths where the hardware permits;
- tasks: ordinary conversation, code completion, reasoning.

Collect `TTFT_ms`, `TPOT_ms`, `output_tokens_per_second`, `total_output_tokens_per_second`, `draft_acceptance_rate`, `peak_memory_bytes`, power/thermal notes and failure count. Report token accounting explicitly and never use theoretical full-acceptance output as measured throughput.

**Pass:** reproducible improvement on at least one target workload without an unacceptable correctness/memory regression. Document negative results.

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
