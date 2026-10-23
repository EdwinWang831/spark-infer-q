# SparkInfer-Q

**English** | [简体中文](README.zh-CN.md)

**Experimental inference acceleration for Qwen models on NVIDIA DGX Spark (GB10 / SM121), inspired by [DeepInfer](https://github.com/deepseek-ai/DeepInfer).**

## Latest measured results

As of **2026-10-21**, the E4–E9 implementation, tests and sanitized evidence are integrated into `main`. The latest complete-generation executor uses **eight-row prompt chunks and a full-vocabulary Q4 draft head**.

| E8 paired comparison, same ten tasks | Previous serial prompt | Eight-row prompt | Change |
| --- | ---: | ---: | ---: |
| Wire time to first token (TTFT), summed task medians | 19.69 s | **4.23 s** | **4.66× faster** |
| Complete-request time, summed task medians | 37.92 s | **22.67 s** | **1.67×; 40.22% less time** |
| Weighted decode throughput | 45.91 tok/s | 45.39 tok/s | -1.14% |

These are sums of per-task medians, **not individual request latencies**. The improvement comes from prompt processing; decode did not improve. One model load, ten reused short-context tasks and four paired repetitions produced 80 timed requests. All 40 full-generation logit/state oracle comparisons (3,348 committed logit rows) and all 80 timed output-ID/post-request state checks passed. See the [E8 report](docs/E8_INTEGRATION_REPORT.md), [data](results/e8/2026-10-21-integrated-prompt.json) and [executor usage](docs/PROMPT_USAGE.md).

**Latest native comparison (E7, before prompt integration):** 200 balanced, warmed timings measured our dynamic Q4 path at **45.62 decode tok/s** versus native TensorFold MTP at **49.18** overall; on eight natural-completion tasks, **45.57 versus 43.58**. Cross-engine output hashes matched on 8/10 tasks; native full IDs/logits/state were unavailable for an exact comparison. Native TensorFold has **not been rerun after E8**, so no overall lead is established. This is a pinned same-machine recipe, not a comparison against the latest upstream performance claims. [Report](docs/E7_TENSORFOLD_MATCH_REPORT.md) · [data](results/e7/2026-10-21-tensorfold-match.json).

**Latest diagnosis (E9, no new speedup):** Untraced E8 requests spend **83.81%** of decode in target verification and **14.39%** in drafting. Three Nsight samples show **91.76% GPU activity timeline coverage**, which is not SM utilization. Expert, qmmf and BF16 matrix kernels account for **78.6% of cumulative kernel time**. About 99.1% of CPU synchronization intervals overlap GPU work; synchronization time cannot be counted as directly removable overhead. Capture failures, sampling overhead and control limits are retained in the [report](docs/E9_DECODE_PROFILE_REPORT.md) and [data](results/e9/2026-10-21-decode-profile.json).

Earlier [E6 draft-head measurements](docs/E6_DRAFT_HEAD_REPORT.md) found full-vocabulary Q4 dynamic decoding **9.23% faster** than the same-task BF16-full draft head (45.54 versus 41.70 tok/s; 256 timings and 64 oracle checks). Q4 is now used by the complete executor. This quantizes the draft output head only; the target uses NVFP4 and the draft backbone, selector and KV remain BF16. This is not a general claim of BF16-equivalent quality. Separate study gains must not be multiplied.

**E13 experimental branch (opt-in):** On three synthetic 1,024/1,536/1,920-token inputs, incremental DFlash context projections improved weighted decode **39.73→42.22 tok/s (+6.27%)**, including initial cache build and append. All 24 timings passed output/live-state checks; six separate committed-logit oracles passed. The 12 shorter tasks showed **−0.31%**, so the global default is unchanged. This is decode-only, 96-token-capped, same-executor evidence; no new native TensorFold ranking or 4K+ claim. A separate confidence screen reduced held-out Brier error 18.64%, without an online policy speed claim. The research worker keeps the existing weights resident. [Report](docs/E13_DRAFT_CACHE_CONFIDENCE_REPORT.md) · [cache usage](docs/DRAFT_CONTEXT_CACHE_USAGE.md).

**E14 online confidence trial (no confirmed speedup):** All 108 timings and 27 committed-logit/live-state oracles passed across nine new tasks (including 1K–1.8K inputs). The margin policy measured **37.790 versus 37.765 tok/s (+0.065%)** against its post-draft depth control; the time-reduction interval crossed zero. Both controls chose identical width sequences. All 32 margin-bin combinations chose anchor+5 drafts under the frozen curve; fresh cost diagnostics still chose that width in 30–32/32 combinations. Deep conditional acceptance was overestimated on the new tasks. Keep this experimental: broader depth calibration and action sensitivity must precede another online speed claim. Same resident load, no new native TensorFold comparison. [Report](docs/E14_CONFIDENCE_BUDGET_REPORT.md) · [data](results/e14/2026-10-21-online-budget.json).

**E15 trained confidence budget (experimental):** A seven-depth lightweight predictor was fitted on 104 new synthetic prompts /8,344 serial positions, with grouped train/calibration/test splits and SQL held out as a family. On six separately frozen online tasks, the selected F1 path measured **44.30 decode tok/s versus 40.28 for the cached lagged-budget control (+10.00%)**, and 39.96 for calibrated fixed-seven drafting (+10.88%). All six task medians and five aggregate repeats were positive; 90 timed output/state checks and 18 independent oracles (2,286 committed logit rows) passed. Features, CPU transfers and prediction are timed. This is 128-token-capped greedy decode, including 1K/1.5K synthetic inputs; target/draft weights remain frozen and resident. This pilot did not cover natural EOS or broad tasks; E16 below adds those checks and the F0/F1 ablation. No new native TensorFold ranking or default change. [Report](docs/E15_CONFIDENCE_TRAINING_REPORT.md) · [evidence](results/e15/2026-10-21-confidence-training.json) · [experimental predictor](results/e15/2026-10-21-selected-model.json).

**E16 bilingual expansion (verified 2026-10-22, experimental):** Collected 600 authored synthetic prompts (240 English /240 Chinese /120 mixed), 36,846 sampled serial positions and 463 natural completions; trained both frozen-backbone heads. On 20 separately frozen online tasks /300 timings /100 independent oracles (31,540 committed logit rows), selected F1 measured **44.08 tok/s versus lagged 40.21 (+9.61%) and the old E15 F1 42.72 (+3.18%)** on the same tasks and cost curves. English improved **7.60%**, Chinese **10.75%** versus lagged; 15/20 online tasks reached EOS, with selected 1K/1.5K input and up-to-1K output checks. Two tasks regress against the old predictor. F1's 0.30% aggregate edge over F0 is inconclusive; all prediction/transfer/cache costs are timed. Same resident load, opt-in; no default change, 4K+, user-traffic generalization or new native TensorFold ranking. [Report](docs/E16_BILINGUAL_EXPANSION_REPORT.md) · [evidence](results/e16/2026-10-21-bilingual-expansion.json) · [predictor](results/e16/e16-selected-model.json).

**E17/E18 preparation (2026-10-22, no new online speed claim):** A training-only 512-position regret pool and bounded reweighting recipe are ready. A new OASST1 source manifest freezes 480 reviewed human English/Chinese roots plus 120 grouped mixed-language derivatives (420/90/90 split), all tokenization checks passing; formal collection is covered by the execution snapshot below. True block4 Q4 drafting passed 1,872 paired action checks on 48 training prefixes, but reduced summed warm draft time by only 0.74%. A hindsight prefix oracle suggests examining skip-draft/full-draft gating; its +3.06% estimate is not online throughput. No default change or reload. [E17 preparation](docs/E17_TARGETED_RETRAINING_REPORT.md) · [E18 results](docs/E18_PREDRAFT_COMPUTE_REPORT.md).

**E17 completed (2026-10-22 UTC):** All collection, sealed calibration selection, held-out evaluation and 20-root /360-run online controls finished. The selected ordinary F1 linear head achieved **36.83 decode tok/s**, **+11.02% versus fixed budget**, but only **+0.09% versus E16** (the grouped interval crosses zero). All 120 committed-logit/state oracle checks passed; no default replacement or reload. [Final report](docs/E17_COMPLETED_TRAINING_REPORT.md) · [completed evidence](results/e17/completed/2026-10-22-targeted-training.json).

**E19 MLP completed, negative result (2026-10-22 UTC):** Both fixed 11,157-parameter MLP candidates fitted; calibration selected the ordinary MLP. On the reused E17 held-out split its cost-inclusive offline efficiency was **0.55% below the selected linear head**, failing the prespecified promotion gate, so online MLP timing was not run. No measured MLP decode TPS or new external-test claim. [Report](docs/E19_MLP_CAPACITY_REPORT.md) · [completed evidence](results/e19/completed/2026-10-22-mlp-capacity.json).

**E22 post-test diagnosis and CPU prototype (2026-10-23 UTC):** About 77% of the observed MLP deficit is attributable to its extra prediction fee along the declared counterfactual path; equal-fee decisions still show no clear advantage. An optional NumPy scalar runtime reduced head-only latency **0.3353→0.0118 ms (28.38×)** on Spark. All 56,227 saved positions matched within 8e−16 probability error and had no budget-action differences at matched fees. This is CPU-head acceleration and exploratory offline replay, **not model decode speedup or promotion**. [Diagnosis and next steps](docs/E22_PREDICTOR_DIAGNOSIS_REPORT.md).

**E20/E21 learning-label collector (verified 2026-10-22):** The recommended adaptive collector reuses the frozen E16 F1 budget plus past-only serial fallback. On six training-root probes /36 paired timings, summed task-median collection computation fell **83.47→65.83 s (1.268×; 21.13% less time)**; four fresh confirmation roots improved 1.286×. All output/full committed-feature/logit/state/label checks and the real two-prompt entry smoke passed without reloading weights. A known Chinese case still costs 2.15% more than serial, reduced from fixed-eight E20's 22.1% regression. Small single-request greedy collection study, excluding final JSON writes and extra oracle comparisons; not serving decode TPS. Existing E17/E19 sealed producers/data remain unchanged. [Report](docs/E21_ADAPTIVE_COLLECTOR_REPORT.md) · [usage](docs/FAST_COLLECTOR_USAGE.md) · [evidence](results/e21/2026-10-22-fast-collector.json).

## Research question

Can we retain DeepInfer's engine, scheduling, speculative verification and memory-management advantages while adapting its model executor and GPU kernels to Qwen3.8-Flash-Next on **a single DGX Spark**? Or does a vLLM / SGLang-based implementation provide a better verified speed-versus-effort tradeoff?

**We will let experiments choose the inference foundation**, rather than assume one from the beginning.

## Current implementation

The current prototype combines **TensorFold model execution**, an adapted **DFlash2 proposer**, and **DeepInfer's original MTPVarlen verification budget**. Our Python execution loop connects target features, draft proposals, accepted-prefix state commits and experimental local policies. Target CUDA kernels and GDN/QSA commit/replay operations come from TensorFold.

The complete Q4 SSE research executor is [`scripts/e8_ours.py`](scripts/e8_ours.py); it defaults to eight-row prompt chunks using [`spark_infer_q.prompt.prompt_state`](spark_infer_q/prompt.py). It preserves the validated decode arithmetic and commit path during prompt processing. See [prepared-environment usage](docs/PROMPT_USAGE.md); external dependencies and checkpoints are required. This is a synchronous, single-request experimental executor. E3–E9 DFlash2 measurements are not benchmarks of a complete DeepInfer engine.

A separate [native-scheduler bridge (E5)](docs/E5_ENGINE_BRIDGE_REPORT.md) lets the real DeepInfer C++ Scheduler drive the complete Qwen target-only path. Seven GPU scenario groups passed, checking 364 full-logit rows and 79 live states. Device state remains owned by the Qwen backend; native blocks provide logical accounting. GPU requests execute serially. The full Worker/cache/async pipeline and actual multi-request GPU batching remain open.

The next measured priorities are target-kernel efficiency and useful verification budgets, followed by draft-backbone compression. Graph execution and synchronization changes need their own controlled comparisons. A fresh native TensorFold comparison is still needed after E8. E13 adds selected 1K–2K cache/state and decode checks; 4K+ context, concurrency, random sampling and official BF16 draft fidelity remain unvalidated.

## Investigation stages

| Stage | Goal | Status |
| --- | --- | --- |
| E0 | Hardware/software fingerprint + upstream compatibility audit | PASS_STATIC; no CUDA execution tests |
| E1 | Build DeepInfer core and probe SM121 kernel compatibility independently | Minimal GPU gate passed; native DeepGEMM BF16 blocked; full backend compatibility incomplete |
| E2 | Minimal model adapter, forward pass, prefill and decode | Partial: full target weights loaded and selected token/logit execution ran; decode-arithmetic prompt chunking subsequently validated in E8; independent full-model reference and complete DeepInfer engine integration remain open |
| E3 | Draft adapters, speculative verification, GDN/QSA state correctness | Bounded greedy DFlash2 integration passed with native DeepInfer variable budget; random sampling, full scheduler and native Qwen MTP remain open |
| E4 | Apples-to-apples engine comparison, including TensorFold / vLLM / SGLang | Partial: [short-context TensorFold / DFlash2 measurements](docs/E4_MATCHED_REPORT.md) completed; warmup/numerical limits retained; vLLM / SGLang, long context and concurrency not measured |
| E5 | Minimal native engine bridge | Bounded target-only C++ Scheduler / full-Qwen lifecycle checks passed; full Worker/cache/async and GPU batching remain open |
| E6 | Optional draft-head compression | 256 bounded timings / 64 oracle runs passed; full Q4 dynamic +9.23% versus same-task BF16-full; fixed subset coverage and cross-device limits retained |
| E7 | Warmed matched native comparison | 200 timings / 40 candidate nonserial oracles passed; overall dynamic 45.62 vs native 49.18 tok/s before E8; prompt integration subsequently validated in E8 |
| E8 | Prompt chunking and complete Q4 generation | Prompt: 81 exact comparisons / 160 timings, 4.66×; integrated: 40 full-generation oracles / 80 timings, TTFT 4.66× and complete requests 1.67×; decode -1.14% |
| E9 | Decode CPU/GPU time distribution | Diagnostic complete: 3 valid traces, 12 generation oracles, 15 output/state checks; GPU activity 91.76%; no new speedup claim |

For precise pass criteria and failure classification, see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
See also [architecture candidates](docs/ARCHITECTURE.md) and the [initial source audit](docs/PORTING_MATRIX.md).
The [adapter interface map](docs/ADAPTER_INTERFACES.md) separates engine/cache and MTP requirements from completed operator experiments.

## Safe Phase-0 probes (no weights downloaded)

Python 3.12 is the target for eventual DeepInfer builds. The following inspection scripts use only Python's standard library and **do not install packages or modify GPU settings**.

```bash
git clone https://github.com/EdwinWang831/spark-infer-q.git
cd spark-infer-q

# Prints a JSON environment report without importing PyTorch or probing torch.cuda.
python3 scripts/probe_environment.py --static-only

# Point to an existing local Hugging Face config.json; do not download checkpoints.
python3 scripts/inspect_hf_config.py /path/to/model/config.json

# Standard-library unit tests; no GPU needed.
python3 -m unittest discover -s tests -v
```

**Privacy:** environment reports are not committed by default. Inspect reports before sharing; never commit API keys, tokens, model weights, private paths or personal identifiers.

## Scope

- **In scope:** verified SM121 compatibility, Qwen model adapters, GPU-side speculative verification, accurate GDN/QSA state rollback/commit, latency/throughput profiling.
- **Out of scope (initially):** retraining Qwen, building a production web service, assuming DeepInfer already supports GB10, claims of speedups without measurements.
- **Baselines:** compare against TensorFold, vLLM and SGLang on the same Spark, input set, sampling settings and checkpoint format. Published results provide context; matched performance and runtime availability must be verified rather than assumed.

## Upstreams and attribution

- [DeepSeek DeepInfer](https://github.com/deepseek-ai/DeepInfer) (Apache-2.0)
- [TensorFold](https://github.com/ashhart/TensorFold) (model execution and CUDA kernels; retain the applicable upstream licenses and notices)
- [Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (externally supplied draft checkpoint and reference implementation)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

**Do not vendor upstream source or weights without reviewing the respective licenses and notices.** This repo contains planning notes, diagnostic/build/test harnesses, a bounded experimental request executor and adapters to externally supplied TensorFold blocks/full target and a bounded eager DFlash2 bridge, plus sanitized evidence. It does not vendor a complete Qwen model implementation, third-party kernels or model weights; the full loading/smoke harness uses externally supplied TensorFold source and checkpoint files.

License: [Apache 2.0](LICENSE).
