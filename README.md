# SparkInfer-Q

**English** | [简体中文](README.zh-CN.md)

A research project on accelerating Qwen inference on a single NVIDIA DGX Spark (GB10 / SM121), inspired by [DeepInfer](https://github.com/deepseek-ai/DeepInfer).

## At a glance

As of 2026-10-23, this repository includes E0–E32 implementations, tests and sanitized evidence (E26 cancelled at the user’s request). E29 measured target Graph +1.28% and Graph＋W8 +2.04% on eight reused roots. E30 rejects a BF16 head-layout candidate. E31 halves static expert-scale conversion instructions but finds no reliable pair speedup (+0.23% warm, −0.18% cache-perturbed); no new full-decode result. E32 removes unused activation staging rows but real projection replay slows about 1%; the candidate is rejected.

- **What it does**: Qwen3.8-Flash-Next (NVFP4, TensorFold kernels) runs speculative decoding with the DFlash2 drafter. DeepInfer's MTPVarlen decides how many drafts to verify at each step. The executor is a synchronous, single-request, greedy research prototype.
- **Largest single decode gain**: a lightweight confidence predictor trained in E15/E16. On separately frozen new tasks it is about 10% faster than the default lagged budget (E16: 44.08 vs 40.21 tok/s). It is opt-in for now.
- **Prompt processing**: eight-row prompt chunks make complete requests 1.67× faster, cutting summed time to first token from 19.69 s to 4.23 s; decode speed is essentially unchanged (E8).
- **Not there yet**: no overall lead over native TensorFold (45.62 vs 49.18 tok/s in E7; not rerun since E8). No multi-request GPU batching, 4K+ context or random sampling yet.

## Key results

| Experiment | Change | Result | Status |
| --- | --- | --- | --- |
| [E6](docs/E6_DRAFT_HEAD_REPORT.md) | Full-vocabulary Q4 draft output head | decode 41.70 → 45.54 tok/s (+9.23%) | Default |
| [E8](docs/E8_INTEGRATION_REPORT.md) | Eight-row prompt chunks using the decode arithmetic | complete requests 1.67× faster; decode −1.14% | Default |
| [E13](docs/E13_DRAFT_CACHE_CONFIDENCE_REPORT.md) | Incremental cache for DFlash context projections | +6.27% on 1K–2K inputs; −0.31% on short inputs | Opt-in |
| [E15](docs/E15_CONFIDENCE_TRAINING_REPORT.md) | Trained seven-depth confidence predictor | +10.00% over the lagged budget and +10.88% over fixed seven drafts, on 6 new tasks | Opt-in |
| [E16](docs/E16_BILINGUAL_EXPANSION_REPORT.md) | Retrained on 600 English/Chinese prompts | +9.61% over the lagged budget and +3.18% over the E15 predictor (2 tasks slower), on 20 new tasks; English +7.60%, Chinese +10.75% | Opt-in |
| [E17](docs/E17_COMPLETED_TRAINING_REPORT.md) | Targeted bilingual predictor retraining | 36.83 tok/s; +11.02% vs fixed budget, only +0.09% vs E16 (interval crosses zero), on 20 roots | No confirmed gain over E16 |
| [E19](docs/E19_MLP_CAPACITY_REPORT.md) | Same-data 11,157-parameter MLP | Offline efficiency −0.55% vs linear; promotion gate failed | E19 candidate not promoted |
| [E21](docs/E21_ADAPTIVE_COLLECTOR_REPORT.md) | Adaptive training-label collector | Collection computation 21.13% less time on six roots; known Chinese case 2.15% slower | Validated collection entry, not serving TPS |
| [E22](docs/E22_PREDICTOR_DIAGNOSIS_REPORT.md) | Optional NumPy scalar MLP runtime | Head-only 0.3353→0.0118 ms (28.38×); diagnostic replay still −0.06% vs linear | CPU prototype, no new decode TPS |
| [E24](docs/E24_COVERAGE_EXPANSION_REPORT.md) | Retrain on 504 new QA/code/math/dialogue/long-input roots | Offline +0.252% vs frozen MLP, +0.212% vs linear; positive intervals | Offline gate passed |
| [E25](docs/E25_ONLINE_COMPARISON_REPORT.md) | Actual decode, 12 roots / 108 paired runs | New MLP 47.15, frozen MLP 47.28, linear 47.49 tok/s; both relative intervals cross zero | No confirmed online gain |
| [E27](docs/E27_LINEAR_COST_REPORT.md) | Matched linear weights: NumPy execution and causal live cost control | Original 43.39, NumPy 43.65, live 43.42 tok/s; NumPy +0.60%, repetition-sensitive interval crosses zero | Default unchanged; MLP/hybrid deferred |
| [E28](docs/DRAFT_BACKBONE_W8_REPORT.md) | Five draft MLP blocks, INT8 weights / BF16 activations | Draft-stage time −6.89%; fixed-budget decode 42.04→42.29 tok/s (+0.61%, interval −0.52%…+1.66%) | Correctness passes; no confirmed decode gain |
| [E29](docs/E29_TARGET_VERIFICATION_REPORT.md) | Target CUDA Graph with unchanged DFlash feature taps | Matched linear budget: 47.05→47.65 tok/s (+1.28%); Graph＋W8 48.01 (+2.04%); both intervals positive | Controlled gain on 8 reused roots; default unchanged |
| [E30](docs/E30_TARGET_PROFILE_REPORT.md) | Profile frozen W8/linear/Graph; exact BF16 head weight layout | Target kernel time: experts41.3%, quantized projections26.9%, BF1623.4%; layout head-only+0.8%, below gate | Profile completed; candidate rejected, no new decode TPS |
| [E31](docs/E31_EXPERT_SCALE_REPORT.md) | Packed FP8 expert-scale conversion, unchanged precision | Conversion instructions halved; expert pair +0.23% warm / −0.18% cache-perturbed, both intervals cross zero | Exact checks pass; candidate rejected, no decode promotion |
| [E32](docs/E32_COMPACT_STAGING_REPORT.md) | Compact activation staging in MXFP8 projections | Shared memory/CTA −16%…−31%; real projection replay −1.06% warm / −0.92% cache-perturbed | 224 preflight +96 real checks pass; candidate rejected |
| [E11](docs/E11_PROJECTION_REPORT.md) | MXFP8 split-K projections | decode +0.62%, complete requests −0.64% | Opt-in, default unchanged |
| [E10](docs/E10_TARGET_KERNEL_REPORT.md) | Expert-kernel warps and staging depth | +0.08% on holdout | Not adopted |
| [E12](docs/E12_EXPERT_VERIFICATION_REPORT.md) | Verification cost from the routed-expert union | offline cost error −26.17%, but the union is only known after verifying; features available online gave −2.42% | Not adopted |
| [E14](docs/E14_CONFIDENCE_BUDGET_REPORT.md) | Budget from binned draft margins | +0.065%, interval crosses zero | Not adopted |

**How to read these numbers**

- Each experiment uses its own task set, so tok/s values in different rows are not comparable, and gains must not be multiplied.
- All results are single Spark, single request, greedy decoding. Most tasks are synthetic prompts with context up to about 2K tokens.
- Generation experiments check output tokens, committed logits and GDN/QSA state against serial execution. CPU-only and collection studies describe their separate validation scope in the linked reports.
- Feature extraction, CPU transfers and prediction are included in the timings.

The remaining experiments laid groundwork or diagnosed:

| Experiment | Scope | Finding |
| --- | --- | --- |
| E0–E2 | Environment audit, DeepInfer build, SM121 kernel probes, model loading and forward pass | Full target weights load and run; native DeepGEMM BF16 rejects SM121 |
| [E3](docs/E3_WIDE_BLOCK_REPORT.md) | DFlash2 with DeepInfer's native dynamic budget, eight rows / up to seven drafts | State checks pass; with a depth-only prior the dynamic budget ties a suitable fixed length (42.21 vs 42.20 tok/s) |
| [E4](docs/E4_MATCHED_REPORT.md) | Short-context same-machine comparison | Dynamic +2.12% over fixed six drafts; no lead over native TensorFold MTP |
| [E5](docs/E5_ENGINE_BRIDGE_REPORT.md) | DeepInfer C++ Scheduler driving Qwen target-only requests | Seven GPU scenario groups pass; GPU requests still run one at a time |
| [E7](docs/E7_TENSORFOLD_MATCH_REPORT.md) | Warmed comparison with native TensorFold | 45.62 vs 49.18 tok/s overall; 45.57 vs 43.58 on eight natural-completion tasks |
| [E9](docs/E9_DECODE_PROFILE_REPORT.md) | Nsight profile of decode | Target verification 83.81%, drafting 14.39% |

[E20 fixed-width collection](docs/E20_FAST_COLLECTOR_REPORT.md) remains a research control; the adaptive [collector usage](docs/FAST_COLLECTOR_USAGE.md) is the recommended validated entry. [E18 short drafting](docs/E18_PREDRAFT_COMPUTE_REPORT.md) reduced draft-only time by 0.74% and did not establish an online win.

**E23–E25 (2026-10-23 UTC):** the fixed-data action-regret objective [failed its fresh gate](docs/E23_OBJECTIVE_COMPARISON_REPORT.md). The subsequent [504-prompt coverage retraining](docs/E24_COVERAGE_EXPANSION_REPORT.md) finished and passed its 72-root offline gate. Its [actual decode comparison](docs/E25_ONLINE_COMPARISON_REPORT.md) found −0.263% vs frozen MLP and −0.717% vs linear, with both intervals crossing zero. No confirmed online improvement or policy replacement; offline estimates remain distinct from measured decode.

**E26–E27 (2026-10-23 UTC):** the [hybrid study](docs/E26_RESIDUAL_HYBRID_PLAN.md) was cancelled at the user’s request after 18/108 timings, with no performance conclusion. The [linear online comparison](docs/E27_LINEAR_COST_REPORT.md) completed 12 new roots / 108 timings: live cost control +0.07% vs original and −0.53% vs frozen-cost NumPy; NumPy-only +0.60%, with a repetition-sensitive interval crossing zero. All correctness checks pass; resident weights were retained and the default is unchanged.

**E28 (2026-10-23 UTC):** [draft MLP W8A16](docs/DRAFT_BACKBONE_W8_REPORT.md) completed 8 reused engineering roots / 64 timings, with exact target tokens, committed logits and live state checks. Observed draft acceptance is 73.74%→73.58%; verification rounds increase by 16. BF16 controls remain resident, so packed-weight size is not a measured net memory saving. Fixed budget only; dynamic policy remains untested with this quantization.

**E29 (2026-10-23 UTC):** [target verification](docs/E29_TARGET_VERIFICATION_REPORT.md) completed 32 calibration prefixes / 1,536 width timings, 96 three-arm decode timings and 32 contemporaneous budget-control timings. The existing linear budget reduces computed verification rows 36.72%, verification time 12.70%, and improves decode 42.32→47.31 tok/s (+11.78%) versus fixed seven drafts; this revalidates the existing policy, not a new predictor gain. Pure Graph +1.28% is the new execution result with exactly matched decision/commit traces. The W8/cost change adds +0.75% over Graph alone, but that interval crosses zero. Tokens/logits/features/live state pass; weights stayed resident.

**E30 (2026-10-23 UTC):** [history audit and current-path profile](docs/E30_TARGET_PROFILE_REPORT.md) confirms E10/E11 already tried expert launch/staging and projection tuning. Four reused calibration roots /32 early steps identify actual target GPU kernels; profiler overhead is4.85%–8.67%. A new NK→KN BF16 output-head layout preserves exact logits on32 real activations but saves only0.8% head time while adding1.27GB, so it stops before full decode. Target/draft stay resident; no retraining or precision change.

**E31 (2026-10-23 UTC):** [expert-scale pair conversion](docs/E31_EXPERT_SCALE_REPORT.md) passes all 65,536 scale pairs, 24 real routed expert cases and eight diagnostic Graph/logit/feature/state comparisons. CUDA already provides this packed operation; it is an implementation optimization attempt, not a new algorithm. Static conversion instructions halve, but neither timing regime shows reliable gain, so the candidate stops before full decode. Weights remain resident and the default path is unchanged.

**E32 (2026-10-23 UTC):** [compact staging](docs/E32_COMPACT_STAGING_REPORT.md) removes zero-filled unused rows while retaining exact MMA/accumulation. All 224 arithmetic/guard cases and 96 real inputs pass, plus 8 diagnostic logit/feature/state checks. A first collector missed class-method calls and failed its count assertion; preserved evidence and a private-proxy fix complete the screen. Both timing intervals are negative, so no full-decode promotion. Weights remain resident; default unchanged.

## How it works

Each decode step has three parts:

1. **Draft**: DFlash2 reads the target's committed features and proposes up to seven draft tokens at once. The output head is Q4; the default draft backbone and selector stay BF16. E28 separately tests INT8 MLP weights with BF16 activations.
2. **Budget**: decide how many rows (1–8) the target verifies this step. The default is DeepInfer's native MTPVarlen with lagged acceptance rates; optionally the rates come from the E16 confidence predictor.
3. **Verify and commit**: the target verifies those rows at once, accepts the longest prefix that matches greedy decoding, and writes only that prefix into the GDN/QSA state.

| Part | Source | Code in this repo |
| --- | --- | --- |
| Target execution, CUDA kernels, GDN/QSA commit | TensorFold (externally supplied source and checkpoint) | `scripts/e3_target.py`, `spark_infer_q/csf_loader.py` |
| Drafter | DFlash2 | `spark_infer_q/dflash2_block.py`, `draft_head.py` (Q4 head), `draft_context.py` (context cache) |
| Verification budget | DeepInfer MTPVarlen | `spark_infer_q/verification_budget.py`, `confidence_head.py`, `confidence_features.py` |
| Greedy verification contract | this repo | `spark_infer_q/speculative.py` |
| Prompt chunking | this repo | `spark_infer_q/prompt.py` |
| Scheduler bridge (E5) | DeepInfer C++ Scheduler | `spark_infer_q/deepinfer_bridge.py`, `executor.py` |

The complete executor is [`scripts/e8_ours.py`](scripts/e8_ours.py), which serves SSE. It needs a prepared research environment, external dependencies and checkpoints; see [usage](docs/PROMPT_USAGE.md). A [resident worker](docs/RESIDENT_RESEARCH_WORKER.md) keeps the weights loaded between experiments.

## Next steps

- Keep the E29 Graph path as an opt-in research candidate. Prioritize target data movement/fusion and real shared GPU compute; isolated W8 benefit beyond Graph is still unconfirmed.
- Rerun the native TensorFold comparison after E8.
- Keep MLP/hybrid deferred. Diagnose whether linear budget changes reduce actual verification work before further training; retain independent tests and actual decode gates.
- 4K+ context: handle DFlash's 2048-token sliding window and absolute positions first.
- Let two requests share one target forward, starting from the E5 scheduler bridge. The overall multi-request design is in the [global scheduler RFC](docs/GLOBAL_COST_AWARE_SCHEDULER_RFC.md) (Chinese; design only).
- Five written but unrun proposals (prefill numerics reference, expert-footprint rate, the online stages of the expert-aware budget, draft fidelity, exact head screening) are listed under [proposals without results](docs/EXPERIMENTS.md#proposals-without-results-yet).

## Getting started (no GPU, no weights downloaded)

Python 3.12 is the target for eventual DeepInfer builds. The scripts below use only Python's standard library and **do not install packages or modify GPU settings**.

```bash
git clone https://github.com/EdwinWang831/spark-infer-q.git
cd spark-infer-q

# Prints a JSON environment report without importing PyTorch or probing torch.cuda.
python3 scripts/probe_environment.py --static-only

# Point to an existing local Hugging Face config.json; do not download checkpoints.
python3 scripts/inspect_hf_config.py /path/to/model/config.json

# Unit tests; no GPU needed. A few tests skip without PyTorch.
python3 -m unittest discover -s tests -v
```

**Privacy:** environment reports are not committed by default. Inspect reports before sharing; never commit API keys, tokens, model weights, private paths or personal identifiers.

## Where to find things

| Looking for | Go to |
| --- | --- |
| Plans, pass criteria and conclusions for every experiment | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| A single experiment | `docs/E*_PLAN.md` (frozen plan), `docs/E*_REPORT.md` (report), `results/eN/` (sanitized data) |
| Usage | [prompt-chunk executor](docs/PROMPT_USAGE.md), [Q4 draft head](docs/DRAFT_HEAD_USAGE.md), [draft context cache](docs/DRAFT_CONTEXT_CACHE_USAGE.md), [projection route](docs/E11_PROJECTION_USAGE.md), [resident worker](docs/RESIDENT_RESEARCH_WORKER.md) |
| Architecture and porting | [architecture candidates](docs/ARCHITECTURE.md), [adapter interfaces](docs/ADAPTER_INTERFACES.md), [initial source audit](docs/PORTING_MATRIX.md) |
| Design proposals | [Global scheduler](docs/GLOBAL_COST_AWARE_SCHEDULER_RFC.md), [Kaifra vision](docs/KAIFRA_VISION_RFC.md), [implementation roadmap](docs/KAIFRA_IMPLEMENTATION_ROADMAP.md), [decode frontier](docs/KAIFRA_DECODE_FRONTIER_RFC.md) and [validation matrix](docs/KAIFRA_DECODE_VALIDATION_MATRIX.md). Proposals, not implemented serving capabilities |

## Research question and scope

Can we bring DeepInfer's engine, scheduling, speculative-verification and memory-management strengths to **a single DGX Spark**, while adapting the model executor and GPU kernels to Qwen3.8-Flash-Next? Or does a vLLM / SGLang-based implementation reach verified speed with less engineering effort? **Experiments choose the inference foundation**; none is assumed up front.

- **In scope:** verified SM121 compatibility, Qwen model adapters, GPU-side speculative verification, accurate GDN/QSA state rollback/commit, latency/throughput profiling.
- **Out of scope (initially):** retraining Qwen, building a production web service, assuming DeepInfer already supports GB10, claims of speedups without measurements.
- **Baselines:** compare against TensorFold, vLLM and SGLang on the same Spark, input set, sampling settings and checkpoint format. Published results provide context only; matched performance and runtime availability must be verified.

## Upstreams and attribution

- [DeepSeek DeepInfer](https://github.com/deepseek-ai/DeepInfer) (Apache-2.0)
- [TensorFold](https://github.com/ashhart/TensorFold) (model execution and CUDA kernels; retain the applicable upstream licenses and notices)
- [Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (externally supplied draft checkpoint and reference implementation)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

**Do not vendor upstream source or weights without reviewing the respective licenses and notices.** This repo contains planning notes, diagnostic/build/test harnesses, a bounded experimental request executor, adapters to externally supplied TensorFold blocks and the full target, a bounded eager DFlash2 bridge, and sanitized evidence. It does not vendor a complete Qwen model implementation, third-party kernels or model weights; the full loading/smoke harness uses externally supplied TensorFold source and checkpoint files.

License: [Apache 2.0](LICENSE).
