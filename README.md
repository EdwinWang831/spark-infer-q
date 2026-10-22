# SparkInfer-Q

**English** | [简体中文](README.zh-CN.md)

A research project on accelerating Qwen inference on a single NVIDIA DGX Spark (GB10 / SM121), inspired by [DeepInfer](https://github.com/deepseek-ai/DeepInfer).

## At a glance

As of 2026-10-22, the code, tests and sanitized data for E0–E16 are all on `main`.

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
| [E11](docs/E11_PROJECTION_REPORT.md) | MXFP8 split-K projections | decode +0.62%, complete requests −0.64% | Opt-in, default unchanged |
| [E10](docs/E10_TARGET_KERNEL_REPORT.md) | Expert-kernel warps and staging depth | +0.08% on holdout | Not adopted |
| [E12](docs/E12_EXPERT_VERIFICATION_REPORT.md) | Verification cost from the routed-expert union | offline cost error −26.17%, but the union is only known after verifying; features available online gave −2.42% | Not adopted |
| [E14](docs/E14_CONFIDENCE_BUDGET_REPORT.md) | Budget from binned draft margins | +0.065%, interval crosses zero | Not adopted |

**How to read these numbers**

- Each experiment uses its own task set, so tok/s values in different rows are not comparable, and gains must not be multiplied.
- All results are single Spark, single request, greedy decoding. Most tasks are synthetic prompts with context up to about 2K tokens.
- Every timed run's output tokens, committed logits and GDN/QSA state were checked against serial token-by-token execution.
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

## How it works

Each decode step has three parts:

1. **Draft**: DFlash2 reads the target's committed features and proposes up to seven draft tokens at once. The output head is Q4; the draft backbone and selector stay BF16.
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

- Rerun the native TensorFold comparison after E8.
- Wire the confidence predictor into the resident execution entry point and test generalization on independent external tasks (the next step named in the E16 report).
- 4K+ context: handle DFlash's 2048-token sliding window and absolute positions first.
- Let two requests share one target forward, starting from the E5 scheduler bridge.
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
