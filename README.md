# SparkInfer-Q

**English** | [简体中文](README.zh-CN.md)

A research project on accelerating Qwen3.8-Flash-Next inference on a single NVIDIA DGX Spark (GB10 / SM121), borrowing scheduling and speculative verification from [DeepInfer](https://github.com/deepseek-ai/DeepInfer). As of 2026-10-28 it records experiments E0–E58 and runtime milestones K1–K4.

## Research question

On a desktop machine whose CPU and GPU share 128 GB of memory, can a large MoE model run faster, with longer context and several requests at once, while its **output stays token-for-token identical to the original model**?

- Model: Qwen3.8-Flash-Next, NVFP4-quantized, 48 layers (36 GDN linear-attention + 12 QSA sparse-attention), 512 routed experts per layer with top-10 routing, about 70 GiB of target weights.
- Scope: a single Spark and greedy decoding. No Qwen retraining, no production service, and no speedup claim without a measurement. Experiments choose the inference foundation (DeepInfer, TensorFold, vLLM, SGLang); none is assumed up front.

## Method

Each decode step has three parts:

1. **Draft**: the DFlash2 drafter reads the target's committed features and proposes up to seven tokens at once.
2. **Budget**: from predicted acceptance and measured cost, decide how many rows the target verifies this step (DeepInfer's MTPVarlen).
3. **Verify and commit**: the target verifies those rows in one forward pass, accepts the longest prefix that matches greedy decoding, and writes only that prefix into the GDN/QSA state.

Every experiment follows the same rules:

- **Correctness contract**: each candidate is checked against serial token-by-token execution on output tokens, committed logits and the full state; any mismatch is a failure.
- **Plans frozen in advance**: tasks, variables and pass gates go into `*_PLAN.md` before the run; results go into `*_REPORT.md`.
- **Negative results are kept**: a candidate that misses its gate is recorded as not adopted; thresholds are not loosened afterwards.
- **All overheads are timed**: feature extraction, CPU transfers and prediction count toward the measured time.

## Key results

| Area | Result | Experiments |
| --- | --- | --- |
| Decode: verification budget | A trained lightweight acceptance predictor is about 10% faster than the lagged budget (44.08 vs 40.21 tok/s on 20 new tasks) | [E15](docs/E15_CONFIDENCE_TRAINING_REPORT.md), [E16](docs/E16_BILINGUAL_EXPANSION_REPORT.md) |
| Decode: drafting | Full-vocabulary Q4 draft output head, +9.23% | [E6](docs/E6_DRAFT_HEAD_REPORT.md) |
| Decode: target execution | Exact screened output head +3.41% (46.91 → 48.51 tok/s); CUDA Graph +1.28% | [E33](docs/E33_EXACT_HEAD_SCREEN_REPORT.md), [E29](docs/E29_TARGET_VERIFICATION_REPORT.md) |
| Prefill | Eight-row chunks make complete requests 1.67× faster; switching to the owned prefill with 512-row chunks then cuts a 1,536-token request from 15.35 s to 2.13 s | [E8](docs/E8_INTEGRATION_REPORT.md), [E41](docs/E41_PREFILL_GRAPH_REPORT.md) |
| Long context | A single 256K request (262,080 tokens) runs with 3/3 retrieval; with a 256K input, dynamic DFlash makes decode 88% faster than target-only (16.36 → 30.81 tok/s) | [E46](docs/E46_256K_REPORT.md), [E49](docs/E49_LONG_DYNAMIC_REPORT.md) |
| Multiple requests | Two requests share one target forward, +63.4% aggregate decode throughput; two 256K requests run together with outputs and state identical to running each alone | [Joint target](docs/JOINT_TARGET_EXECUTION_REPORT.md), [E53](docs/E53_LONG_CONCURRENCY_REPORT.md), [E54](docs/E54_REQUEST_DRAFT_CACHE_REPORT.md) |

Each number holds only for its report's tasks and timing boundaries; rows are not comparable and gains must not be multiplied. Most tasks are synthetic prompts. **One-line conclusions for every experiment are in [docs/README.md](docs/README.md).**

## What did not work (also a result)

- **Kernel-by-kernel tuning gives little.** Target verification is 84% of decode time (E9), and expert kernels are its largest part (E30). Yet 11 candidates — warp and prefetch tuning, staging, weight reuse, dual streams, instruction pipelining (E10, E30–E32, E34, E35, E37–E39, E43, E52) — were not adopted: gains of a few percent at most, or slowdowns.
- **Bigger predictors did not speed up decode.** After retraining, an MLP was slightly better than the linear model offline (+0.21%), but not in actual decode (E19, E24, E25).
- **A better cost model alone helps little.** The routed-expert union explains verification cost, but it is only known after verifying (E12).

The gains that held came from deciding what to compute (verification budget, drafting, head screening) and from restructuring prefill.

## Current blockers

- **Memory**: the target is eight concurrent 256K requests under the 88 GiB CUDA allocation cap. Even with INT4 KV the forecast is 92.82 GiB (E55).
- **KV compression accuracy**: FP8 / INT4 KV give 8.42% / 11.23% relative logit error, above the pre-registered 3% gate; protecting sensitive layers, affine codecs and deferred compression did not fix it (E55–E58).
- **Multi-request serving**: two-request dynamic DFlash with async SSE runs, but does not yet consistently beat the optimized single-request engine run sequentially ([PR #19](https://github.com/EdwinWang831/spark-infer-q/pull/19), in progress).
- **Native TensorFold comparison**: 45.62 vs native 49.18 tok/s overall in E7; not rerun since.

## Repository layout

| Directory | Contents |
| --- | --- |
| [`spark_infer_q/`](spark_infer_q) | Reusable library: greedy verification contract, budgets and predictors, draft caches, prefill, long-context state, runtime (`kaifra_*`; Kaifra is the engine's working name) |
| [`scripts/`](scripts) | Run, collection and export scripts for each experiment, named by experiment number |
| [`tests/`](tests) | CPU unit tests; no GPU needed |
| [`docs/`](docs) | Plan and report for every experiment, with a [themed index](docs/README.md) |
| [`results/`](results) | Sanitized data for every experiment, re-checkable by its export script |

`main` collects finished work; active branches and PRs are listed in the [repository map](docs/REPOSITORY_STATUS.md).

## Getting started (no GPU, no weights downloaded)

The scripts below use only Python's standard library and **do not install packages or modify GPU settings**. The research environment targets Python 3.12.

```bash
git clone https://github.com/EdwinWang831/spark-infer-q.git
cd spark-infer-q

# Prints a JSON environment report without importing PyTorch or probing torch.cuda.
python3 scripts/probe_environment.py --static-only

# Point to an existing local Hugging Face config.json; do not download checkpoints.
python3 scripts/inspect_hf_config.py /path/to/model/config.json

# Unit tests; no GPU needed. Some tests skip without PyTorch or NumPy.
python3 -m unittest discover -s tests -v
```

Running the full model on a Spark needs a prepared research environment, external TensorFold source and checkpoints; see the [prompt-chunk executor](docs/PROMPT_USAGE.md), the [K2 runtime](docs/K2_DFLASH_SCHEDULER_USAGE.md) and the [resident worker](docs/RESIDENT_RESEARCH_WORKER.md).

**Privacy:** environment reports are not committed by default. Inspect reports before sharing; never commit API keys, tokens, model weights, private paths or personal identifiers.

## Upstreams and attribution

- [DeepSeek DeepInfer](https://github.com/deepseek-ai/DeepInfer) (Apache-2.0): C++ Scheduler and MTPVarlen verification budget
- [TensorFold](https://github.com/ashhart/TensorFold): model execution and CUDA kernels; retain the applicable upstream licenses and notices
- [MiaAI-Lab's TensorFold recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) (MIT): E43/E44 adapt its prefill expert tiles; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- [Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2): externally supplied draft checkpoint and reference implementation
- [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) (Apache-2.0): prompt source for E17 predictor retraining; see the [attribution](docs/data/OASST1_ATTRIBUTION.md)
- [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang): candidate baselines, not yet measured

**Do not vendor upstream source or weights without reviewing the respective licenses and notices.** This repo contains planning notes, diagnostic/build/test harnesses, an experimental request executor and runtime, adapters to externally supplied TensorFold blocks and the full target, a DFlash2 bridge, and sanitized evidence. It does not include a complete Qwen model implementation or model weights; loading and the remaining target operators use externally supplied TensorFold source and checkpoints.

License: [Apache 2.0](LICENSE).
