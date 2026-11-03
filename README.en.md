# SparkInfer-Q

[简体中文](README.md) | **English**

Research on cost-aware speculative decoding and runtime execution for local large language models on a single NVIDIA DGX Spark (GB10 / SM121). Current hardware experiments use Qwen3.8-Flash-Next and a DFlash2 drafter, integrating scheduling and speculative verification from [DeepInfer](https://github.com/deepseek-ai/DeepInfer).

**Start here:** [research overview](docs/RESEARCH_OVERVIEW.md) · [中文研究导览](docs/RESEARCH_OVERVIEW.zh-CN.md) · [CPU predictor example](examples/adaptive_budget.py) · [experiment index](docs/README.md).

The overview connects research questions to implementation, evidence and limitations. Documentation reviewed on 2026-11-03, covering E0–E58, runtime milestones K1–K4 and four-request 128K engine validation.

## Unified engine entry

This branch integrates main and committed K4 work as Kaifra v0.4, with one model load, 128K total tokens per request, four running requests, adaptive DFlash, streaming/nonstreaming chat, cancellation, bounded admission, health, metrics and orderly shutdown. Run these commands in the prepared GPU environment.

```bash
python -m spark_infer_q doctor --root /path/to/prepared/runtime --checkpoint /path/to/local/target
python -m spark_infer_q serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target
```

Four 131,008-token BF16 caches were populated simultaneously, each producing 64 tokens. One 128K prefix matches the full-index reference at every chunk. Four-request HTTP outputs, valid state, features and dynamic traces match token-by-token and isolated dynamic controls. The 128K profile uses a 2,048-row raw index ring, exact shared views of fused draft weights and 512-row prefill, preserving full K/V and pooled history. Long GPU execution remains serial. See the [usage guide](docs/ENGINE_V04_USAGE.en.md) and [validation report](docs/ENGINE_V04_REPORT.md).

At 64K, the verified `--prefill-experts` extension continues to select 1,024-row prefill. In the earlier paired four-request 32K study, that profile reduced mean complete-cohort time from 114.90 to 94.99 seconds, a 17.33% reduction. Each capacity has separate timing evidence. Historical reports remain available for [v0.3](docs/ENGINE_V03_REPORT.md), [v0.2](docs/ENGINE_V02_REPORT.md) and [v0.1 cold startup](docs/ENGINE_V01_REPORT.md).

## Research question

On a desktop machine whose CPU and GPU share 128 GB of memory, can a large MoE model run faster, with longer context and several requests at once, while its **output and committed state match the same quantized target run serially**? Compressed target-KV candidates also face a separate BF16-KV-relative accuracy gate.

- Model: Qwen3.8-Flash-Next, NVFP4-quantized, 48 layers (36 GDN linear-attention + 12 QSA sparse-attention), 512 routed experts per layer with top-10 routing, about 70 GiB of target weights.
- Scope: a single Spark and greedy decoding, with frozen target and draft backbones during predictor training. The current execution layer uses external TensorFold operators and DeepInfer scheduling/budget interfaces. vLLM and SGLang remain unmeasured candidate baselines; a production service and cross-model generalization are not established.

## Project work

| Workstream | What this repository develops | Implementation and evidence |
| --- | --- | --- |
| Acceptance prediction | Training, calibration and inference for a lightweight predictor of draft-prefix acceptance, using signals available before target verification | [Predictor](spark_infer_q/confidence_head.py), [features](spark_infer_q/confidence_features.py), [E16](docs/E16_BILINGUAL_EXPANSION_REPORT.md), [E17](docs/E17_COMPLETED_TRAINING_REPORT.md) |
| Dynamic verification budget | Combine predicted progress and verification cost to select the current verification width; integrate the decision with upstream MTPVarlen | [Budget rule](spark_infer_q/confidence_head.py), [runtime integration](spark_infer_q/kaifra_speculative.py), [K2](docs/K2_DFLASH_SCHEDULER_USAGE.md) |
| Runtime and evaluation | Request/state ownership, prefill execution, shared target work, lifecycle checks, paired experiments and evidence exports | [Joint target](spark_infer_q/kaifra_joint_target.py), [prefill](spark_infer_q/prefill.py), [E54](docs/E54_REQUEST_DRAFT_CACHE_REPORT.md), [tests](tests) |

The project builds these prediction, execution and evaluation components on upstream model architectures, DFlash2, DeepInfer Scheduler/MTPVarlen and TensorFold operators. See the attribution below for the dependency boundaries.

## Method

Each decode step has three parts:

1. **Draft**: the DFlash2 drafter reads the target's committed features and proposes up to seven tokens at once.
2. **Budget**: from predicted acceptance and measured cost, decide how many rows the target verifies this step (DeepInfer's MTPVarlen).
3. **Verify and commit**: the target verifies those rows in one forward pass, accepts the longest prefix that matches greedy decoding, and writes only that prefix into the GDN/QSA state.

The predictor estimates progress; it does not generate the answer. The current post-draft policy chooses **verification length**, after drafting has already been paid for. Cross-request scheduling and GPU batching are separate mechanisms, described in the [overview](docs/RESEARCH_OVERVIEW.md).

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

## Candidates not adopted

- **The tested kernel candidates missed their promotion gates.** Target verification is 84% of decode time in E9, and expert kernels are its largest part in E30. Eleven candidates — warp and prefetch tuning, staging, weight reuse, dual streams, instruction pipelining (E10, E30–E32, E34, E35, E37–E39, E43, E52) — were not adopted. This does not rule out other kernel designs or workloads.
- **The tested larger predictor has no confirmed decode gain.** After retraining, an MLP was slightly better than the linear model offline (+0.21%), but not in actual decode (E19, E24, E25).
- **Post-verification features cannot directly make a causal decision.** The routed-expert union explains verification cost, but it is only known after verifying (E12).

The gains that held came from deciding what to compute (verification budget, drafting, head screening) and from restructuring prefill.

## Current blockers

- **Memory**: the target is eight concurrent 256K requests under the 88 GiB CUDA allocation cap. Even with INT4 KV the forecast is 92.82 GiB (E55).
- **KV compression accuracy**: all compressed candidates still miss the pre-registered 3% relative-logit-L2 gate. In E58, BF16 prefill followed by FP8 / INT4 decode gives 4.06% / 7.22% on the same 100 teacher decode positions across six 4K–16K prefixes. This is not a 256K quality result or a throughput measurement.
- Four-request capacity now reaches 128K per request, sharing serial GPU scratch. This profile defaults to 512-row prefill to retain memory headroom. The 64K profile continues to support 1,024-row prefill. Long GPU batching, fair queues and sustained stress remain open.
- **Native TensorFold comparison**: 45.62 vs native 49.18 tok/s overall in E7; not rerun since.

## Feedback-driven self-improvement (proposal)

[PR #15](https://github.com/EdwinWang831/spark-infer-q/pull/15) proposes learning from usage feedback, evaluating candidate policies on temporally held-out data, and promoting or rolling back them based on measured net benefit. Its v2 plan has **no executed EV0–EV7 stages**. Frozen acceptance prediction, manual engineering gains and adaptive width decisions are not evidence of recursive self-improvement. See the [overview](docs/RESEARCH_OVERVIEW.md#feedback-and-rsi) for the connection to scientific agents and the proposed evaluation loop.

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

# Runs the released E17 predictor and budget rule on synthetic features, using only the standard library.
# No model generation or throughput measurement; see the overview for the input and cost contract.
python3 -m examples.adaptive_budget

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
