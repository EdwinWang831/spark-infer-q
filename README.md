# SparkInfer-Q

**English** | [简体中文](README.zh-CN.md)

**Experimental inference acceleration for Qwen models on NVIDIA DGX Spark (GB10 / SM121), inspired by [DeepInfer](https://github.com/deepseek-ai/DeepInfer).**

> **Research status:** A same-machine TensorFold native MTP / DFlash2 study now contains 336 timed runs on eight new English prompt texts (14 settings). Within our path, frozen DeepInfer dynamic budgeting measured 47.37 decode tok/s versus fixed6/fixed7 at 46.39/46.30 (+2.12%/+2.31%). Cross-engine outputs differ on six settings, and the original native timings show a first-repetition effect with asymmetric warmup; a separate native per-case warmup control reached **55.99 tok/s**, alongside the untouched main data. The control is post hoc and sequential, with cross-engine output differences retained; no speed lead over native TensorFold is established. Our cold prompt path remains a major end-to-end bottleneck; target verification accounts for about 80% of dynamic decode, the entire drafter about 18%. See the [matched report](docs/E4_MATCHED_REPORT.md), [method](docs/E4_MATCHED_PLAN.md) and [full data](results/e4/2026-10-20-matched.json).
>
> Candidate timed output IDs matched its own serial oracle; 84 separate nonserial runs checked committed logits and final live state. All 240 objectively checkable main-study answers passed. Prose has no semantic quality score, and capped code is only a prefix. Official draft BF16 fidelity, target prefill numerical differences, long context and full-engine validation remain open. Earlier [wide-block](docs/E3_WIDE_BLOCK_REPORT.md), [local-policy](docs/E3_LOCAL_POLICY_REPORT.md) and [draft-fastpath](docs/E3_DRAFT_FASTPATH_REPORT.md) gains are separate studies and must not be multiplied.

## Research question

Can we retain DeepInfer's engine, scheduling, speculative verification and memory-management advantages while adapting its model executor and GPU kernels to Qwen3.8-Flash-Next on **a single DGX Spark**? Or does a vLLM / SGLang-based implementation provide a better verified speed-versus-effort tradeoff?

**We will let experiments choose the inference foundation**, rather than assume one from the beginning.

## Current implementation

The current prototype combines **TensorFold model execution**, an adapted **DFlash2 proposer**, and **DeepInfer's original MTPVarlen verification budget**. Our Python execution loop connects target features, draft proposals, accepted-prefix state commits and experimental local policies. The target CUDA kernels and GDN/QSA commit/replay operations come from TensorFold.

The complete DeepInfer engine is not integrated. E3/E4 speeds describe the DFlash2 experimental execution loop; they are not benchmarks of the separate scheduler bridge below or of a full DeepInfer engine.

A separate [bounded native-scheduler bridge](docs/E5_ENGINE_BRIDGE_REPORT.md) now lets the real DeepInfer C++ Scheduler drive the complete Qwen target-only path. Seven GPU scenario groups passed, with 364 full-logit rows and 79 live-state checks. Device state remains owned by the Qwen backend; native blocks are logical accounting. GPU requests execute serially, and this is not the full Worker/async pipeline or a throughput result. The next scoped experiment is DFlash2 draft-head profiling and compression.

## Investigation stages

| Stage | Goal | Status |
| --- | --- | --- |
| E0 | Hardware/software fingerprint + upstream compatibility audit | PASS_STATIC; no CUDA execution tests |
| E1 | Build DeepInfer core and probe SM121 kernel compatibility independently | Minimal GPU gate passed; native DeepGEMM BF16 blocked; full backend compatibility incomplete |
| E2 | Minimal model adapter, forward pass, prefill and decode | Partial: full target weights loaded and selected token/logit execution ran; prefill/serial numerical review, independent full-model reference and complete DeepInfer engine integration remain open |
| E3 | Draft adapters, speculative verification, GDN/QSA state correctness | Bounded greedy DFlash2 integration passed with native DeepInfer variable budget; random sampling, full scheduler and native Qwen MTP remain open |
| E4 | Apples-to-apples engine comparison, including TensorFold / vLLM / SGLang | Partial: [short-context TensorFold / DFlash2 measurements](docs/E4_MATCHED_REPORT.md) completed; warmup/numerical limits retained; vLLM / SGLang, long context and concurrency not measured |
| E5 | Minimal native engine bridge | Bounded target-only C++ Scheduler / full-Qwen lifecycle checks passed; full Worker/cache/async and GPU batching remain open |

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
