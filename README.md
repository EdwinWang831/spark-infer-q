# SparkInfer-Q

**Experimental inference acceleration for Qwen models on NVIDIA DGX Spark (GB10 / SM121), inspired by [DeepInfer](https://github.com/deepseek-ai/DeepInfer).**

> **Research status:** The complete 48-layer target has loaded on SM121: 24,576 routed experts plus projections, embedding/head and PLE. Target CUDA weight storage is 69.8511 GiB; the 26.8222 GiB n-gram table stays host-mapped. MTP/vision weights are retained in their original representation without executing those models. A selected complete-target token/logit smoke ran, but prefill versus serial logits differ by about 10.68%, so full-model correctness remains under review. Earlier block/executor graph gains are local measurements. DeepInfer scheduler integration, model quality, long-context capacity and end-to-end speedups remain unverified. See [complete loading results](docs/E2_LOAD_REPORT.md) and [executor results](docs/E2_EXECUTOR_REPORT.md).

## Research question

Can we retain DeepInfer's engine, scheduling, speculative verification and memory-management advantages while adapting its model executor and GPU kernels to Qwen3.8-Flash-Next on **a single DGX Spark**? Or does a vLLM / SGLang-based implementation provide a better verified speed-versus-effort tradeoff?

**We will let experiments choose the inference foundation**, rather than assume one from the beginning.

## Investigation stages

| Stage | Goal | Status |
| --- | --- | --- |
| E0 | Hardware/software fingerprint + upstream compatibility audit | PASS_STATIC; no CUDA execution tests |
| E1 | Build DeepInfer core and probe SM121 kernel compatibility independently | Minimal GPU gate passed; native DeepGEMM BF16 blocked; full backend compatibility incomplete |
| E2 | Minimal model adapter, forward pass, prefill and decode | Partial: full target weights loaded and selected token/logit execution ran; prefill/serial numerical review, independent full-model reference and DeepInfer scheduler integration remain open |
| E3 | Native MTP, speculative verification, GDN/QSA state correctness | Not executed |
| E4 | Apples-to-apples comparison against vLLM / SGLang | Not executed |

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
- **Baselines:** retain the observed TensorFold deployment as the existing-service baseline. Separately establish vLLM and SGLang baselines on the same Spark, input set, sampling settings and checkpoint format; their availability/performance must be verified rather than assumed.

## Upstreams and attribution

- [DeepSeek DeepInfer](https://github.com/deepseek-ai/DeepInfer) (Apache-2.0)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

**Do not vendor upstream source or weights without reviewing the respective licenses and notices.** This repo contains planning notes, diagnostic/build/test harnesses, a bounded experimental request executor and an adapter to externally supplied TensorFold blocks, plus sanitized evidence. It does not vendor a complete Qwen model implementation, third-party kernels or model weights; the full loading/smoke harness uses externally supplied TensorFold source and checkpoint files.

License: [Apache 2.0](LICENSE).
