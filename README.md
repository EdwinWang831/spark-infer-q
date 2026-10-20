# SparkInfer-Q

**Experimental inference acceleration for Qwen models on NVIDIA DGX Spark (GB10 / SM121), inspired by [DeepInfer](https://github.com/deepseek-ai/DeepInfer).**

> **Research status:** E0/E1 and selected E2 expert probes passed. The original TensorFold packed-weight/BF16-input expert kernel passed 11 SM121 tests, including graph replay; GPU weights remain compressed. Extra FP4 activation quantization remains a separate fidelity tradeoff. DeepGEMM native BF16 still rejects SM121. Full Qwen inference, total runtime capacity and speedups remain unverified. See [native expert results](docs/E2_NATIVE_REPORT.md) and [earlier quantization comparison](docs/E2_EXPERT_REPORT.md).

## Research question

Can we retain DeepInfer's engine, scheduling, speculative verification and memory-management advantages while adapting its model executor and GPU kernels to Qwen3.8-Flash-Next on **a single DGX Spark**? Or does a vLLM / SGLang-based implementation provide a better verified speed-versus-effort tradeoff?

**We will let experiments choose the inference foundation**, rather than assume one from the beginning.

## Investigation stages

| Stage | Goal | Status |
| --- | --- | --- |
| E0 | Hardware/software fingerprint + upstream compatibility audit | PASS_STATIC; no CUDA execution tests |
| E1 | Build DeepInfer core and probe SM121 kernel compatibility independently | Minimal GPU gate passed; native DeepGEMM BF16 blocked; full backend compatibility incomplete |
| E2 | Minimal model adapter, forward pass, prefill and decode | Partial: native BF16-input packed expert and independent CSF checks passed; routing/model-state/full-memory integration incomplete |
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

**Do not vendor upstream source or weights without reviewing the respective licenses and notices.** This repo contains independent planning notes, diagnostic/build/test harnesses and sanitized E0/E1 evidence; it does not contain a Qwen inference implementation or model weights.

License: [Apache 2.0](LICENSE).
