# DeepInfer -> Qwen / DGX Spark porting matrix

**Static source review only. Hardware and model execution have not been tested.**

Snapshot examined: DeepInfer main at commit `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` (October 20, 2026). Re-check these guards if E1 uses a later commit.

| Boundary | Confirmed DeepInfer assumption | Test/porting need | Status |
| --- | --- | --- | --- |
| Model registry | `_MODELS` registers only `DeepseekV41ForCausalLM` | Add Qwen architecture adapter | Source blocker |
| HF config parser | Requires `model_type == deepseek_v41`, top-level `dtype == bfloat16`, a `text_config` with `hidden_act=silu`, no attention bias/dropout, and quantization `fp8` with `fp4` experts, `[32, 32]` weight blocks, `ue8m0` scales and dynamic activations | New parser and weight mapper | Source blocker |
| Attention / KV heads | Parser and `worker/cache_engine.py` (two asserts) require `num_key_value_heads == 1`; attention is the V4.1 MLA path | Qwen GQA uses several KV heads: replace attention kernels and KV cache allocation, not only the guard | Requires redesign |
| Model loader | Requires a `vl_aligner` config | Decouple text/multimodal logic where needed | Source blocker |
| Speculative head | DFlash-only check in `models/mtp/dflash.py` | Adapt native Qwen MTP and verification metadata | Requires redesign |
| Cache layout | V4.1 sparse/SWA/YoCo/index and NextN-specific allocation | Map Qwen QSA KV + GDN recurrent state | Requires proof |
| Scheduler | C++ scheduling/blocks and async path are present | Test whether their metadata contracts generalize | Unknown |
| Hardware detection | `deep_cluster.hardware.get_bf16_tflops_limit()` raises `RuntimeError("unknown GPU type")` unless Ascend or compute capability major 10; `LLMEngine.__init__` calls it unconditionally for stats logging. `is_sm_major_10()` also gates a ViT BF16 path | Add an SM12x branch (GB10 peak BF16 TFLOPS) or make the stats limit optional; audit every `is_sm_major_10()` call site | Source blocker |
| SM121 kernels | README officially lists NVIDIA SM100 and Ascend 950 | Isolated kernel compilation and numerical tests | Not tested |
| Toolchain | `requirements-nvidia.txt` pins `torch==2.13.0+cu132` and `flash-attn-4==4.0.0b34` | Confirm aarch64 (Grace) wheels or source builds exist for these pins | Not tested |
| Multi-GPU EP | Engine supports MoE expert parallelism and EPLB | Determine which code paths can be disabled on one GB10 | Not tested |

## Primary code links

- [registry](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/models/__init__.py)
- [config parser](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/hugging_face/config.py)
- [model loader](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_config_loader.py)
- [DFlash](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/models/mtp/dflash.py)
- [scheduler](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/scheduler.hpp)
- [worker](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/worker/worker.py)
- [cache engine](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/worker/cache_engine.py)
- [hardware detection](https://github.com/deepseek-ai/DeepInfer/blob/main/deep_cluster/hardware/device.py)
- [engine init](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/engine/llm_engine.py)

`scripts/inspect_hf_config.py` mirrors the parser guards above so a local checkpoint can be checked without importing DeepInfer.

**Interpretation:** direct Qwen loading is ruled out by explicit guards. Running on GB10 is also blocked before any kernel executes, by the hardware-detection guard. Retaining the full DeepInfer core is an experimental question, not ruled out and not yet demonstrated.
