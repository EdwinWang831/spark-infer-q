# DeepInfer -> Qwen / DGX Spark porting matrix

**Static source review only. Hardware and model execution have not been tested.**

Snapshot examined: DeepInfer main, October 20, 2026. Pin an exact commit before running E1.

| Boundary | Confirmed DeepInfer assumption | Test/porting need | Status |
| --- | --- | --- | --- |
| Model registry | `_MODELS` registers only `DeepseekV41ForCausalLM` | Add Qwen architecture adapter | Source blocker |
| HF config parser | Requires `model_type == deepseek_v41` and DeepSeek-specific quantization/field layout | New parser and weight mapper | Source blocker |
| Model loader | Requires a `vl_aligner` config | Decouple text/multimodal logic where needed | Source blocker |
| Speculative head | DFlash-only check in `models/mtp/dflash.py` | Adapt native Qwen MTP and verification metadata | Requires redesign |
| Cache layout | V4.1 sparse/SWA/YoCo/index and NextN-specific allocation | Map Qwen QSA KV + GDN recurrent state | Requires proof |
| Scheduler | C++ scheduling/blocks and async path are present | Test whether their metadata contracts generalize | Unknown |
| SM121 | README officially lists NVIDIA SM100 and Ascend 950 | Isolated kernel compilation and numerical tests | Not tested |
| Multi-GPU EP | Engine supports MoE expert parallelism and EPLB | Determine which code paths can be disabled on one GB10 | Not tested |

## Primary code links

- [registry](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/models/__init__.py)
- [config parser](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/hugging_face/config.py)
- [model loader](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_config_loader.py)
- [DFlash](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/models/mtp/dflash.py)
- [scheduler](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/scheduler.hpp)
- [worker](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/worker/worker.py)

**Interpretation:** direct Qwen loading is ruled out by explicit guards. Retaining the full DeepInfer core is an experimental question, not ruled out and not yet demonstrated.
