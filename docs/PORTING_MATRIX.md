# DeepInfer -> Qwen / DGX Spark porting matrix

**E0 static audit, E1 minimal GPU probes and selected E2 expert, MoE and GDN/QSA decoder/state probes completed. Full model execution remains untested.** See [E0](E0_REPORT.md), [E1 GPU results](E1_GPU_REPORT.md) and [E2 decoder/state results](E2_BLOCK_REPORT.md).

Pinned DeepInfer revision: `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` (October 20, 2026). Re-audit these constraints if this revision changes.

| Boundary | Confirmed DeepInfer assumption | Test/porting need | Status |
| --- | --- | --- | --- |
| Model registry | `_MODELS` registers only `DeepseekV41ForCausalLM` | Add Qwen architecture adapter | Source blocker |
| HF config parser | Requires `model_type == deepseek_v41`, top-level BF16 dtype and `num_key_value_heads == 1`; observed Qwen uses `qwen4_exp`, nested BF16 dtype and two KV heads | New parser and weight mapper; do not misreport nested BF16 as a non-BF16 model | Source blocker |
| Quantization | DeepInfer requires FP8/MXFP4 expert schema, `[32,32]` block size and `ue8m0`; observed checkpoint uses ModelOpt mixed NVFP4/MXFP8/W4A16_NVFP4 | Complete expert CSF/weight decoding matched reference source; selected native packed-weight/BF16-input expert cases passed; extra FP4 activation quantization has a separate measured fidelity cost | Native complete MoE and shared MXFP8 selected paths verified; other formats/full loader remain incomplete |
| Model loader | Requires a `vl_aligner` config | Decouple text/multimodal logic where needed | Source blocker |
| Speculative head | DFlash-only check in `models/mtp/dflash.py`; observed Qwen config declares one native MTP layer under different field names | Adapt native Qwen MTP and verification metadata; distinguish layer count from configured draft depth | Requires redesign |
| Cache layout | V4.1 sparse/SWA/YoCo/index and NextN-specific allocation; Qwen config has 36 linear-attention and 12 full-attention layers | Selected original TensorFold GDN/QSA block state/partition/restore/growth checks met gates; bind these to independent DeepInfer request/state handles | Selected native state checks verified; DeepInfer adapter unimplemented |
| Scheduler | C++ scheduling/blocks and async path are present; E1 core compilation/import and 14 sequence-interface tests passed | Full scheduler execution and generalized metadata contracts remain untested | Core build verified; scheduler semantics unknown |
| SM121 | GB10 capability 12.1; original TileLang RMSNorm and cuBLASLt probes execute correctly; native DeepGEMM BF16 dispatch rejects architecture 12 | Replace/port unsupported native paths; selected CUDA Graph replay passed | Minimal GPU gate passed; native backend blocker observed |
| Vendor architectures | DeepSelect and FlashMLA setup scripts explicitly target SM100a/SM103a; selected native DeepGEMM FP4/layout guards require major 10 | Architecture-specific redesign and numerical tests; changing TORCH_CUDA_ARCH_LIST alone is insufficient | Source constraints verified; DeepSelect/FlashMLA not executed |
| Alternative expert backend | Unmodified TensorFold expert source compiled into an SM121 cubin and passed 11 functional probes with compressed weights/BF16 inputs | Complete layer-0 routing/top-10/shared MXFP8/combine passed 8 joint probes; selected decoder/state checks also met gates, with CPU numerical route divergence retained; next executor and matched GPU baseline | Selected native blocks verified; DeepGEMM's native architecture guard remains |
| Runtime capacity | Single-expert packed weights: 2,764,812 bytes; all main routed expert packs: metadata budget 63.2815 GiB | All checkpoint payload categories and source-derived KV/GDN scenarios recorded; selected one-block Torch peaks about 1.42–1.45 GiB; measure full loading/graphs/duplicates next | Metadata accounting plus component peaks; full-model fit unconfirmed |
| Toolchain | Core built in two Torch environments; selected GPU probes passed in clean PyTorch 2.13.0+cu132 with TileLang 0.1.15 / TileKernels 2.1.0; compiler CUDA 13.3 | Full inference/vendor dependency stack still unverified | Core and selected CPU/GPU interfaces verified |
| Multi-GPU EP | Engine supports MoE expert parallelism and EPLB | Determine which code paths can be disabled on one GB10 | Not tested |

## Primary code links

- [registry](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/__init__.py)
- [config parser](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/hugging_face/config.py)
- [model loader](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_config_loader.py)
- [DFlash](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/mtp/dflash.py)
- [scheduler](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/scheduler.hpp)
- [worker](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py)

**Interpretation:** direct Qwen loading is ruled out by explicit guards. Retaining the full DeepInfer core is an experimental question, not ruled out and not yet demonstrated.
