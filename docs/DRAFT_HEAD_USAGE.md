# Optional draft projection / 可选草稿投影

The reusable API is `spark_infer_q.draft_head`. It imports no TensorFold kernel, compiles nothing and uses no machine paths. PyTorch is required when creating a projection; the repository's remaining standard-library tools still work without PyTorch. BF16 is the default. Q4 requires an explicitly supplied backend with its own device/shape checks. The eager `HeadDraft` wrapper remains specific to this repository's bounded Qwen DFlash2 implementation; this API does not turn the full executor into a generic inference server.

可复用接口为 `spark_infer_q.draft_head`，不会自动导入或编译 TensorFold 内核，也没有个人机器路径。创建投影需要 PyTorch。默认保留 BF16；Q4 需要使用者明确提供后端，并由后端检查设备和矩阵形状。`HeadDraft` 包装器仍针对本仓库的有界 Qwen DFlash2 实现，尚非通用服务端。

```python
import torch
from spark_infer_q.draft_head import make_projection, vocabulary_ids, HeadDraft

# target_head: existing BF16 [vocabulary, hidden] tensor; it is never modified.
full = make_projection(target_head)
# token_ids must come from a vocabulary for this exact model/tokenizer.
selected = vocabulary_ids(token_ids, eos_ids, target_head.shape[0])
ids = torch.tensor(selected, device=target_head.device, dtype=torch.int64)
subset = make_projection(target_head, ids=ids)
drafter = HeadDraft.attach(existing_block_drafter, subset)
proposals, diagnostics = drafter.propose(features, anchor_token)
```

Subset IDs are unique, sorted by `vocabulary_ids`, range checked and united with every EOS ID. The projection maps top-k row indices back to global token IDs before calling the selector. Other models need their **own tokenizer-compatible subset**; do not reuse Qwen token IDs, EOS IDs or a language-specific subset blindly. A full-vocabulary path is available for coverage comparison. Reduction can change proposals even without quantization.

子词表合并全部 EOS，检查范围，并在 selector 之前将局部行号还原成全局 token ID。其他模型必须提供匹配 tokenizer 的词表；不能直接套用 Qwen 的 token 编号或 EOS。缩词表本身也会改变提案，需要对照完整词表评估语言与任务覆盖。

An optional provider is explicit:

```python
from spark_infer_q.draft_head import Q4Backend

provider = Q4Backend(
    name="your.tested.backend",
    supported=your_device_and_weight_check,
    quantize=your_weight_only_quantizer,
    matmul=your_bf16_activation_matmul,
)
projection = make_projection(target_head, ids=ids, precision="q4", backend=provider)
# Optional fallback, only when provider is absent/declares unsupported:
projection = make_projection(target_head, precision="q4", backend=provider,
                             unavailable="bf16")
print(projection.requested_precision, projection.effective_precision, projection.backend)
```

The default unsupported-backend policy is an error. Explicit `unavailable="bf16"` emits a warning and records the effective precision. Kernel failures are propagated, never silently hidden. A quantized provider must return an object with `nbytes()` and must implement `matmul(hidden, packed)`; its output is `[rows, selected_vocabulary]` with the chosen numerical contract. No provider is automatically selected by GPU model. `HeadProjection` exposes the lower-level injection API; applications own provider loading and version/license management.

后端不支持时默认报错。只有显式允许 `unavailable="bf16"` 才提示回退，结果包含实际精度和后端名；内核运行错误会直接传出。量化对象须提供 `nbytes()`，矩阵乘实现须保持约定的激活和输出精度。程序不会仅根据显卡名称自动切换未经验证的实现。

## Validation scope / 验证范围

| Path | Requirement | Evidence boundary |
| --- | --- | --- |
| Dense projection | PyTorch linear; same device/dtype as supplied head | CPU adapter tests and E6 Spark real-weight study are separate checks |
| Subset projection | Matching global token IDs plus all EOS | Range/remap tests; E6 uses a fixed external 79,591-ID Qwen subset |
| E6 Q4 provider | External TensorFold group-32 affine Q4, BF16 activation; pinned CUDA/Torch build on SM121 | This one Spark only; no other NVIDIA/AMD/Apple GPU speed or compatibility claim |
| New provider/device | User-supplied implementation and capability check | Needs stored-weight arithmetic, stream/tail tests, proposal/acceptance and full target-state validation |

E6 obtains the fixed vocabulary from TensorFold's externally installed `draft_vocab.txt`; its SHA256 is recorded. E6 receives it through `--draft-vocab`, rather than assuming a personal cache location. The research driver still intentionally uses the pinned E2/E3 external build layout and SM121 loader; it is not a one-command portable model loader. External sources, CUDA binaries and weights are not redistributed here.

E6 通过 `--draft-vocab` 接收外部固定词表并记录 SHA256，不依赖个人缓存目录。实验驱动仍依赖固定 E2/E3 目录和 SM121 加载器；复用投影接口与跨平台部署完整模型是两个不同的完成标准。外部源码、CUDA 二进制和权重均不随仓库分发。

## What “close to BF16” means / “接近 BF16”的含义

E6 compares **draft-head BF16** with compressed alternatives while keeping the already loaded **NVFP4 target** and BF16 DFlash backbone unchanged. Exact greedy outputs/logits/live states are checked against this target's serial path. This does **not** establish equivalence to an entirely BF16 Qwen target. Candidate top-k overlap, proposal agreement and accepted-prefix counts diagnose draft fidelity; speed must be measured after accounting for those changes. Q4 is an experiment, not a promise of BF16-like proposals. Q8 or mixed-precision projection would be a separately measured alternative if Q4 compromises acceptance. Random sampling needs its own correct acceptance/correction implementation and validation.

本轮对照的是**草稿输出头 BF16**，主模型仍为已有 **NVFP4 target**，DFlash backbone 保留 BF16。逐 token/logits/有效状态一致性只针对该 target 的串行路径，不能称作整模 BF16 等价。候选重合率、提案一致率和接受前缀反映草稿损失；速度还要计入这些变化。若 Q4 接受率下降，Q8 或混合精度须另做实测。随机采样的接受/校正尚未验证。

Compression is of an additional draft projection: the target's original BF16 output matrix is still needed. A full BF16 draft reuses that matrix with zero additional weight payload; a compressed copy can make drafting cheaper while **increasing** total residency relative to sharing. `payload_bytes` reports additional payload, not model-wide memory saved. All four E6 variants stay resident during balanced timings; report actual residency/peak separately.

压缩的是额外草稿投影：主模型的 BF16 输出矩阵仍要保留。完整 BF16 草稿直接共享它，额外权重为零；压缩副本可能加速草稿，却增加整机常驻内存。`payload_bytes` 表示新增 payload，不能写成整模节省。E6 轮换计时时四配置共同驻留，另报实际常驻和峰值。
