# E2：完整真实权重单专家验证

**已完成一个完整专家的局部数值验证，尚未完成 Qwen 模型适配。** CSF 解码和解量化权重与 TensorFold 参考源码一致；完整 NVFP4 计算路径通过核对。但 FP4 激活在本组合成输入上引入约 14.6%–15.4% 的专家输出相对 L2 差异，因此不能把这条路径直接当作 BF16 输入计算的等价替换。

后续已完成保留 BF16 激活、GPU 权重保持压缩的原生专家验证，见 [E2_NATIVE_REPORT.md](E2_NATIVE_REPORT.md)。本报告保留前一轮量化路线与诊断对照的原始范围。

实验时间：2026-10-20 06:58–07:12 左右 America/Los_Angeles，精确 UTC 时间和逐项指标见 [JSON 记录](../results/e2/2026-10-20-single-expert.json)。最终状态：`EXPERT_ARITHMETIC_PASS_ACTIVATION_FIDELITY_UNRESOLVED`。

## 真实权重和独立参考

checkpoint revision 仍为 `b9ce18a6887e18c089fdea8509a33e63429a3300`，config/index 摘要与 E0 一致。本次读取主模型 `model.language_model.layers.0.mlp.experts.0`：

| 投影 | 完整逻辑矩阵 shape | CSF 字节不一致数 | 解量化 FP32 元素不一致数 |
| --- | --- | --- | --- |
| gate | 640 × 2560 | 0 | 0 |
| up | 640 × 2560 | 0 | 0 |
| down | 2560 × 640 | 0 | 0 |

CSF 对照直接编译并执行当前 TensorFold 的原版 `cuda_csf.zig`，与我们 Python 解码器的全部输出字节比较，合计 307,200 个缩放字节一致。原版 codec 自带的两项测试也通过。之后，三个矩阵共 4,915,200 个解量化 FP32 值与 TensorFold 原版 `nvfp4.py` 的 `dequantize` 逐元素一致。

这是对参考源码的独立执行；没有重建整个服务来证明新参考库与正在运行的 TensorFold 二进制等价。外部参考源码和编译产物的 SHA256 均已记录，未将第三方源码或权重字节加入仓库。

## 完整专家计算与两类误差

执行 `BF16 输入 → gate/up → SwiGLU → down → BF16 输出`，输入 shape 为 `[1,2560]`、`[8,2560]`、`[32,2560]`，每组使用固定种子的标准正态合成数据并舍入到 BF16，没有使用生产请求或捕获模型激活。

中间约定：linear 输出 BF16；`BF16(SiLU(BF16 gate)) × BF16 up` 后舍入 BF16，再送入 down；无 clipping。该舍入形式与已检查的 TensorFold SwiGLU helper 一致，但本次没有直接执行 TensorFold 原生完整专家 forward 做逐输出对照。

FP4 激活量化使用 checkpoint 的每投影 `input_scale` 和 TensorFold 的实际 `nvfp4q.cuh`，通过一个小型 CUDA 包装内核执行。12 组输入/投影检查中，packed code 和缩放字节全部与独立 CPU 参考一致，包含零输入、零缩放块、异常大值导致的缩放饱和分支。完整维度的 cuBLASLt NVFP4 支持 FP32 和 BF16 输出执行；本轮完整链路选择 FP32 GEMM 输出，在 GPU 实际乘回输入/权重全局缩放后舍入 BF16。

| Batch | GPU 相对同一量化表示参考的 L2 误差 | 激活量化造成的输出 L2 差异 | 保留 BF16 输入的 GPU 对照误差 |
| --- | --- | --- | --- |
| 1 | 约 0.00000515% | 约 14.632% | 0 |
| 8 | 约 0.000234% | 约 15.357% | 约 0.00209% |
| 32 | 约 0.00339% | 约 14.553% | 约 0.01636% |

第一列比较 GPU 与 CPU 在**相同量化激活、相同存储权重和相同边界舍入**下的完整结果。三个完整用例及 gate/up/down 局部运算均通过预设数值门槛。

第二列只在 CPU 参考中改变是否将 BF16 激活量化到 FP4，权重仍使用同一 checkpoint 的已量化表示；它包含量化误差沿非线性和后续投影的传播。输入本身的 FP4 量化 L2 差异约 9.49%–9.51%。这些数值**不是整模型精度损失、困惑度或任务质量下降比例**，也不是相对原始未量化权重的误差。该组输入仅代表本次合成分布。

第三列是新增的诊断对照：保留 BF16 激活，仅将这一个专家的权重临时解量化到 FP32，在 GPU 做 FP32 矩阵乘法并按上述边界舍入。三组都通过，与第二列共同支持差异主要来自激活量化。这里额外暂存 19,660,800 bytes FP32 权重，**不是已经实现原生 W4A16 内核，也不建议全模型按此方式展开权重**。

所选门槛及实际最大绝对误差、RMSE 均在 JSON 中保留。量化计算路径的门槛检验算术一致性；没有把 14%–15% 的激活差异隐藏在 PASS 后面，也没有给整模型质量设定或通过未经验证的门槛。

此前小切片约 0.17% 的误差同样针对已量化输入下的计算一致性，未衡量 BF16→FP4 激活量化。两个阶段的参考不同，不能直接用百分比高低判断进退。

## 环境、资源和现有服务

- 继续使用隔离环境的 PyTorch `2.13.0+cu132`；编译器 CUDA 13.3，GPU GB10/SM121。
- Zig `0.17.0` ARM64 官方归档与现有 TensorFold recipe 固定的 SHA256 一致。远程下载超过 180 秒后，改从客户端下载同一工件、核对摘要并复制到实验目录；最终编译成功。
- 无 GPU 的实验容器完成参考库和量化包装的构建；GPU 实验容器执行数值测试。均未修改现有服务包或宿主机软件。
- 限制仍为 2 CPU、6 GiB RAM，Torch allocator 512 MiB。完整 checkpoint 没有挂载进容器，只复制一个专家的 packed 权重和解码缩放，共 2,764,800 bytes。
- 提取读取 2,623,284 bytes tensor payload（包含 CSF 流和全局缩放），另读取 58,170,038 bytes index/header 元数据。未校验完整权重文件哈希。
- FP4 完整专家进程 Torch 峰值 allocated/reserved 为 20,732,928 / 20,971,520 bytes；BF16 输入对照为 53,993,472 / 56,623,104 bytes，均不代表包含 context/外部库的进程总 GPU 内存。
- GPU 容器 cgroup 观测 peak 716,464,128 bytes，max/OOM/OOM kill 均为 0。结束后两个实验容器均停止，产物保留。
- 服务前后及最终 health 均正常，PID、启动时间、重启次数（0）、镜像、命令、环境和挂载不变。上游 DeepInfer tracked source 未修改。快照不证明连续可用或零瞬时延迟影响。

## 下一步决策

1. 将保留 BF16 激活的 SM121 专家实现作为优先验证对象，评估复用 TensorFold 原生权重解码/计算内核，并对同一输入直接比较 native expert forward。
2. A4W4 cuBLASLt 路径可作为已通过所选算术验证的候选；进一步使用代表性模型激活和最终 logits/任务评价评估其代价，暂不据此替换当前计算语义。
3. 验证单 GPU top-10 路由、专家加权与共享专家；之后推进完整 Qwen 层和 GDN/QSA 状态，同时按 [接口清单](ADAPTER_INTERFACES.md) 推进引擎及 MTP 接入。

本次没有路由、共享专家、完整 decoder layer、端到端 logits、MTP 或性能比较结果；E2 整体仍未完成。

复现脚本：[提取与 CSF 对照](../scripts/e2_extract_expert.py)、[Zig 包装](../scripts/e2_csf_reference.zig)、[激活量化 CUDA 包装](../scripts/e2_quantize_reference.cu)、[完整量化专家](../scripts/e2_expert_smoke.py)、[BF16 输入对照](../scripts/e2_expert_bf16_control.py)、[隔离环境运行入口](../scripts/e2_run_expert.sh)。外部参考需按记录的内容摘要准备；不运行全部 TensorFold 构建。
