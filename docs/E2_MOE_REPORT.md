# E2：完整 MoE 层联合验证

**layer 0 的完整 MoE 子层在 SM121 上通过 8 项联合数值检查。** 本轮使用真实 checkpoint 的全部 512 个路由专家、BF16 router 和 MXFP8 共享专家，将路由、Top‑10、设备端分组、压缩专家计算和加权汇总一起执行。最大整层输出相对 L2 误差约 **0.02821%**。这里的“完整”指 MoE 子层，不包含 attention、归一化和 residual，也不表示完整 Qwen 或 DeepInfer 接入已完成。

最终 GPU 运行时间为 2026-10-20 15:12:29–15:13:12 UTC；此前进行了构建、首轮测试及舍入诊断。完整数据和首轮失败记录见 [JSON](../results/e2/2026-10-20-moe-layer.json)。此前的[单专家检查](E2_NATIVE_REPORT.md)保留为分项证据。

## 使用的权重和执行路径

- Checkpoint revision：`b9ce18a6887e18c089fdea8509a33e63429a3300`，模型主干 layer 0；hidden 2560、专家中间维度 640、512 个路由专家、每行 Top‑10，加一个 sigmoid 门控共享专家。
- 从 safetensors 按 tensor 范围流式读取，未整份载入 shard。全部 **1536 个路由投影的 CSF 标度**经 Python 和独立编译的原版 Zig 解码器逐字节核对；不是整个模型文件的完整哈希验证。
- 使用原版 TensorFold `cuda/moe.py` 的 BF16 router/Top‑10/combine、`cuda/experts.cu` 的设备分组计划，以及 `cuda/nvfp4/experts.py/.cu` 的 packed-weight/BF16-input 专家计算。
- 共享专家保留 checkpoint 的 MXFP8 格式，使用原版 `Mx8Linear`、`qmmf.cu` 和 BF16 prompt `prompt.cu`。没有将共享专家重新量化成 FP4；没有启用 FP8 激活。
- 使用原版 `nvfp4_moe.moe` 串起常规用例。仅把扩展提供者指向隔离构建的模块，并增加一个规划器薄绑定；17 个选定外部源码文件保持不变，摘要写入记录。以 namespace package 加载所需模块，避免导入无关后端。
- 三个扩展均以 `sm_121` 编译，`cuobjdump` 分别确认包含 SM121 cubin。工具链为独立 PyTorch `2.13.0+cu132`、CUDA 编译器 13.3；构建容器不挂 GPU。

TensorFold 参考来自现有 recipe 的 patched source，base 为 `db281878ddb836fd0df510d8771ecb7e0fe47d26`。本轮确认这些选定源码未修改，没有证明 Python/CUDA 路径与现有服务 Zig 二进制逐 bit 相同。

CPU 数值参考独立计算 router、稳定 Top‑10、softmax/共享 sigmoid、SwiGLU、down 和加权汇总。FP32 解量化权重仅用于 CPU 参考，每次处理一个被选专家；GPU 保持压缩权重。输入为固定种子的合成 BF16 hidden states，尚未采集真实模型激活。

## 联合测试结果

| 用例 | 关键覆盖 | 输出相对 L2 误差 |
| --- | --- | --- |
| batch 1 decode | 真实 router/共享专家/汇总 | 0.01096% |
| batch 17 decode | 输入 row stride 2576、尾部、117 个不同专家 | 0.02493% |
| batch 97 decode | 1067 个 row-slot pairs，跨越 1024 分组阈值；真实路由选中 269 个不同专家 | 0.02626% |
| batch 17 prefill | BF16 slot 输出、MXFP8 BF16 prompt 路径 | 0.02821% |
| batch 33 重复输入 | 集中路由到相同 10 个专家、一个专家跨多个 tile | 0 |
| 零输入 prefill | 相同 logits 按较小专家 ID 打破平局，共享权重 0.5 | 0 |
| 全 512 专家压力测试 | 显式注入路由 logits，64 行覆盖所有专家的真实权重计算 | 0.02758% |
| 整层 CUDA Graph | 换输入后重放，79 个 routed pick slots 改变，结果重新核对 | 0.02561% |

全部用例的专家 ID 与 CPU 参考精确一致；分组成员完整、无重复遗漏，item 范围和输出保护尾行检查通过。所有 512 个专家都被实际执行并比较，但全覆盖来自明确标记的注入 logits 压力用例，不能宣称自然路由样本已经覆盖全部专家。没有运行内存 sanitizer。

整层输出 L2 门槛预设为 0.005（0.5%），hidden 为 0.003，routed/shared slots 为 0.004；本轮实际值低于门槛。使用实际 GPU slot 输出单独检查 combine，最大相对 L2 差约 `3.48e-8`。这些是所选输入下的数值误差，不能解释为模型任务质量变化。

## 首轮失败与舍入诊断

首轮 7/8 通过，batch 97 在“CPU/GPU 路由权重逐 bit 相同”的要求处失败。没有修改内核或删除这个失败记录。

诊断结果：1067 个权重中，2 个与 CPU FP32 router 参考相差 **一个 BF16 编码刻度**；CPU FP64 router 也给出同样两处差异，Top‑10 ID 始终一致。将**相同 GPU FP32 logits**交给 CPU 重算权重后，全部权重逐 bit 一致。差异来自所选 router 的 CPU/GPU 累加路径跨过后续 BF16 舍入边界。

因此把检查分为两层：相同 logits 下权重精确一致；CPU/GPU router 下权重最多相差一个正 BF16 刻度、整体相对 L2 ≤ 0.001。实际权重 L2 差约 `7.94e-5`。整层输出门槛保持不变，最终输出仍使用 CPU router 权重作独立参考。复测 8/8 通过。完整诊断数值、原检查要求、更新原因和两个 runner 摘要均保留在 JSON 中。

这不保证未来接近 Top‑10 边界的任意输入都选择相同专家；真实激活和完整模型 logits 对照仍需继续。

## 内存与现有服务

| 本轮实测项目 | Bytes | 解释 |
| --- | --- | --- |
| 全 512 个 routed experts 持久 GPU pack | 1,415,583,744 | 约 1.3184 GiB，压缩权重及缩放 |
| 共享专家持久 MXFP8 权重/缩放 | 5,068,800 | 约 4.83 MiB |
| BF16 router 及共享门控行 | 2,626,560 | 约 2.50 MiB |
| Torch allocated 峰值 | 1,447,868,928 | 约 1.3484 GiB，包含本次激活/输出等 |
| Torch reserved 峰值 | 1,463,812,096 | 1.3633 GiB，包含缓存池 |

本轮流式 CPU pack 后写入一份完整 GPU pack，未在 GPU 展开整层 FP32/BF16 权重。隔离容器仍为 2 CPU、6 GiB RAM，Torch allocator 上限从单专家的 512 MiB 调整到 3 GiB。容器在首轮、诊断及复测期间 cgroup peak 为 2,549,284,864 bytes，memory max/OOM/OOM kill 计数均为 0。Torch 统计不含 CUDA context、驱动和外部库分配；这些统计不能直接外推整模容量。

实验结束后两个实验容器均停止。现有服务 health 正常、重启次数保持 0，PID/启动时间及配置、镜像、参数、环境、挂载摘要与上一轮结束快照一致。这里是快照检查，没有测量实验期间的瞬时服务延迟或连续可用性。

## 下一里程碑与复现

基础算子阶段结束后，按可验收的集成里程碑推进，保留内部诊断但不再每完成一个小步骤就停下来：

1. **无 MTP 的 Qwen decoder block 与状态执行。** 将 MoE 后端接入代表性的 GDN/QSA 两类层，联合验证归一化、投影、attention、residual、分块 prefill→decode 的输出/状态，以及状态分配回收；补全其他权重格式与加载峰值预算。
2. **最小引擎闭环。** 串起请求/slot 生命周期、padding、取消与复用、CUDA Graph 和 target-only logits 对照，再测整模运行容量。已通过的 MoE 不替代这些验证。
3. **原生 MTP 与性能。** 在 target-only 正确基础上联合验证提议、验证、接受前缀和 GDN/QSA 状态提交/恢复，再比较固定 K、自适应预算和引擎优化的独立收益。MTP 与调度、缓存、算子移植都属于整体目标。

复现入口：[源码准备](../scripts/e2_stage_moe_sources.py)、[流式权重提取](../scripts/e2_extract_moe.py)、[构建](../scripts/e2_build_moe.py)、[薄绑定](../scripts/e2_moe_plan_bindings.cpp)、[联合用例](../scripts/e2_moe_smoke.py)、[舍入诊断](../scripts/e2_moe_router_probe.py)、[运行脚本](../scripts/e2_run_moe.sh)。需要已有隔离工具链、记录摘要对应的外部源码和独立 Zig CSF decoder；不含第三方源码、权重或二进制。

E2 整体仍未完成；没有整模 logits、质量、MTP、总运行容量或加速比结论。
