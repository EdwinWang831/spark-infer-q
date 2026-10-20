# E2：保留 BF16 激活的原生专家内核

**原生单专家验证通过。** TensorFold 的原版 NVFP4 专家 CUDA 源码以 `sm_121` 编译后，能够在 GB10 上直接读取 BF16 激活和压缩权重，完成 gate/up/SwiGLU/down。11 项测试全部通过，没有把完整权重解量化到 GPU FP32/BF16 矩阵。

这是既有 CUDA 实现的 SM121 编译与执行验证；没有声称已为 SM121 重新设计或调优内核，也没有完成 DeepInfer/Qwen 引擎集成。

实验区间：2026-10-20 07:22–07:31 America/Los_Angeles。精确时间、误差、源码/二进制摘要和服务检查见 [JSON 记录](../results/e2/2026-10-20-native-expert.json)。前一轮对照见 [E2_EXPERT_REPORT.md](E2_EXPERT_REPORT.md)。

## 执行路径与参考

沿用已独立核对的 layer 0 expert 0 真实权重，checkpoint revision 为 `b9ce18a6887e18c089fdea8509a33e63429a3300`。

- 使用未修改的 TensorFold `cuda/nvfp4/experts.cu`、`cuda/experts.cuh` 和原版 Python pack/gate_up/down wrappers。
- 只新增一个薄 C++ 绑定，将原版 `nvfp4_experts_cuda` 暴露给 Python；实验中替换 `linear._ext` 的扩展提供者，未改第三方源码。
- 构建只编译绑定和该专家内核，未构建完整 TensorFold 扩展。`cuobjdump` 实际列出 `experts.sm_121.cubin`。
- 使用手工构造的单专家分组计划，反转 members 行映射；没有运行 router 或 top-10 选择。
- CPU 用原版 `dense` 检查原生 pack 还原值，并与此前独立解量化参考比较；三类矩阵均通过 `rtol=2e-7, atol=0` 的布局数值检查。
- 数值参考只在 CPU 保留 FP32 解量化权重。GPU 只持有 native packed blocks、缩放和激活；内核内部按块解码并使用 BF16 MMA、FP32 累加和缩放。

工具链仍为独立 PyTorch `2.13.0+cu132`、CUDA 编译器 13.3。构建在无 GPU 容器中完成，导入后 CUDA 初始化仍为 false；随后在独立 GPU 容器实际执行。

## 11 项测试

| 测试 | 覆盖 | 结果 |
| --- | --- | --- |
| 完整专家，8 项 | batch 1/8/17/32 × down 输出 BF16/FP32 | 全部 PASS |
| 零输入 | BF16 激活，FP32 输出 | PASS，输出与零参考一致 |
| 跳过专家 | skip=0，gate/up 和 down 均不修改输出 | PASS |
| CUDA Graph | batch 8 完整 gate/up/SwiGLU/down，捕获后改变输入再重放 | PASS |

batch 17 同时覆盖 16 行 tile 的尾部和输入 row stride 2576（有效宽度 2560）。完整用例检查输出前后保护行不变；分别比较 hidden、完整输出及使用实际 hidden 的独立 down 计算。没有运行内存 sanitizer，因此不把保护行检查当成所有内存访问均正确的证明。

固定种子合成 BF16 输入、完整矩阵尺寸和边界舍入与上一轮参考一致。BF16 输出的完整专家相对 L2 误差：

| Batch | 相对 L2 误差 | 最大绝对误差 |
| --- | --- | --- |
| 1 | 0 | 0 |
| 8 | 约 0.03780% | 0.0009765625 |
| 17 | 约 0.01121% | 0.0009765625 |
| 32 | 约 0.01839% | 0.001953125 |

FP32 down 输出的完整误差最高约 0.00927%；使用实际 hidden 独立核对 down 时，FP32 相对误差最高约 `4.44e-7`。图重放的完整输出相对 L2 误差约 0.000547%。预设门槛是完整结果 L2 ≤ 0.003、hidden L2 ≤ 0.002，并要求逐元素 `atol=0.005, rtol=0.01`；实际值均保留在 JSON 中。

这些差异来自所选累加/缩放/非线性/舍入路径的数值对照，不能宣称与 CPU 或部署二进制逐 bit 一致。此前 FP4 激活路线的 14.6%–15.4% 差异是在额外量化激活后测得；本轮保留 BF16 输入，不能将两个比例解释为模型任务质量的提升倍数。

## 内存：保留 BF16 激活是否还能放下模型

原生路径不会将整份 NVFP4 权重展开成 BF16：

| 实测单专家项目 | Bytes | 说明 |
| --- | --- | --- |
| GPU 持久压缩权重和缩放 | 2,764,812 | 约 2.64 MiB，packed blocks + 三个 FP32 全局缩放 |
| 对应 Torch 分配增量 | 2,765,824 | 包含 allocator 对齐 |
| 整个测试进程 Torch 张量峰值 allocated | 3,477,504 | 约 3.32 MiB，包含所选激活/输出等 |
| Torch 峰值 reserved | 23,068,672 | 22 MiB，包含缓存池 |
| FP32 解量化参考权重 | 19,660,800 | 仅在 CPU，用于核对，没有上传为 GPU 权重 |

Torch 统计不包含 CUDA context、驱动和外部库内存；不能把 3.32 MiB 当作新增进程总 GPU 内存。

另用只读 index/header 扫描核对全部主模型 routed experts：48 层 × 512 个专家，合计 24,576 个专家、73,728 个投影。按原版 native block 格式保留**一份** packed 权重和缩放，预算为：

**67,948,019,712 bytes ≈ 63.2815 GiB。**

该预算来自真实矩阵 shape 和原生 pack 的 32×32 tile/144 int32 几何，不是 BF16 权重体积；扫描未读取 tensor payload。它仅覆盖主模型 routed experts，**不含**共享专家、普通投影、embedding/ngram、vision、MTP、KV/GDN/conv state、图缓存、工作区和运行时开销。也假设没有长期保留重复权重；DGX 统一内存中的 CPU 副本同样占容量，加载器还需要控制转换峰值。

因此，本轮确认了这一内核保持压缩权重的内存结构，没有因 BF16 激活而展开整份模型。现有服务已能运行同一 checkpoint 是部署基线；适配后整套模型的峰值、可用上下文/并发仍未测量，尚不能保证所有配置都能放下。

## 服务、边界与下一步

本轮继续使用 2 CPU、6 GiB RAM 的独立容器，Torch allocator 上限 512 MiB。GPU 容器 cgroup 观测 peak 623,505,408 bytes，max/OOM/OOM kill 均为 0。源码与已提取微小权重保留在私有实验区，未挂载完整 checkpoint 进入容器。

结束后构建/GPU 实验容器均停止。服务前后及最终 health 正常，PID、启动时间、重启次数（0）、镜像、命令、环境和挂载不变；DeepInfer tracked source 未修改。快照不量化瞬时延迟或证明连续可用。

**下一步：** 将已验证调用封装为可复用的 SM121 专家后端，验证真实路由、top-10 加权汇总、共享专家和混合请求布局，再接入 Qwen 层/状态接口。继续核算其他权重及 KV/GDN/MTP/图缓存的内存预算。引擎和 MTP 按 [接口清单](ADAPTER_INTERFACES.md) 推进。

本次没有完整 MoE、全模型加载/logits、真实模型激活、MTP 正确性、吞吐或加速比结果。E2 整体仍未完成。

复现入口：[薄绑定](../scripts/e2_native_bindings.cpp)、[构建器](../scripts/e2_build_native.py)、[原生用例](../scripts/e2_native_expert_smoke.py)、[专家存储预算](../scripts/e2_native_storage.py)、[运行脚本](../scripts/e2_run_native.sh)。需先准备记录摘要一致的外部源码和前轮已核对的专家数据；第三方源码、权重和二进制没有加入本仓库。
