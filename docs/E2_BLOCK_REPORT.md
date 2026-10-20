# E2：GDN/QSA decoder block 与无 MTP 状态闭环

**两类完整 decoder block 的选定执行和状态检查通过；严格数值等价仍未证明。** 本轮分别运行真实 layer 0（GDN）和 layer 3（QSA），包含四路 hyper-connection 的归一化、门控混合、attention、MoE 和分支写回。16 项用例达到预设数值/状态门槛，但 GDN 的 CPU 全链路参考在 17 行中的 1 行选择了不同的第 10 个专家。这个差异保留为后续数值验证事项，不能把结果表述为整模正确性通过。

最终复测：2026-10-20 08:45:50–08:46:23 America/Los_Angeles。完整数据见 [block JSON](../results/e2/2026-10-20-decoder-blocks.json)，只读内存账本见 [budget JSON](../results/e2/2026-10-20-model-budget.json)。

## 实际执行范围

- Checkpoint revision `b9ce18a6887e18c089fdea8509a33e63429a3300`；固定种子的合成 BF16 四路 hidden states，输入宽度 `4 × 2560`，每层分别使用自己的真实权重。
- 选定的两层没有 n-gram 分支；没有把 layer 0 和 layer 3 当作相邻层串联，没有执行其他 46 层、embedding 或 lm_head。MTP 关闭。
- 复用原版 TensorFold `layer_forward`、hyper-connection、GDN/QSA、MoE、`State`、`Buffers` 和 commit。只替换隔离构建的扩展提供者，没有修改外部 forward 或内核源码。87 个选定外部文件的摘要留在 JSON 中。
- 新增原版 GDN decode/replay、GDN front/back、shared GDN prefill 三个扩展；均以 `sm_121`、原有 `--fmad=false` 编译，`cuobjdump` 确认 SM121 cubin。沿用前轮三个 MoE/MXFP8 扩展。
- 两类 block 的普通投影保留 MXFP8，hyper-connection 投影保留 BF16，路由专家保留压缩 NVFP4。测试依次加载一层，不展开整模权重；CPU 参考按需解码专家。
- 选定两层的 HC norm 原始均值分别约 `-0.06354`、`-0.26380`，符合 centered storage；按原 loader 语义使用 `1 + weight`，GDN gated norm 使用原值。

181,175,744 bytes 的非路由投影/标度等数据按 tensor 范围提取，没有整份读取 shard。layer 3 的 1536 个专家投影 CSF 标度也与独立编译的原版 Zig 解码器逐字节一致；layer 0 沿用前轮已核对的数据。不是完整模型文件哈希验证。

## 16 项联合检查

两层各执行以下 7 项，共 14 项：

| 检查 | 实际覆盖/观察 |
| --- | --- |
| 完整 block CPU 参考，17 行 prefill | CPU FP32 运算及显式 BF16 边界，对照输出、attention 分支、MoE 输入和状态 |
| 分块 prefill→decode | 对照 `[17]` prefill 与 `[5,7]` prefill 后逐 token decode；检查全部输出、最终状态和专家集合 |
| decode window→serial | `[3,7,7]` decode windows 与 17 次单 token 执行的输出及状态逐 bit 一致 |
| snapshot→restore→replay | 保存第 5 行前缀，执行后续及额外行，再恢复重放；所选后续输出及状态逐 bit 一致 |
| 扩容及 reset | capacity 8→24 保留已提交状态；后续输出与足够容量基线一致；reset 后与新状态一致，所检查 rec/conv 地址不变 |
| 两请求不同前缀/行范围 | 前缀长度 3/5，窗口长度 2/4，联合 forward 后分别 commit；与各自独立运行的输出、状态逐 bit 一致 |
| CUDA Graph 单步 forward | 固定已提交前缀 5，三次改变当前输入；与 eager 输出逐 bit 一致，GDN proposed recurrent state 也一致；没有在图内提交状态 |

QSA 另执行两项：

- **真实 2048 稀疏预算：** 预填 2051 行合成 BF16 K/V/index cache，再用真实 block 权重处理两行。原版 GPU pooling 与 CPU 参考核对，实际进入 sparse 分支，分别选择 2048/2049 个 key，后者保留未成块尾行。给定 GPU query/pools 的 CPU key 选择完全一致；独立 CPU 全链路选择也一致。旧 cache 是合成 priming，不能声称完成了 2051-token 模型历史执行。
- **FP8 KV 分块一致性：** 同样比较 17 行整体 prefill 与分块 prefill→decode。输出 L2 差约 0.01817%，已提交 K/V 存储字节相同，index key 有微小差异。这里没有独立 CPU FP8 全 block oracle，也没有验证 FP8 KV 对任务质量的影响。

snapshot/restore 的契约要求保存点之前的 KV prefix 仍在；本轮满足这个前提。没有测试任意 KV 前缀被覆盖后的恢复、取消/slot 复用、padding、图内 commit、图与扩容失效的联动，也没有 speculative rejection/accepted-prefix 测试。

## 数值结论及路由差异

门槛在运行前设定：CPU block 输出 L2 ≤ 0.01，所检查分支/状态 ≤ 0.005；原生 partition 输出 ≤ 0.003，状态 ≤ 0.001。没有为消除差异而放宽门槛。

| 对照 | GDN layer 0 | QSA layer 3 |
| --- | --- | --- |
| CPU 全 block 输出相对 L2 误差 | 0.11932% | 0.12926% |
| CPU attention 分支相对 L2 误差 | 0.15987% | 0.19039% |
| CPU MoE 输入相对 L2 误差 | 0.20039% | 0.30907% |
| 原生分块 prefill→decode 输出 L2 差 | 0.01761% | 0.02066% |
| 原生 prefill/decode 专家集合 | 完全一致，排序也一致 | 完全一致，部分排序不同 |

GDN CPU recurrent state L2 误差约 0.02739%，conv state 约 0.06304%。原生分块比较的 recurrent state L2 差约 `9.34e-7`，conv 约 `3.87e-5`。QSA CPU K/V/index key 误差及完整逐项数据保留在 JSON 中。

**不能忽略的 Top‑10 分歧：** GDN 的第 6 行（零起始）在 CPU 全路径中选中 expert 114，而 GPU 全路径选中 expert 296，其余九个相同。CPU 第 10/11 名 logit 间隔约 `7.74e-5`；上游 MoE 输入数值差异造成该行 router logits 最大变化约 `0.00296`。

隔离诊断表明：

1. 给定同一份 GPU logits，CPU stable Top‑10 与 GPU 选择完全一致。
2. 给 CPU router 相同的 GPU MoE 输入，两层选出的专家集合都一致；router 本身的 L2 差约 `3e-6`。
3. 原生 prefill/decode 切分下，两层专家集合一致；QSA 允许出现排序变化和小的累计舍入差异。

因此，本次观察指向上游数值扰动跨过路由边界，没有发现相同 logits 下的 Top‑10 选择错误。还需要匹配精度的 GPU 基线、真实激活和整模 logits/任务对照，才能评价实际影响。这些百分比不是模型质量下降比例。

首轮自有 CPU QSA 参考把输出 sigmoid 提前舍入为 BF16；源码复核后改为原版 `attn_gate` 所规定的 FP32 sigmoid 乘法、最终 BF16 输出，再运行全部用例。外部内核及门槛没有改动；原始 runner 摘要和审计说明保留在记录中。

## 内存与现有服务

最终测试的 Torch allocated 峰值：GDN **1.4453 GiB**，QSA **1.4207 GiB**。这些是分别加载一层后的张量统计，含该层专家/投影、激活和所测状态；不含 CUDA context、驱动和外部库分配，不是进程全部统一内存，更不是整模峰值。

仍使用 2 CPU、6 GiB RAM 的独立容器，Torch allocator 上限 3 GiB。最终运行的 cgroup peak 记录为 1,382,957,056 bytes，max/OOM/OOM kill 均为 0；cgroup 与 CUDA allocator 的记账范围不同，不能相加或相互替代。

结束后实验容器均停止。现有服务 health 正常、PID/启动时间和重启次数（0）不变，配置/镜像/参数/环境及挂载定义一致。Docker 返回的 Mounts 数组顺序曾变化，核对定义及反转数组摘要后确认仅顺序不同。快照不代表测量过实验期间的瞬时服务延迟。

## 全模型只读账本

扫描全部 index/header，未读取 tensor payload：

| 项目 | GiB | 含义 |
| --- | --- | --- |
| checkpoint tensor payload 合计 | 95.1243 | 不含文件 header 等开销 |
| 主干 routed expert 原始 payload | 59.9441 | 含压缩 CSF、global/input scales 等 |
| 主干 routed expert 一份 native pack | 63.2815 | 前轮已核对的格式公式 |
| n-gram tables 原始 payload | 26.8222 | 是否驻留或流式读取会影响实际内存 |
| 其他主干层权重原始 payload | 4.1863 | 非 routed experts，包含共享专家和投影等 |
| MTP 原始 payload | 1.4871 | loader 重打包可改变实际体积 |
| native 主专家 + 其他原始 payload | 98.4617 | 混合格式账本，**不是实际运行权重体积** |
| 上项排除 MTP | 96.9747 | 仍含 vision、ngram 等；不是实测峰值 |

按所检查 Python `State/KVCache` 几何，12 个 target attention 层在 256K context 使用 FP8 KV，加 index keys/pools，约 **4.03125 GiB/请求**；36 个 GDN 层双 bank recurrent + conv 约 **0.2130 GiB/请求**。5 个各自完整 256K 请求的这两部分约 **21.22 GiB**；一个 MTP attention cache 层若启用，还会增加约 1.68 GiB/5 请求。

该状态公式未计 prefix cache、snapshot、replay scratch、PLE、MTP proposal/其他状态、图池、工作区和运行时开销；也不是现有 Zig 服务的实测分配。完整加载还涉及非专家 padding、转换重叠、CPU 副本和 n-gram 放置。按单层结果不能保证整模容量，`parallel=5` 也不能当作五个完整 256K 请求已经验收。

## 下一里程碑

把已检查的 block/state 接到最小执行器契约：输入行与请求/位置对应、state handle 的身份和代次、padding、取消/复用、buffer/graph 失效和已提交位置，再连接 DeepInfer 调度边界。同时建立匹配精度的 GPU 数值基线，继续核对权重加载和峰值。MTP 的提议/验证与接受前缀状态提交仍放在 target-only 路径正确之后，整体目标继续包含调度、内存、算子和模型移植。

复现：[外部源码准备](../scripts/e2_stage_block_sources.py)、[非专家权重提取](../scripts/e2_extract_blocks.py)、[MoE 提取（新增 `--layer`）](../scripts/e2_extract_moe.py)、[构建](../scripts/e2_build_blocks.py)、[权重映射](../scripts/e2_block_support.py)、[CPU 参考](../scripts/e2_block_reference.py)、[联合用例](../scripts/e2_block_smoke.py)、[只读预算](../scripts/e2_model_budget.py)、[运行入口](../scripts/e2_run_blocks.sh)。第三方源码、权重、二进制和原始日志留在私有实验区。
