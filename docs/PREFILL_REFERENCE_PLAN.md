# 提议：用高精度参考决定 prompt 数值路径

状态：**PROPOSED，尚未执行，没有新的 GPU 结果。** 实际排期时再分配 E 编号，执行前按本仓库惯例冻结输入、阈值和计时方法。所有提议的状态汇总在 [EXPERIMENTS.md](EXPERIMENTS.md#proposals-without-results-yet)。

## 动机

- E8 保留 decode 算术的 8 行分块 prompt 为 **109.23 prompt tok/s**；[TensorFold 审计](E4_TENSORFOLD_COMPARISON.md)记录的对方公开 sparkDash 4k prefill 约 **2665 tok/s**（作者自报，未在本机复现）。按 E8 短 prompt 实测速率线性外推，4k prompt 约 37 秒，对方约 1.5 秒。外推未验证，但长文档、多 Agent 场景下 prompt 路径显然是首要差距。
- 当前不用原生 prefill，是因为 [E2 完整加载](E2_LOAD_REPORT.md)中 prefill/serial 末行 logits 相对 L2 差 **10.68136%**：layer 0 已有 0.19118%，layer 6 达 2.51921% 并首次出现专家集合变化，15/48 层专家集合不同。[E3 followup](E3_FOLLOWUP_REPORT.md) 的 F32 MoE slot 控制没有改善（10.69269%）。
- 但 serial decode 算术本身不是真值。两条路径都是对同一份存储权重的近似；“与 serial 逐 bit 一致”保证的是投机解码的同路径正确性，回答不了哪条 prompt 路径更接近模型本身。需要一个独立于两者的参考。

## 参考的定义

- 同一 NVIDIA FP4 checkpoint revision `b9ce18a6887e18c089fdea8509a33e63429a3300`，按存储格式精确解量化（NVFP4 / MXFP8 / BF16），全部 FP32 计算，TF32 关闭。路由、GDN recurrence/conv、QSA attention 与索引、PLE/n-gram、hyper-connection、输出头均为 FP32。
- 以 [E2 执行器](E2_EXECUTOR_REPORT.md)的 CUDA 参考为起点（专家按需逐个解码），扩展为 48 层整模。它与现有 CPU NVFP4 解码器共用格式实现，不是完全独立的格式实现。
- 参考衡量的是“同一量化模型的高精度计算”，**不是** NVFP4 相对原始模型的量化损失，也不是任务质量上限。
- 资源：不与 69.85 GiB 快速 target 同驻，单独 session 从 checkpoint 按需读取。每个位置只保存 top-64 logits、logsumexp 和 argmax；另对抽样位置保存完整 248,320 维 logits，用于校核截断 KL 的近似误差。

## 对照路径

| 路径 | 说明 |
| --- | --- |
| A | serial decode 算术（rows1），即现行 oracle |
| B | rows8 decode 算术分块；E8 已证明与 A 逐 bit 一致，只抽查一次 |
| C | TensorFold Python 原生 prefill，即产生 10.68% 差异的路径 |
| D | 原生 Zig prefill；当前 API 不返回 logits，取不到就明确记为缺失 |

## 输入

新冻结提示，不复用 E3/E6/E7 任务：中英文、代码、JSON、数学、长文档摘录；长度分 64 / 512 / 2k / 4k 桶。当前 capacity 为 1024，必须先扩到覆盖 4k，并单独检查状态与容量，不能假设足够。E2 的四个 synthetic token 保留为回归。

## 指标（逐位置 teacher-forced）

- 与参考的 top-1 一致率；KL(参考‖路径)（top-64 加尾部质量近似，抽样位置用完整 KL 校核）；logits 相对 L2。
- 每层 Top-10 专家集合与参考不同的比例；末层 hidden 相对 L2。
- 全部按位置和长度分桶报告，不只看末行。

## 事前决策规则（执行前冻结；下列阈值为初始建议）

1. C 在每个长度桶上，top-1 一致率不低于 A 超过 0.5 个百分点、中位 KL 不超过 A 的 1.25 倍，且没有随长度累积恶化 → C 可作为 prompt 路径进入下一步。
2. C 明显更差 → 保持 rows8，并对照**参考**（而不是 A）逐层定位 C 的问题算子。
3. A 也明显偏离参考 → 记录为两条路径共有的数值问题，不再把 A 作为“正确性”基准表述。

## 切换后的正确性契约

采用 C 后，输出 IDs 会不同于 E7/E8，旧 IDs 回归不再适用。oracle 改为：**从同一 prefill 状态出发**，投机 decode 与 serial decode 的输出、已提交 logits 和有效 GDN/QSA/PLE 状态逐 bit 一致。另检查 prefill→decode 的状态交接（格式、位置、pending token）和客观任务检查。

## 性能

TTFT 与完整请求时间：rows8 对比 C，在 512 / 2k / 4k 上，预热与 ABBA 方法同 E7/E8，并在同轮重测原生 TensorFold。

## 边界

不改 checkpoint 与外部内核；只覆盖 greedy；参考是数值控制，不是质量评测，任务质量仍需另设评测集；不涉及随机采样、多请求或 vision。
