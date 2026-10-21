# 提议：BF16 草稿算术偏差是否降低接受率

状态：**PROPOSED，尚未执行，没有新的 GPU 结果。** 实际排期时再分配 E 编号，执行前冻结提示、阈值和分析方法。

## 动机

- decode 每步提交的 token 数取决于接受率。E6 固定策略的“接受数 / 观测草稿数”只有 37.85%（BF16-full）和 39.51%（Q4-full）。
- 当前 BF16 草稿实现与官方 HF 分步算术之间的差异一直保留着，但从未单独测过它对接受率的影响：
  - [E3 followup](E3_FOLLOWUP_REPORT.md)：三例 logits 相对 L2 为 4.52% / 1.60% / 10.78%，其中两例 proposal 不同；差异从 context 投影和归一化开始（hidden 约 0.28%），之后逐步累积。
  - [E3 宽 block](E3_WIDE_BLOCK_REPORT.md)：八行草稿 BF16 的 logits 为 4.32% / 2.00% / 7.96%，synthetic 例 proposal 不同。
  - 两份报告都写明“可能影响接受率，但尚未单独做 A/B 归因”。
- FP32 草稿与官方实现的 proposal 全部一致。如果它的接受率明显更高，对齐 BF16 的舍入位置可能不增加成本就提高 decode 速度；如果没有差别，这个问题在性能上可以关闭。

## 方法

greedy 下，一次 serial target 运行就固定了整条轨迹和每个位置捕获的五层特征。各草稿变体在每个 serial 位置只需各自给出 proposal，再用 [`step_outcome`](../spark_infer_q/draft_fidelity.py) 对照 serial 续写，就能得到本步输出数、接受数和前缀匹配长度，不必为每个变体重跑 target 验证。这与 [E3 局部策略](E3_LOCAL_POLICY_PLAN.md)收集标签的方式相同。

| 变体 | 说明 |
| --- | --- |
| A | 当前 BF16 适配器（SGLang 风格的融合 residual/norm 边界），即现行默认路径 |
| B | 官方 HF 分步算术，BF16 |
| C | 官方完整模型，FP32，作为数学控制，不能直接部署 |

主比较使用当前默认的完整词表 Q4 草稿头；另做一组 BF16-full 头，用来区分草稿头和 backbone 的影响。target、特征、anchor 和 selector 规则对各变体完全相同。

- **输入**：新冻结提示，覆盖中英文、代码、JSON、数学和散文，不复用 E3/E6/E7 任务；旧任务只作回归，单独报告。
- **指标**：每步平均输出 token 数（主指标）；按深度的条件接受率（`leading_matches` 输入现有的 `conditional_prior`）；与 C 的 proposal 一致率。
- **统计**：用 [`paired_ratio`](../spark_infer_q/draft_fidelity.py) 按提示整体重抽样做 bootstrap，给出比值和区间。同一提示内的位置相关，不能把位置当作独立样本。

## 事前决策规则（执行前冻结；下列阈值为初始建议）

1. B 或 C 相对 A 的每步输出比值不低于 1.02，且区间下限高于 1 → 进入定位：对照 C，逐个替换 A 的融合边界（从 context 投影和归一化开始），找出影响接受率的位置，并实测替换后的草稿耗时。
2. 选出的 BF16 变体进入完整 decode 配对 A/B（E8 执行器，循环 Latin square），沿用输出、已提交 logits 和有效状态 oracle。
3. 比值低于 1.01，或区间包含 1 → 记为负结果，关闭“草稿数值偏差拖累接受率”这一性能假设。作为数值保真问题，它仍然开放。

## 解读

decode 速度 ≈ 每步输出数 ÷ 每步耗时。只有变体的草稿耗时不变时，每步输出比值才是 decode 收益的上限。C 本身更慢，只用来判断空间有多大。逐位置均匀统计与在线运行时实际访问到的位置分布不同，因此在线收益必须以第 2 步的完整 A/B 为准。

## 边界

只覆盖 greedy，不推断随机采样；不改 target、DFlash2 权重或 selector 规则；FP32 诊断副本需单独记账内存（此前 FP32 诊断阶段峰值约 80 GiB）；短上下文结论不外推到长上下文。
