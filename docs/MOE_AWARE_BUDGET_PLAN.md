# 提议：按 MoE 专家并集建模验证成本

状态：**PROPOSED，尚未执行，没有新的 GPU 结果。** 依赖[实际带宽核算](ACHIEVED_BANDWIDTH_PLAN.md)采集的路由数据。实际排期时再分配 E 编号。

## 动机

- [E9](E9_DECODE_PROFILE_REPORT.md) 汇总的 E8 数据：40 次请求共验证 5,492 行，只提交 3,348 行，2,144 行（39.04%）未提交。在 MoE 中，每多验证一行都可能带进新的路由专家，读取字节随专家并集增长，而不是随行数线性增长。
- DeepInfer `MTPVarlen` 的运行时成本是“启动曲线 + 斜率 × graph size”；[RFC #2](https://github.com/EdwinWang831/spark-infer-q/pull/2) 的成本按模式、batch、bucket 和上下文分档。当前仓库没有把专家并集当作成本变量。
- 单请求下成本曲线较平（[E3 宽 block](E3_WIDE_BLOCK_REPORT.md)校准：1 行 43.39 ms → 8 行 68.63 ms），所以这个方向在单请求上的收益可能很小。多请求时，不同请求的行共享专家更少，成本更陡，跨请求分配的价值更大。
- **新颖性尚未检索**。主张之前须先核对：TensorFold 当前 `allocate.zig` / `depth.zig` 的成本项，SGLang/vLLM 的 MoE 投机成本模型，以及相关论文。

## 假设

- **H1**：step 时间 ≈ a + b·rows + c·U（U 为各层专家并集之和），对独立 step 的预测误差明显低于 a + b·rows。
- **H2**：只把 `MTPVarlen` 的时间模型换成 H1 模型、分配顺序不变时，会选出不同宽度，并在低接受阶段减少无效行。
- **H3**：多请求下 H2 的收益大于单请求。

先只换时间模型，是为了做单变量消融。按“接受率 ÷ 边际成本”重新排序的变体（可以在请求之间挪预算）留到 H2 有结论之后再测。

## 阶段

**A. 离线成本模型。** 数据来自带宽核算的采集，校准提示与测试提示分开。拟合 M0（rows）和 M1（rows + U），报告 held-out step 的中位绝对误差，以及按宽度和上下文分桶的残差。rows 与 U 高度相关，需要不同任务、上下文和宽度 1..8 的重放样本；退化样本直接报错，不输出不可辨识的系数。**事前门槛**：M1 相对 M0 的中位绝对误差至少降低 25%，否则停止，并记为负结果。

**B. 离线策略比较。** 沿用 [budget audit](E3_BUDGET_AUDIT_REPORT.md) 的逐位置宽度重放（每个位置宽度 1..8，各三次），选三个任务，含低接受的解释任务。比较校准固定宽度、`MTPVarlen` 和 MoE-aware 三种策略在模型内的总成本。MoE-aware 只能使用由过去 step 估计的“每深度新增专家数”（校准先验加同请求在线滑动平均），不能读取当前 step 的路由，因为它要等 forward 之后才知道。重放结果只是模型内诊断，不是在线收益。

**C. 在线单请求。** 新冻结的测试提示，不复用 E3/E6/E7。比较四组：serial、仅用校准选定的固定宽度、`MTPVarlen` dynamic7、MoE-aware dynamic7。采用循环 Latin square，每组至少五次，oracle 同 E3。报告重复级配对比值及其 bootstrap 区间；区间跨 1 时写“未证明”。

**D. 多请求。** K4 之前只做离线估计（行合并求并集，基于 batch 不变性假设）。K4 完成后再做在线双请求对照，报告总吞吐与逐请求 TPOT p50/p95。

## 停止条件

阶段 A 未过门槛，或阶段 C 相对校准固定宽度在区间内不显著 → 保持简单策略，MoE-aware 只保留为研究分支。

## 边界

只覆盖 greedy；不改内核、权重或 DFlash2；当前 step 的路由不能作为决策输入；离线多请求估计不能替代 batch 实测；随机采样需另行证明调度依赖不引入偏差。

## 参考实现

[`spark_infer_q/moe_cost.py`](../spark_infer_q/moe_cost.py) 提供：
- `fit_step_cost`：最小二乘拟合，退化样本报错；
- `predict_ms`；
- `plan_widths`：沿用 `MTPVarlen` 的分配顺序（接受率降序，再按请求序号），只把时间模型换成 rows + 专家并集；`per_expert_ms=0` 时退化为不分 graph bucket 的 rows-only 选择。

均为标准库 host-only 代码，有单元测试，未在 GPU 上运行。
