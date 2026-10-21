# 提议：DFlash2 草稿后置信度头的设计与训练

状态：**PROPOSED，尚未执行，没有新的 GPU 结果。** 实际排期时再分配 E 编号；执行前按本仓库惯例冻结提示、特征、超参数和阈值。

## 定位与已有工作

“用学习到的接受率预测器动态决定草稿或验证长度”是投机解码中研究较多的方向，**本提议不主张算法新颖性**：

- DeepInfer 自己的 DFlash 带有学习的 confidence head，并据此调度验证；本提议本质上是把这一设计迁移到 Qwen 的 DFlash2，并接入原生 `MTPVarlen`。
- SpecDec++（训练接受概率头，决定何时停止起草）、DISCO（分类器决定是否继续起草）、EAGLE-2（按草稿置信度选择验证节点）、AdaEAGLE（轻量预测器预测草稿长度）、SmartSpec（多请求下按接受率与 batch 成本调整投机长度）都有相近的机制。
- TensorFold 使用累计概率阈值规则；[decode RFC](https://github.com/EdwinWang831/spark-infer-q/tree/design/kaifra-decode-frontier-rfc) 还列出了 EcoSpec、EVICT 等 MoE 相关工作。上述工作仅凭记忆列出，正式表述前需要重新检索。

可表述的贡献是：**在 Qwen 混合注意力 MoE、块并行 DFlash2 草稿和单台 GB10 的组合下，给 `MTPVarlen` 补上逐草稿置信度，并量化它在不同任务上拿到了多少收益上限。** 下面两点或许有新意，但都没有查新：一是起草成本固定、只剩“草稿后选几行验证”这一个决策；二是直接以决策后悔值评估训练效果。

## 收益上限

[`scripts/oracle_headroom.py`](../scripts/oracle_headroom.py) 读取 [E8 完整生成数据](../results/e8/2026-10-21-integrated-prompt.json) 中 dynamic7/rows8 的 40 次请求、1,021 步，并按同一批数据的各宽度中位验证成本计算：

| 项目 | 结果 |
| --- | ---: |
| 完美预测下只验证实际接受的前缀（草稿后决策） | 省 **12.2%** decode 时间 |
| 再在零接受步跳过起草（草稿前决策） | 再省 **3.4%** |
| 合计上限 | **15.6%** |
| 一个草稿都没接受的步 | 23.7% |
| 已验证草稿全部被接受的步（可能需要更多行，未计入上限） | 33.2% |
| 上一步零接受时下一步也零接受 / 上一步有接受时下一步零接受 | 35.7% / 20.5% |

上限集中在散文：中文散文 23.7%、英文散文 22.6%、长散文 19.6%；整数 10.2%、thinking 算术 9.1%、长代码 8.3%、中文 JSON 6.0%；函数代码和 LaTeX 接近 0。

这是近似上限：不同策略会移动步边界，后续草稿也会随之改变；成本只取中位数，宽度 2、5 是插值；没有计入预测器开销。E3 局部策略只拿到其事后空间的约 1/5（1.95% 对 10.6%）。按 20–40% 估算，整体约 3–6%，散文约 5–10%。这只是条件估算，不是预测。

## 设计决定

1. **只做草稿后决策。** 草稿前的启停上限只有 3.4%，零接受在相邻步之间的持续性弱（35.7% 对基准约 24%），E3 的“仅启停”净收益为负。草稿后决策时，草稿成本已经支付，必须计入所有候选宽度（[`choose_rows`](../spark_infer_q/confidence_head.py) 的约定，与 E3 `choose_width` 一致）。
2. **输出直接喂给 `MTPVarlen`。** 对 7 个草稿输出累计概率 P(验证全部草稿时输出超过 j 个 token)，单调不增，作为当前步的 `accept_rates`。greedy 下使用当前步置信度不改变输出；随机采样必须保留 DeepInfer 的滞后约束或另行证明无偏，不在本提议范围内。
3. **模型从简。** 沿用 E3 的条件 logistic 头加独立正斜率 Platt 校准，扩展到 N 个深度（[`fit_heads`](../spark_infer_q/confidence_head.py)），推理直接复用 `local_policy.ProgressModel`。E3 冻结源保持不变。只有 logistic 未通过离线门槛时才评估小 MLP，并同样计入推理成本。

## 特征（必须在草稿后、验证前可得）

| 组 | 内容 |
| --- | --- |
| F0 | 7 个位置各自的 unary margin、选中候选与 unary 第一名的差距、实际 predecessor 下的 edge margin（21 维）；上下文长度、已输出长度、thinking 开关、剩余输出上限 |
| F1 | F0 加每个草稿位置 DFlash2 末层 hidden 的固定随机投影（例如每位置 16 维，种子固定，投影不训练） |
| F2 | F1 加 E3 前置特征：target 捕获的 64 维投影、anchor embedding 的 32 维投影、target top-2 差距 |

禁止使用当前步的 target 验证结果或任何未来 token。三组作为消融，事先固定，不根据测试结果挑选。

## 训练数据

- 每条 prompt 先完整运行一次 serial target，记录每个位置捕获的特征和 serial 续写。然后在**每个** serial 位置运行 8 行 DFlash2，得到 7 个 proposal、分数和 hidden。
- 标签由 [`draft_fidelity.emitted_by_width`](../spark_infer_q/draft_fidelity.py) 和 `cumulative_labels` 生成，遵循 `speculative.greedy_result` 的契约：输出不越过 EOS 和输出上限，EOS 被接受后不再计算后续进展。[草稿保真诊断](DRAFT_FIDELITY_ACCEPTANCE_PLAN.md)用的是同一个函数，两边的结果口径一致。
- 规模目标约 2,000 条 prompt，覆盖中英文、散文、代码、JSON、数学和 thinking，量级为 10⁵ 个位置。粗估 serial 生成约 3–4 小时，草稿约 1 小时。E3/E6/E7/E8 的测试提示不进入训练，只作回归。不使用用户私有 prompt，外部数据集需核对许可。
- 按 prompt 划分训练 / 校准 / 测试 = 70/10/20（[`prompt_split`](../spark_infer_q/confidence_head.py)，哈希确定）。另外整体留出一个任务族，检验泛化。绝不按位置随机拆分。
- 草稿算术若因 [PR #4](https://github.com/EdwinWang831/spark-infer-q/pull/4) 的结论而改变，标签必须重新生成，模型必须重训。因此先完成 PR #4 的保真诊断。

## 离线评估与事前门槛

- 成本曲线：在校准 prompt 上实测 1..8 行的验证加提交成本（同 E3 宽 block 的方法），再加上已支付的草稿成本。
- 在测试 prompt 上用 [`headroom_capture`](../spark_infer_q/confidence_head.py) 比较三者的 token/ms：仅用校准集选出的最佳固定宽度、模型策略、无损 oracle（输出最多 token 的最少行数）。报告“拿到的上限比例”，按 prompt 整体重抽样给出区间。逐深度 Brier 和校准曲线只作辅助。
- **门槛**：拿到上限的比例 ≥ 30% 且区间下限 > 0，才进入在线 A/B。离线重放是按位置均匀访问，不是在线分布；结论只能用于决定是否上 GPU。

## 在线 A/B

E8 执行器、rows8 prompt、完整 Q4 草稿头。四组：serial、校准选定的固定宽度、原版滞后 `MTPVarlen` dynamic7、`MTPVarlen` 加置信度头。新冻结测试提示，循环 Latin square，每组至少五次。输出、已提交 logits 和有效状态 oracle 同 E8。特征提取与预测开销全部计入；第一版可以像 E3 一样在 CPU 上推理（约 0.5 ms/次，约占一步的 0.7%），后续再移到 GPU，避免额外 host 同步。报告每任务结果，单独列出散文。

## 停止条件

离线未过门槛，或在线相对校准固定宽度的区间包含 1 → 保持固定宽度或滞后规则，置信度头只保留为研究分支。不为“学习型调度”增加未经证实的复杂度。

## 后续

多请求下，`MTPVarlen` 按这些概率在请求之间分配验证行，价值可能更大；需等 K4 的真实 batch 验证后再做。草稿前启停、MoE 专家并集成本（[PR #1](https://github.com/EdwinWang831/spark-infer-q/pull/1)）各自独立评估，不在本轮混入。

## 参考实现

- [`spark_infer_q/confidence_head.py`](../spark_infer_q/confidence_head.py)：`cumulative_labels`、`prompt_split`、`choose_rows`、`headroom_capture`、`fit_heads`（torch，CPU float64 LBFGS，输出与 `ProgressModel` 兼容的元数据）。
- 逐宽度输出数 `emitted_by_width` 在 [`spark_infer_q/draft_fidelity.py`](../spark_infer_q/draft_fidelity.py) 中，和 `step_outcome` 共用同一个 greedy 实现。
- [`scripts/oracle_headroom.py`](../scripts/oracle_headroom.py)：复现上文的上限表。
- 单元测试覆盖标签契约（拒绝位置、EOS、输出上限）、按 prompt 划分、宽度选择与 E3 规则一致、完美概率拿到全部 oracle 空间、恒定概率拿到 0，以及在可分信号上训练（无 torch 时跳过）。均不在 GPU 上运行。
