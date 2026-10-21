# E12：专家成本感知 Verification

状态：实现观测实验；在线策略尚未启用。默认执行路径不变。

目标是检验专家信息能否改进单请求验证长度选择。沿用 E8 的 TensorFold Target、BF16 激活、Q4 DFlash 输出头和 DeepInfer MTPVarlen 基线；不叠加 E10/E11 可选内核。复用 [PR #1](https://github.com/EdwinWang831/spark-infer-q/pull/1) 的 `moe_cost.py` 与测试，源提交 `af74ef0563b97e445f9cb72a8c086a19a9e3e3fe`。本分支修改其字节说明：专家 pack footprint **不是测得的 DRAM 读取量或带宽下界**。

## 一轮观测回答三个问题

1. 同样行数和上下文下，专家并集变化是否对应可重复的验证耗时变化？
2. 增加真实专家并集的离线解释模型，能否胜过宽度及上下文基线？
3. 仅用上一轮已执行路由的廉价模型，是否也能改善预测，且足以改变预算选择？

第 2 项是未来信息诊断，不能直接上线。第 3 项才有线上可用性；离线行为策略覆盖仍不能替代在线测试。

## A：固定前缀反事实观测

实验任务与参数在 [e12_tasks.py](../scripts/e12_tasks.py) 冻结：6 个校准、6 个独立 prompt 测试，覆盖中文、英文、代码、JSON、推理和较长输入。不是随机拆分同一 prompt 的 step，也不是通用质量评测。

- 每个 prompt 从 rows8 prompt 状态开始，最多观察 6 个真实行为 step；行为策略固定 width4，遇 EOS 停止。
- 在同一 committed state / anchor / DFlash 草稿上重放 widths1–8，各正式计时 3 次，宽度顺序循环轮换。先预热，状态复制不计时。
- 正式计时走原 `WideTarget.run`；包含特征采集与同步的 verify wall，另记 commit 和完整 draft wall。CUDA events 记录 GPU stream elapsed，它含主机发射间隙，不能当累计 kernel 时间。
- 路由另跑一次观察路径，每层复制 Top-10 ID 到预分配 GPU 缓冲；48 层后统一读取，不逐层 `.cpu()` / `.item()`。观察路径耗时独立报告，不能替换正式耗时。
- 检查观察路径与原路径所有 logits、draft features 一致；每个宽度的 greedy 已提交 logits、EOS、GDN/QSA/PLE live state 对照 serial。无效或 rejected suffix 不作为 committed state。
- 每层单独统计唯一专家与前缀并集，并导出 pack footprint。跨层相同 ID 不合并。保存 raw routes 用于复核，公开汇总不含权重或私有环境路径。
- 下一步特征只能引用上一个 **行为 width4** 已执行的路由；不能使用上一个反事实 width8 的路由冒充历史。

## B：成本分析与止损

先对各前缀/宽度取三次正式计时中位数，报告重复噪声与观察开销。只在 train prompts 拟合，test prompts 只评估：

| 模型 | 输入 | 用途 |
| --- | --- | --- |
| M0 | 每宽度的实测成本曲线 | 强于全局线性 rows 的基线 |
| M1 | 每宽度截距及 context 斜率 | 排除输入长度差异 |
| M2 | M1 + 当前轮真实专家并集 | 离线解释能力；不能上线 |
| M3 | M1 + 上一行为 step 的专家并集 | 仅历史信号的低成本候选 |

上下文涵盖 QSA 的有效长度变化；本轮不声称已覆盖所有长上下文、图桶及稀疏 attention 形状。各宽度分别拟合，秩不足明确记录，不返回伪系数；负耗时预测视为无效，不静默裁剪。

报告 held-out **mean absolute error** 与 median absolute error、逐宽度残差、逐 prompt 配对误差、成本排序和冻结接受率先验下的预算选择差异。新增专家信息相对 M1 的 mean absolute error 改善至少 25% 才值得进一步在线验证；同时对照 M0。PR #1 的 median 门槛同时保留报告，不事后挑好看的指标。

若 M2 无明显解释增量，停止该特征路线；若 M2 有用而 M3 无用，说明需要更好的验证前信号，尚不能宣布算法失败或训练必然有效。置信区间很宽时写“证据不足”，不当作已证伪。按 prompt 做不确定性统计，不能把重复计时当独立样本。

真实成本选择的同前缀收益只是诊断：固定相同 acceptance prior、计入 draft/commit；不使用未来 acceptance 伪装在线决策，不声称这是完整路径的严格收益上界。

## C：有证据后接入在线预算

先用无训练规则/小回归替换成本估计，保持接受率先验、合法单链前缀及 Target 路由不变。四组：serial、独立校准选定的固定宽度、原 MTPVarlen、专家成本策略。采用新的 prompts、平衡顺序、至少 5 个重复，报告有效 decode tok/s、完整请求耗时、TPOT、接受/提交长度、额外预测与同步成本、峰值内存及配对区间。

输出、已提交 logits 和 live state 必须过同路径 serial oracle；只覆盖 greedy。在线输出策略改变会改变未来历史，离线预测改善不能替代这一轮实测。区间包含无收益或有明显任务退化时保持 opt-in，不升默认。

## 后续依赖与研究定位

先单链，再考虑 DFlash 多候选/树；树须另解 GDN/PLE 分支状态。双请求 batch 单独推进后，才研究跨请求专家复用。当前不训练完整路由预测器，不新增专家缓存或改量化。

[EcoSpec](https://arxiv.org/abs/2607.12696) 已有专家边际成本感知选择，[PR #5 R1](https://github.com/EdwinWang831/spark-infer-q/pull/5) 是本轮方向依据；“专家感知”本身不作为首创新意。研究增量应由 Spark 上真实成本、低开销历史预测和混合状态正确性消融证明。
