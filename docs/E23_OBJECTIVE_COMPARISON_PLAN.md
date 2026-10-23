# E23：固定结构和数据，比较训练目标

2026-10-23 UTC，执行前封存方案。E22 的旧 test 只作为诊断背景，不参与本轮拟合或新的独立验证。

## 只改变训练目标

结构保持 137→64→32→7 tanh，seed 20261022，CPU float64，两线程，LBFGS80、L2=.005；输入特征、七条件标签、初始化、归一化、输出预算选择器、独立正斜率 Platt 校准均保持 E19 的定义。

训练集仍为 E17/E19 的 45,081 位置，calibration 为 5,614 位置，不加入新问题。普通 BCE 对照直接复用 E19 已拟合权重。两模型统一使用 E22 NumPy 标量后端，9 轮轮转/反向同输入实测预测费用，不拿原 Python 开销作为新目标的优势。

新候选在原七条件 BCE 和 L2 之外加入 **soft action opportunity loss**：每个 train 位置已有所有可选宽度的真实 `emitted`。冻结旧 train/cal 成本曲线，预测费用使用 E22 已测的 0.0118153632 ms；从 train 单独计算 aggregate-ratio oracle 参考速度 q。实际 action utility 为 `emitted - q*cost`，机会损失为最优 action utility 与各 action utility 的差，并除以 `q*median(train width8 cost)`。预测概率构成各宽度的期望推进量，以 `expected_progress/cost/(q*0.1)` 做带可行宽度掩码的 softmax，最小化该分布下的机会损失。总损失为原 BCE+L2+该项，固定 alpha=1、temperature=.1；不做超参数搜索。

这与 E17 的“给整条样本加难例权重”不同：同一条样本中每个候选宽度都有各自的损失。软化只是可导训练近似，实际推理仍使用原 `choose_rows`。q、机会损失和尺度只读 train 标签；输入仍不能读取当前 Target 结果。

零 action 权重必须精确复现旧 fitter 的层权重与校准，另测有方向的梯度、不可行宽度屏蔽，以及常驻 inference_mode 下 CPU autograd 恢复。Spark 独立 CPU 预检通过才提交正式任务。

## 选型和新的独立测试

只用旧 calibration 的计费回放选型。若新目标不超过冻结 BCE 对照，正常结束为 `COMPLETE_CALIBRATION_NOT_IMPROVED`，保留负结果，不开始新 test 标签采集。

预先准备 72 条问题：新的 OASST1 英/中人工 root 各 30，加各 6 条同 root mixed 派生，共 60 独立 root。排除全部 E17 root，并与 E17/E16 提示做标准化精确与 trigram Jaccard .82 近重复排除。源为已有 Apache-2.0 固定 SHA 文件，无权重下载。输出 cap384、容量2048，实际 tokenizer 预检先执行。新的问题内容和顺序在训练前封存，尚无 Target 标签。

若 calibration 选中新目标，先封存两模型及选型 SHA，再用单独的 `e23_collect` 采集新 test。该生产者验证 E21 加速核心的原始验证 SHA/源字节，复用冻结 E16 轨迹预算；保持 test split，不绕过 E21 学习入口的 test 禁止规则。最多64均匀位置、相同 x/y/emitted 语义，不重新估计 test 成本、不用新模型改变采样位置。

新 test 仅评估一次：除原 fixed-control 的 capture≥.30、bootstrap 正下界、英中正下界与动作变化门槛，还要求相对 BCE MLP 和 E17 线性头的分组增益下界都大于0。按 root 聚类，mixed 保持同组。失败不重新选型/调参。通过仅表示可进入另行受控在线 A/B；本轮回放不提供在线 decode TPS，不自动替换默认执行器。

## 执行边界与后续数据覆盖

常驻 worker 串行队列：065 tokenizer/重叠预检 → 066 fit/calibration → 条件执行067～069三片新 test → 070独立评估。源码、训练输入、已有模型、E21验证和新 test 清单 SHA 全部在065前封存。原 E17/E19/E21 源不修改，Target/DFlash对象不重载，失败停止链路。

数据覆盖扩展属于独立 E24：先准备多来源、英文代码/数学、中文、真实角色多轮和1K以上上下文的清单，再在固定的选定目标下比较加数据的效果。E23 新 test root 必须始终排除；原 test 错例不回灌。E24 不混入 E23 的45,081训练位置，从而分别观察目标与数据的作用。数据源许可、去重、分组与token容量须完成预检后才能采集。
