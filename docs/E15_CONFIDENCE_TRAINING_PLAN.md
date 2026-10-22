# E15：小规模七深度置信度训练

执行前冻结。基于PR #6的`af861f9`训练工具，先修复oracle口径；不合并或修改远端PR，不训练Qwen/DFlash主体。

## 数据与隔离

104条新合成提示，48个语义组各2个变体，组内同split；另8个SQL任务整族留出。60训练/24校准/12测试/8 OOD。中英文解释、代码、JSON、thinking数学和结构化输出；12个输入为1024/1536 token合成背景，其余短输入。生成cap64/96，EOS可提前。E3–E14提示不用于本轮训练或调参。

每提示用同一resident Target生成serial轨迹，在每个有续写的serial位置调用完整8行DFlash，得到7草稿。每位置同时比较原cached proposer，确保新特征提取不改变草稿。标签为各宽度实际输出数、累计进展事件，遵循greedy EOS/cap契约；条件头只在上一进展事件成功的样本上拟合。拒绝以后的累计事件0是已知进展结果，不是把未观察的条件接受标签当失败。输出上限作为已知动作约束，而非只要求模型学会。

## 模型与特征

固定两种消融：F0=七位置edge margin/unary margin/selected-unary gap，共21维，加log(context)/log(output)/log(remaining)/thinking，共25维。F1加七位置末层hidden的固定2560→16随机投影，共137维；投影种子20261021。目标当前verification不可作为输入。七个条件logistic头，CPU float64 LBFGS80、L2=.005，独立校准集正斜率Platt。仅以校准集选模型；不在测试结果上触发MLP或调超参数。

## 成本与oracle

仅在校准提示首个serial位置，逐宽度1..8各测3次verify+实际greedy commit，state clone不计时；对committed logits/live state作serial oracle。按短/长context汇总中位数。每位置另测native cached draft与完整F1 draft+特征，校准集计算各context类别中位数。策略成本加非负特征增量及实测CPU预测成本；固定基线与理想oracle不收预测器费，已付草稿成本对所有宽度均计入。

修复`headroom_capture`：用Dinkelbach迭代求均匀前缀集合上SUM(tokens)/SUM(cost)最优oracle；保留最大进展的lossless oracle作单独诊断。它不是在线引擎吞吐上界。所有策略强制剩余输出容量限制。

## 门槛与在线验证

在校准集选出最佳固定宽度及F0/F1模型后冻结。测试与OOD分开报告按语义组bootstrap；Brier/校准/各深度支持量为诊断。仅当新测试集相对固定基线的oracle-gap捕获≥30%、组bootstrap下限>0，并且模型在校准与测试均具有动作差异和正收益时，进入新冻结提示的在线GPU对照。未过门槛则保留完整负结果，不在同一测试上换模型再宣称独立验证。

在线需固定宽度、原生lagged预算、预测器三组，同一缓存、同一成本及七草稿可用范围；计入所有特征、CPU传输与预测费用，输出/logits/live state对照，不计prompt/clone/oracle。执行前另冻结具体提示与重复数。

在线入口已预先冻结于`e15_online.py`：六条新提示，中英文/代码/JSON/thinking数学，含1024和1536-token合成输入；每模式5轮、cap128，平衡顺序，共90次正式计时及18次独立oracle。固定组按post-draft预算对照也生成完整草稿；末尾只剩一个输出时所有组跳过草稿。如果校准选中固定1行，另需target-only真实成本对照，不能将支付草稿的固定1组视为完整性能基线。原生组保留pre-draft决策和runtime反馈，所有差异单列，收益不得只归因于预测精度。

## 常驻与记录

复用PID319的现有Target/DFlash/Q4头，load_count保持1。每个实验模块和依赖用SHA固定，结果逐提示落盘，任务完成后保持READY。首个smoke单独检查1..8宽度和特征提取路径；训练过程恢复CPU线程设置。公开报告保留训练模型、指标、数据SHA和源SHA，原始合成轨迹/特征在gitignored目录。不下载权重。
