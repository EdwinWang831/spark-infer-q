# E28：DFlash2 MLP backbone 的 W8A16 原型

首轮只量化五层MLP的融合gate/up和down矩阵（294,912,000个参数），attention、特征入口、动态卷积、归一化、RoPE和selector保留现有精度；完整词表输出头沿用现有Q4。INT8每输出通道对称量化，FP32尺度，BF16输入/输出，FP32累加。编译独立SM121扩展，计算时在寄存器反量化，不构造完整BF16权重副本。

原始DFlash checkpoint共603,065,600个参数，其中selector127,795,200，MLP294,912,000，attention131,073,280，其他49,285,120。共享target embedding/head不包含在此检查点参数数中。原路径同时保留原始MLP权重与gate/up融合副本，BF16合计937.5MiB；本轮W8唯一执行包约281.25MiB加尺度。由于A/B仍保留BF16控制和原常驻session，新增W8包会增加实验内存，不能把打包比例当作本轮整机实际省内存。

库通过局部TensorLike权重适配现有F.linear边界，只替换新wrapper自己的字典，保持原始短上下文和追加缓存的执行路径，不改全局算子dispatch、target或原draft对象。原有预测器实验不恢复。复用当前READY常驻worker，权重加载次数保持1。

一组工作内完成：CPU接口/激活/不可变性检查；GPU存储权重算术（尾部维度、1/7/8行、独立stream）；真实第一层gate/up、down的数值和配对微基准；四个中英短/长真实target前缀上的hidden变化、完整词表top16重合、selector提案变化与完整proposer计时。量化后的提案不要求与BF16相同，必须如实记录变化。若完整proposer有可测收益，再用固定草稿长度做真实decode A/B，核对同一未变target的greedy token、committed logits和live state，并计入接受率改变带来的额外验证成本。

第一步不同时调整预算预测器或重新训练接受率模型。量化会改变草稿分布；固定预算对照用于隔离其成本/接受长度影响，不能直接作为动态策略晋升证据。微基准变快、权重更小或hidden误差低，均不能替代实际decode结果。其他GPU后端没有在本轮得到验证，不静默退回其他精度。

实际decode方案在新量化输出产生前固定：从既有E25清单按八个family/language分层和加盐SHA各取一条，英5中3；共8个已用工程回归root，不称新的独立测试集。两组固定7草稿、cap/EOS裁剪，沿用Q4完整输出头和相同F1提取成本，不调用预测器或动态预算。每条各重复4次，先后顺序交替，每组在每条提示上占据两次首位；64次计时。按root聚类4000次bootstrap，报告实际token/s、区间、接受/观测草稿数及输出token/验证轮。
