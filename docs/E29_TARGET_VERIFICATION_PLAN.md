# E29：验证工作量和 target 图执行

复用E28之后的常驻worker，不重载权重、不训练预测器、不改变旧运行源或全局dispatch。先查现有执行：NVFP4专家已按专家分组，不能把已有分组当新优化。采用与原生TensorFold graphs相同的stage/compute/commit边界，另加现有DFlash五处特征捕获，测试请求私有、固定状态地址、按1～8行和GDN双parity缓存的CUDA Graph。初版attention context固定为已支持的2048容量；需逐项验证logits、特征和提交state，不能假定扩大launch上界数值不变。图构建与正确性检查不计入稳态decode，另报准备耗时/内存，不能当冷请求收益。

执行候选先在独立诊断提示、1/4/8行、双parity、不同提交长度上与原路径配对测试。失败则保留失败结果并定位，不替换默认；全局内核及常驻session保持原样。

预算诊断使用E24 learning中calibration分区按family/language及加盐SHA选择8个root，与E28旧test root分离，不称新独立语料。每root在输出offset 0/8/24/48附近的串行正确前缀上，BF16/W8各产生7草稿，对1～8行逐一计时、核对真实可提交token。8宽度轮换顺序，各3次，初态clone、oracle、路由观察和正确性检查在timer外。保留EOS/cap裁剪和各深度真实emitted，记录每组draft+F1+CPU线性预测成本，并计入width1的已付草稿费用。冻结E17线性模型，禁止用当前验证结果选择可部署动作。

采集后给出width费用曲线、线性预算动作、固定width8、同前缀吞吐oracle及无损截断参考。oracle只用来诊断空间，不能当在线性能上限或实际decode速度。路由仅在另一未计时forward采集，核对全logits/特征与提交状态；专家并集是选中权重footprint，不能充当真实DRAM流量或当前步骤在线特征。

若Graph正确且配对验证存在正信号，执行完整生成A/B。主对照保持BF16草稿和相同冻结线性决策/费用，隔离target执行变化；W8结合量化后校准成本作为另一个预先注明的组合候选。使用E28八个旧test root（已用工程回归，不作新泛化结论），每组4次、平衡先后顺序；token/logits/live state核对串行target，计入全部草稿/F1/预测/调度/verify/commit/循环费用，图预热及prefill独立报告。实际组数和成本冻结封存于在线任务提交前。按root聚类bootstrap报告区间，默认不自动替换。无多请求batch或原生TensorFold排名结论。

在完整Graph结果产生前补充同期预算控制：同一8个旧test root，仅BF16草稿、原eager target，对比固定7草稿与相同冻结线性预算/校准费用。每条各2次、每组先后各一次，共32次计时，独立串行oracle与相同完整正确性检查。此控制用于量化预算减少验证成本的真实效果，不能与Graph实验的比值机械相乘；不会根据Graph结果更改费用、任务或重复数。
