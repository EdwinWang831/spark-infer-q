# E36 追加门槛：完整模型的 decode 算术 prefill 候选

在244第0层消融完成后追加，复用前五条受控前缀，并另加两个此前未执行的英文JSON/中文技术说明前缀，分别128/384 token。文本/token在candidate stage内封存。这是工程数值与prompt成本控制，不是广泛质量测试。

候选仍使用原生prefill的大行块和状态布局，但A把forward的MXFP8调用改用decode算术，B把GDN recurrence改为decode chain（完整长度scratch与正确立即提交），C将MoE slots改F32并沿用其联动shared-expert路径。三项来自已观测第0层的组合，不单独宣称新算法。保持其他HC/attention/PLE和存储权重不变，所有覆盖限私有forward实例。

按历史4、英文64、中文64、代码512、资料512、新JSON128、新中文384顺序，对现有rows8 prompt控制。必须满足末行完整logits、五层全部prompt DFlash特征、全部有效GDN/QSA/PLE状态逐位一致，再从各自prefill状态用原decode执行8步，逐步对照完整logits/features/state。任意例失败，记录原因/差异并停止候选，**不启动性能测量**。

只有七例全部通过，才各路径每例一次预热，四轮交替AB/BA测prompt-only wall。已经分词，缓冲准备/JIT在外，包含State初始化、stage、前向、完整五层特征收集、commit、末行预测准备和同步；无HTTP/tokenization/draft。按总prompt token/总墙钟及逐例报告，不把prompt-only结果当完整DFlash服务收益。输出核验在timer外。没有设置“低于某幅度就认为不正确”的规则；数值门槛与性能结论独立。
