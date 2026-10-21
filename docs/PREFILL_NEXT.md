# 下一阶段：保留解码算术的分块 prompt 路径

E7 对照期间只准备方案，不修改正在计时的执行器。用户明确将 prefill 提到优先级前列，INT8 草稿仍保留后续候选。

当前候选逐 token 执行 prompt。首选验证利用现有 WideTarget 的 1..8 行 `prefill=False` 执行/commit 路径分块输入 prompt；先测2/4/8行，不直接切换此前产生约10.68%末尾 logits差异的独立prefill算术。E3的44个窗口/保留前缀检查提供接口证据，但不能替代完整prompt验证。

必须对照完整逐token prompt：每行完整logits、捕获的五层草稿特征、GDN/QSA有效状态、PLE历史和最终pending token。覆盖E7十条实际prompt、不同尾块长度和边界，再测后续真实decode，输出/状态均须保持一致。出现差异应保留并停止该配置性能结论。所有宽度先完整预热；成本与接受先验保持E6冻结，避免混入新预算调参或量化。

性能同时报告prompt处理、首token/完整请求与decode，核对计时排除加载和oracle但包含真实状态初始化。先完成分块路径正确性/收益，再评估CUDA Graph、缓冲区复用、独立原生prefill内核；不要一次混入多项数值变化。INT8 backbone+BF16 selector可另做受控实验。
