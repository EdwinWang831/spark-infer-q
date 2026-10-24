# E39：CTA 协作加载 prefill 专家权重

2026-10-24，事前冻结，承接 E38 失败结果，不覆盖已执行源或修改默认路径。沿用 E38 的四条 E37 输入与真实层0/23/47、64–512-token范围、既有E36算术。只测一个候选，不扫描 tile/warps/stages。

候选：32行plan中<=16行item保留原单warp内核；>16行item由一个64线程CTA处理，两个warp各自持有一份16行累加器。原packed NVFP4权重通过cp.async加载到CTA共享内存，双缓冲；每warp从共享内存取得相同packed权重，独立执行原FP4解码、scale FMA、per-expert scale及epilogue。保留BF16输入、原K-block顺序、SwiGLU和F32 down。rows<256继续原MoE。两个互斥分支分别launch，不以减少解码指令或加大单warp累加器替代此候选。

静态审核确保所有CTA线程经过相同barrier：item/column由block统一指定，skip/cnt分支CTA一致；在两warp均读完共享槽后才能覆盖，cp.async wait与CTA同步确保可见。未知/越界行不得读取members或输出。编译时核对固定原source/header SHA与独立SM121 binary/source/resource指纹。

组件采用E38完整协议：40项人工路由（十种行数×连续/stride×F32/BF16 down），核对全部act/y和guard、Graph两次重放及独立caller stream；真实四输入×三层的完整router/logits/pick/wts/act/y精确。两缓存条件×六轮AB/BA×两模式，共288 CUDA-event区间，576次完整MoE调用；warm每区间3次、128MiB扰动每区间1次，捕获/预热/检查/扰动在外。门槛：扰动成本至少降低5%、至少5/6轮更快、每例不能慢超过2%，否则停止，不外推prompt/decode收益。

只有组件过门槛才运行完整prompt：同四输入六轮AB/BA，48次wall，每模式7296 tokens；包含State初始化、stage、48层、五处草稿特征/拼接、commit、末token和同步，准备/JIT/预热/精确检查在外。完整logits/特征/state、每例8步原decode交接全精确；正式采纳门槛为总prompt wall至少降低3%、至少5/6轮更快且任一输入不慢超过2%。

继续使用同一常驻模型和共享串行队列，不reload、不empty_cache，不修改K4工作区。观测回调在finally中恢复；收尾另一个作业GC/同步后核对baseline、同worker/对象/load_count=1、原函数和冻结源。所有失败记录保留，公开输出去除token IDs、私有路径和对象地址。组件不通过即结束本次候选，不承诺新增收益。
