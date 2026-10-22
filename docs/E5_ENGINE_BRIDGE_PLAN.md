# E5 最小真实调度桥接

使用已构建且固定为 `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` 的 DeepInfer `_C.engine.Scheduler`，驱动现有 TensorFold/Qwen target-only 路径。核心二进制沿用 E1 release 构建，SHA256 `f6aa14b790eb7bc729251eea152fd9c83cf215bc5ab912e25ee19c57d8fc1768`。不修改外部源码，不启用草稿、随机采样、fork、抢占重算或 GPU ragged batch。

原生 Scheduler 管理队列、分块计划、Sequence 计数及逻辑 block 配额。Qwen 后端独立拥有真实 GDN/QSA 状态；逻辑 block ID 不冒充 GPU KV 页。`log_stats=False`，不向 DeepSeek 专用 FlopsEstimator 伪装 Qwen 架构。单 rank Gloo 只为满足原生调度接口，不需要 NCCL/多卡。

适配器显式划分 prepare / execute / complete：每次最多一份未完成计划，真实后端完成后才更新原生输出账本；取消后仍保留租约直到该计划收尾。稳定 slot + generation 拒收过期句柄/重复完成。队列中取消使用原生 abort；已被取出队列的 prompt 还需由适配器标记其持有的原生 Sequence 为 aborted。终止请求在安全边界归还原生 block 和真实状态。错误后停止继续调度，不能悄悄使用可能污染的状态。

先以真实 C++ 调度器和 CPU toy backend 检查 admission、chunked prefill、双请求推进、EOS/长度结束、等待/已准备 prompt/已准备 decode 取消、重复完成、代次复用、block 归还及非法操作拒收。再加载一次完整 target，复用 E4 的英文整数/JSON/纯函数输入做回归，验证单请求完整 EOS、双请求交错、长度截断和取消复用。每次实际 forward 的输入位置、完整 logits、输出 IDs 及有效 GDN/QSA 状态对照本路径 serial；不将同后端对照称为独立整模正确性。

GPU 使用短上下文、BF16 KV、逐 token prompt 算术，保持既有数值边界。分块调度不等于批量 prefill 已优化，双请求计划不等于同时执行两个请求的 GPU batch。本轮不作速度提升结论；target prefill/官方 drafter BF16 数值问题保留。原始私有日志/环境/权重不进 git，只公开代码、配置、计数和脱敏结果。

完成本闭环后进入 DFlash2 草稿头专项：拆测 head 成本，依次比较缩词表、量化和组合，并重新测接受率、decode/请求耗时、总内存及正确性。完整 DeepInfer 引擎、图执行和跨请求预算随后基于实测推进。
