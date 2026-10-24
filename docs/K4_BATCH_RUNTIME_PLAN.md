# K4：双请求 GPU batch 与异步 worker

历史初版方案如下。后续已有可选[自有紧凑 Target](K4_OWN_TARGET_REPORT.md)：stage/层编排/GDN-QSA/commit 与有界 Graph 已迁出原控制循环；默认 legacy 路径和其 80 张预备图仍保留。后续自主后端和四请求验收以[更新计划](DEEPINFER_OWN_BACKEND_PLAN.md)为准。

本阶段承接 PR #18 的运行时，但按路线图 K4 的原始定义验收：两个请求必须共享一次完整 Target GPU forward。之前的 `K4_RUNTIME_OVERHEAD.md` 是开销实验，不代表该 batch 阶段完成。

范围：单卡、两个并发请求、greedy、原 E33 辅助头/W8 草稿/E17 NumPy 当前草稿后预算。各请求独立 GDN/QSA/PLE、草稿投影缓存、pending anchor、输出和代次。每请求验证宽度仍为动态 1～8；不恢复固定草稿运行模式。先验证 no-draft，再相同/不同动态宽度、拒绝前缀与终止混合。

执行器使用原 TensorFold stage/48 层共同 compute/按请求偏移 commit。Graph bucket 按各 slot 的 1/2/4/8 行与 GDN parity 预备；不足 bucket 的尾行不发布、不进入有效 state，commit 使用实际接受前缀。没有预备的 bucket 明确 eager fallback，不在请求中 capture。联合调用/真实行/物理 padding 行分别计数。

异步 worker 是请求与 GPU 生命周期的唯一驱动者，命令队列和每请求有界事件队列独立。GPU 结果用 CUDA event 判定完成，在完成前不发布 native output、不复用 slot。取消可以在 GPU 在途时登记，已取消的结果不得发布；回收等待设备 fence。保留必要的 CPU 草稿特征/预算依赖，不承诺消除全部 GPU/CPU 同步，也不使用原 DeepSeek 专用 Worker/cache 布局冒充 Qwen 适配。

服务闭环覆盖：请求准入/满额、预填充、联合验证、按请求 SSE、EOS/长度、显式取消/断连、慢消费者、slot 重用、worker 故障隔离与有界关闭。多请求命令接受不等于联合 GPU forward；只有真实模型计数/布局和数值对照通过才声明 gpu_batch。

验证：本地生命周期/背压/异步完成测试；resident 原生 CPU；两请求真实 GPU 与 solo 完整 tokens/committed logits/features/live state 对照，覆盖全接受/首中末拒绝、padding、不同 parity、取消三个阶段、同 ID 复用、并发 SSE。正式测 total throughput、各请求 TTFT/TPOT p50/p95，保留当前预算轨迹、真实/物理验证行和 batch 大小。组件正确性、联合执行与性能晋升分别报告。

所有作业独立封存，通过共享 resident 队列顺序执行，保留 load_count=1；不修改研究工作区或其冻结模块，不重载/停止共享模型。单请求费用仅作为每请求策略保留，不宣称跨请求成本最优；K5 的全局预算优化不在本阶段。
