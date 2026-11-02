# K1：原生调度器驱动的 Qwen Target-only 运行时

2026-10-23。以 E5 原生 Scheduler 桥接为起点，增加可复用的 `KaifraEngine`：文本/Token 请求、逐步输出事件、EOS/长度结束、取消、slot/generation、TTFT/完整墙钟指标和清理。保留仓库/包名。原生 Scheduler 和逻辑 BlockManager 仍决定请求计划；TensorFold Target 独立拥有 GPU 状态。

本轮先完成 target-only。实际 GPU forward 逐请求执行；DFlash2、异步 Worker、整模 CUDA Graph、GPU 页映射和跨请求 batch 不属于本轮已交付能力。下一阶段在同一运行时接入 DFlash2 与现有预算器；不把旧推测循环包装成已统一的引擎。

## 验收（执行前固定）

- 本地无 GPU 契约测试：事件只含已提交 token，pending 输出 token 不提前推进设备 position；多 EOS、长度、等待/已输出后取消、旧句柄、错误隔离、成功 fence 后释放、Unicode 文本快照、关闭和有界历史。
- 固定 DeepInfer revision/扩展 SHA 与 E5 相同；真实 C++ 调度器通过引擎 API 检查单请求、两个请求交错、chunked prompt、EOS/长度、等待/已输出后取消和槽位复用、非法/满槽请求。
- 复用当前空闲常驻模型。不修改旧实验源、不重载权重、不改变现有 Target/DFlash。新源码独立目录封存 SHA，并通过唯一包名载入，避免覆盖已导入模块。测试结束恢复进程组/环境，worker 回到 READY。
- GPU 使用英文算术、中文简答和 JSON 三个受控输入，max_tokens=16～48，capacity=256、最多两请求、token_budget=32。独立生成同路径逐 token reference，逐行完整 logits、每块有效 GDN/QSA/PLE state、完整输出核对。prefill_rows=1 和8均覆盖；这不证明独立整模数值正确性。
- 取消后没有新增输出，释放状态；两个请求交错时独立状态不污染；max_tokens=1 时设备只处理 prompt，最后输出 token 保持 pending。所有终止请求归还逻辑块和设备 state，运行时不无限保留历史输出。
- 本轮是正确性/生命周期验收，不作性能提升结论。错误停止并保留原始失败；修复后使用新封存源码进行明确重跑。
