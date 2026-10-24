# DeepInfer 与自有 Qwen/Spark 执行后端

2026-10-24 方向更新。本文是后续实现与验收计划，**没有新增执行后端、GPU 结果或性能提升**。当前运行证据仍以 [K4 功能报告](K4_BATCH_RUNTIME_REPORT.md)、[等待修复](K4_WAIT_FIX_REPORT.md)和[联合预算实验](K4_JOINT_BUDGET_REPORT.md)为准。

用户明确要求：学习 TensorFold 的技术思路，更多使用 DeepInfer 和我们自己的实现。最终架构以 **DeepInfer 调度底座 + 自有 Qwen/SM121 执行后端**为目标；TensorFold 是迁移期间的参考和可替换适配器。增加对 TensorFold 内部执行器的接线，不作为自主后端的完成标准。

## 1. 当前能力和真实依赖

| 能力 | 当前实现 | 后续归属 |
| --- | --- | --- |
| 准入、Sequence、schedule/process_outputs | DeepInfer 原生 C++ Scheduler；我们的请求/代次适配 | 继续适配 DeepInfer，保留来源与上游边界 |
| 内存配额 | DeepInfer BlockManager，仅逻辑记账 | 调度配额继续复用；设备 KV/GDN 存储由自有 Qwen StateManager 管理 |
| 动态预算 | E17 本轮预测；已有 MTPVarlen 接线；当前双请求联合成本选择器为自有 Python 实验实现 | 保留 E17 与动态预算，适配真实批次成本，再推进 DeepInfer 跨请求预算集成 |
| 请求、异步 owner、取消、SSE、验证发布 | 自有 Kaifra runtime | 继续使用，扩展到四请求生命周期 |
| 草稿 | 自有 DFlash2/Q4/W8 接入和特征/预测器；逐请求调用 | 自有 batched DFlash 执行，保持当前模型和预算语义 |
| Target joint forward、Graph/event 编排 | 自有编排，调用 TensorFold stage/layer_forward/commit | 自有批计划、执行循环和 GraphManager |
| 权重读取、GDN/QSA/PLE 状态、主要 Target 算子 | TensorFold 模型执行与存储实现 | 逐模块替换为自有 Qwen/Spark 后端；算子按适配、正确性和性能证据选用 |

当前 `JointTarget` 虽合并了 dense/MoE 行，仍不能等同于完整的跨请求 GDN/QSA 融合执行。草稿也尚未批量执行，最大请求数仍为 2。不能据此宣布 DeepInfer 已提供完整 Qwen 模型后端。

DeepInfer 已核对的模型实现以 DeepSeek 为主，没有现成的本模型 GDN/QSA 后端。GB10/SM121 上通用 RMSNorm 与选定 cuBLASLt 路径有执行证据，但固定 DeepGEMM 原生 BF16 dispatch 报 Unsupported architecture。单专家的 native NVFP4 测试使用 TensorFold 原版 CUDA 源码，不能记作自研内核或全模型 DeepInfer 兼容。详见 [E0](E0_REPORT.md)、[E1](E1_GPU_REPORT.md)、[E2 native](E2_NATIVE_REPORT.md)。

## 2. 学习机制，明确我们要实现什么

参考仓库固定为 `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold@656879fd752749bbd11a6e1a065a326c00940911`。其中 native MTP 与我们的外置 DFlash2 不同，不能直接迁移其草稿执行器或用层数推算速度。

| 参考机制 | 自有实现要求 | 关键边界 |
| --- | --- | --- |
| Target + draft 均批处理 | Target 与 DFlash 各有真实的跨请求 GPU 调用 | 循环调用各请求的 proposer 不算 batched draft |
| 紧凑 ragged 行 | BatchPlan 显式记录有效区间、位置与状态句柄，按有效行压紧 | 不再为每请求分别补到 1/2/4/8；若算子/Graph 仍需物理 padding，单独计数并隔离 |
| 多请求 GDN/QSA | 自有设备元数据和执行入口，覆盖不同位置、长度及 parity | 不能只挂接 TensorFold 的内部 tables 就宣称自有实现 |
| GDN accepted-prefix 延迟归并 | StateManager 管理 committed/candidate 状态和 keep_rows | keep=0、拒绝后下一轮、取消、状态版本失效必须独立验证 |
| 有界、按实际组合复用 Graph | 自有 GraphManager，限制数量和显存；稳定形状才捕获 | key 包括地址/存储版本、代次、行布局、parity、context 与后端配置；在途 graph 不驱逐 |
| 短暂合批窗口 | DeepInfer 准入与 worker 联动，作为单独可测策略 | 按 TTFT/TPOT 收费；不照搬 3ms 为默认 |

参考位置：[verifyMany](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/656879fd752749bbd11a6e1a065a326c00940911/patches/0005-engine.patch#L3265)、[draftMany](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/656879fd752749bbd11a6e1a065a326c00940911/patches/0005-engine.patch#L3331)、[Graph 组合与缓存](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/656879fd752749bbd11a6e1a065a326c00940911/patches/0005-engine.patch#L3856)。

## 3. 下一阶段交付顺序

### A — 自有批次与状态契约，接通批量 DFlash

复用现有 request handle、generation 与事件 fence。定义与 TensorFold 对象无关的 BatchPlan/StateHandle，包含 request、位置、有效行范围、已提交位置、临时候选区和设备存储版本。后端接口提供 stage/verify/commit_prefix/discard/release；临时 TensorFold adapter 不外泄对象。

实现 `propose_batch`：不同请求可有不同 context/remaining，DFlash attention/KV 必须按请求隔离；投影、特征输出与 E17 预测恢复到各请求自己的行。保留 Q4 辅助头、W8、E17 和动态验证长度，不新增固定草稿运行模式。先以两个请求与现有逐请求草稿逐项对照，再扩展到四请求。

**验收：**proposal IDs、特征、预算输入和草稿状态有匹配参考；实际存在跨请求 GPU 计算；取消/EOS/剩余一 token 不读写其他请求。接口隔离本身只算迁移基础，不算已摆脱 TensorFold。

### B — 自有 Target 紧凑执行与 GDN/QSA 状态

实现自有 compact staging 和 Qwen StateManager，逐步替换 `stage/layer_forward/commit` 控制循环。先分别做 GDN、QSA，再串联完整 48 层；dense/MoE/PLE 算子可以经显式 provider 暂时复用，报告必须逐项标明来源与剩余依赖。

跨请求 GDN recurrence 和 QSA attention 需要设备端批处理，避免把 Python 中的逐请求循环藏在新后端名字之下。已有可兼容 DeepInfer 算子优先评估；缺失的 Qwen/SM121 能力由我们实现。算法学习与源码复用分别标注；若采用第三方 kernel，保留许可证和源码/二进制指纹。

**验收：**单层以及整模的 token/logit/feature/有效状态对照；包含不同 context、不同 parity、不同接受前缀、取消与 slot 复用。浮点计算顺序变化须单独报告误差及路由/输出影响，不能只报局部近似误差便宣布等价。

### C — 有界 Graph 与四请求完整服务

在 A/B 的实际布局上实现有界 Graph 缓存，不把当前双 slot 全排列预捕获机械扩展到四 slot。捕获在安全边界执行，不能额外提交候选状态；失败可回退 eager，销毁须等待设备完成。暴露捕获时间、命中率、显存、fallback 和主机提交耗时。

把当前两个 slot 的限制和双请求预算枚举分别扩展；联合预算仍应受 DeepInfer token/逻辑内存准入约束，枚举规模需有上限。E17 权重先保持不变，执行路径改变后重新校准 batch cost，旧费用表不能直接宣称适用。

**验收：**四请求真实 batched draft + Target + 异步 worker + SSE；覆盖不等长输入/输出、低高接受混合、自然 EOS、断连、慢消费者和 slot 复用。功能通过与性能晋升分开记账。

### D — 继续减少计算与加载依赖

按整模 profile 替换仍使用的 TensorFold PLE、dense/MoE、量化权重读取和 kernel provider。自有 engine 的完成要求是自行维护执行和状态协议；完全移除 TensorFold runtime import 的里程碑还必须覆盖加载、完整模型和设备状态，不能仅删除一个包名。

不为去依赖而一次性重写所有内核。每替换一层记录归属、正确性、显存和整模性能；允许使用经验证的 DeepInfer 或其他获许可通用算子。

## 4. 统一验收口径

- 同一设备、checkpoint、量化、KV 精度、提示词/输出限制、greedy/EOS 和预热条件；先保留当前 BF16 KV，FP8 KV 单独实验。
- 固定输入对比旧/新数据路径，再启用动态策略比较，避免把预算变化归因于执行器。
- 同时报 cohort 墙钟吞吐、每请求 TTFT/TPOT p50/p95、完整请求耗时和 tokens；server 各流速率之和单列，不与 cohort 吞吐混用。
- 冻结多样混合负载为主；同 prompt 多流单列诊断。参考仓库的同步同 prompt 测试不代表我们的混合任务速度。
- 保留 draft/verify/commit、主机空隙、实际/物理行、batch 大小、Graph 和状态轨迹。局部 kernel 更快、后端独立运行和 SSE 更快是三个独立结论。
- 全部真实 GPU 验证沿共享 resident 队列串行，不重载模型或修改其他实验的冻结模块。

这条路线可以同时减少串行/补齐/调度开销和 TensorFold 依赖；实际吞吐收益仍需上述对照建立，不预报最终 tok/s。
