# DeepInfer 与自有 Qwen/Spark 执行后端

2026-10-24 方向更新。本文定义实现与验收计划；已接通[自有批量 DFlash 研究后端](K4_OWN_BATCH_DRAFT_REPORT.md)和[紧凑 Target 研究后端](K4_OWN_TARGET_REPORT.md)。自有 Target 覆盖状态、完整层编排、批量 GDN recurrence/commit、dense-budget QSA 和 PLE 卷积；仍经显式 provider 使用未迁移算子及 eager prefill。四请求完整服务和性能晋升尚未通过。此前证据见 [K4 功能报告](K4_BATCH_RUNTIME_REPORT.md)、[等待修复](K4_WAIT_FIX_REPORT.md)和[联合预算实验](K4_JOINT_BUDGET_REPORT.md)。

用户明确要求：学习 TensorFold 的技术思路，更多使用 DeepInfer 和我们自己的实现。最终架构以 **DeepInfer 调度底座 + 自有 Qwen/SM121 执行后端**为目标；TensorFold 是迁移期间的参考和可替换适配器。增加对 TensorFold 内部执行器的接线，不作为自主后端的完成标准。

用户进一步要求“做得更好”：自主实现和性能优势分别验收。目标是在真实混合并发、尾延迟和资源可控性上建立优势，并利用 E17 与 DeepInfer 调度作出适合我们负载的决策。

## 1. 当前能力和真实依赖

| 能力 | 当前实现 | 后续归属 |
| --- | --- | --- |
| 准入、Sequence、schedule/process_outputs | DeepInfer 原生 C++ Scheduler；我们的请求/代次适配 | 继续适配 DeepInfer，保留来源与上游边界 |
| 内存配额 | DeepInfer BlockManager，仅逻辑记账 | 调度配额继续复用；设备 KV/GDN 存储由自有 Qwen StateManager 管理 |
| 动态预算 | E17 本轮预测；已有 MTPVarlen 接线；当前双请求联合成本选择器为自有 Python 实验实现 | 保留 E17 与动态预算，适配真实批次成本，再推进 DeepInfer 跨请求预算集成 |
| 请求、异步 owner、取消、SSE、验证发布 | 自有 Kaifra runtime | 继续使用，扩展到四请求生命周期 |
| 草稿 | 自有 DFlash2/Q4/W8 接入和特征/预测器；默认逐请求，另有自有 batch 研究开关 | 验收 batched DFlash 数值变化、成本校准和稳定服务收益 |
| Target joint forward、Graph/event 编排 | 默认 legacy；可选 compact 自有 stage/层循环/commit 与有界 Graph | 四请求完整服务及冷 Graph 成本验收 |
| GDN/QSA/PLE 状态及执行 | compact 使用自有固定容量 State、GDN recurrence/批量 commit、dense-budget QSA、PLE 卷积 | 稀疏 QSA、长上下文和更多模型配置 |
| 权重读取、prefill、dense/MoE/HC/GDN IO/PLE embed-gate | TensorFold 加载、eager prefill 和显式 kernel/workspace provider | 按整模 profile 继续迁移并独立验收 |

默认 `JointTarget` 虽合并了 dense/MoE 行，仍不等同于完整的跨请求 GDN/QSA 融合执行。可选 `CompactTarget` 已在这些部分使用自有设备元数据和批量内核。自有草稿 batch 有四请求组件证据；完整服务最大请求数仍为 2，默认仍是 legacy Target/逐请求草稿。不能据此宣布 DeepInfer 已提供完整 Qwen 模型后端。

DeepInfer 已核对的模型实现以 DeepSeek 为主，没有现成的本模型 GDN/QSA 后端。GB10/SM121 上通用 RMSNorm 与选定 cuBLASLt 路径有执行证据，但固定 DeepGEMM 原生 BF16 dispatch 报 Unsupported architecture。单专家的 native NVFP4 测试使用 TensorFold 原版 CUDA 源码，不能记作自研内核或全模型 DeepInfer 兼容。详见 [E0](E0_REPORT.md)、[E1](E1_GPU_REPORT.md)、[E2 native](E2_NATIVE_REPORT.md)。

## 2. 学习机制，明确我们要实现什么

参考仓库固定为 `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold@656879fd752749bbd11a6e1a065a326c00940911`。其中 native MTP 与我们的外置 DFlash2 不同，不能直接迁移其草稿执行器或用层数推算速度。

| 参考机制 | 自有实现要求 | 关键边界 |
| --- | --- | --- |
| Target + draft 均批处理 | Target 与 DFlash 各有真实的跨请求 GPU 调用 | 循环调用各请求的 proposer 不算 batched draft |
| 紧凑 ragged 行 | BatchPlan 显式记录有效区间、位置与状态句柄，按有效行压紧 | 不再为每请求分别补到 1/2/4/8；若算子/Graph 仍需物理 padding，单独计数并隔离 |
| 多请求 GDN/QSA | 自有设备元数据和执行入口，覆盖不同位置、长度及 parity | 不能只挂接 TensorFold 的内部 tables 就宣称自有实现 |
| GDN accepted-prefix 延迟归并 | StateManager 管理 committed/candidate 状态和 keep_rows | keep=0、拒绝后下一轮、取消、状态版本失效必须独立验证 |
| 有界、按实际组合复用 Graph | 自有 GraphManager，限制数量和显存；稳定形状才捕获 | 静态权重/工作区绑定执行器生命周期；state 地址、布局、parity、位置经设备表间接寻址，代次由 ticket 检查。key 为实际行数/context；在途 graph 不驱逐 |
| 短暂合批窗口 | DeepInfer 准入与 worker 联动，作为单独可测策略 | 按 TTFT/TPOT 收费；不照搬 3ms 为默认 |

参考位置：[verifyMany](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/656879fd752749bbd11a6e1a065a326c00940911/patches/0005-engine.patch#L3265)、[draftMany](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/656879fd752749bbd11a6e1a065a326c00940911/patches/0005-engine.patch#L3331)、[Graph 组合与缓存](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/656879fd752749bbd11a6e1a065a326c00940911/patches/0005-engine.patch#L3856)。

## 3. 下一阶段交付顺序

### A — 自有批次与状态契约，接通批量 DFlash

复用现有 request handle、generation 与事件 fence。定义与 TensorFold 对象无关的 BatchPlan/StateHandle，包含 request、位置、有效行范围、已提交位置、临时候选区和设备存储版本。后端接口提供 stage/verify/commit_prefix/discard/release；临时 TensorFold adapter 不外泄对象。

实现 `propose_batch`：不同请求可有不同 context/remaining，DFlash attention/KV 必须按请求隔离；投影、特征输出与 E17 预测恢复到各请求自己的行。保留 Q4 辅助头、W8、E17 和动态验证长度，不新增固定草稿运行模式。先以两个请求与现有逐请求草稿逐项对照，再扩展到四请求。

**验收：**proposal IDs、特征、预算输入和草稿状态有匹配参考；实际存在跨请求 GPU 计算；取消/EOS/剩余一 token 不读写其他请求。接口隔离本身只算迁移基础，不算已摆脱 TensorFold。

### B — 自有 Target 紧凑执行与 GDN/QSA 状态

两请求研究实现已接通，限定文本/BF16 KV、2048 内 dense-budget QSA；细项和性能结论见[本轮报告](K4_OWN_TARGET_REPORT.md)。这是部分自主执行与数值验收，未完成全部算子或加载迁移。

实现自有 compact staging 和 Qwen StateManager，逐步替换 `stage/layer_forward/commit` 控制循环。先分别做 GDN、QSA，再串联完整 48 层；dense/MoE/PLE 算子可以经显式 provider 暂时复用，报告必须逐项标明来源与剩余依赖。

跨请求 GDN recurrence 和 QSA attention 需要设备端批处理，避免把 Python 中的逐请求循环藏在新后端名字之下。已有可兼容 DeepInfer 算子优先评估；缺失的 Qwen/SM121 能力由我们实现。算法学习与源码复用分别标注；若采用第三方 kernel，保留许可证和源码/二进制指纹。

**验收：**单层以及整模的 token/logit/feature/有效状态对照；包含不同 context、不同 parity、不同接受前缀、取消与 slot 复用。浮点计算顺序变化须单独报告误差及路由/输出影响，不能只报局部近似误差便宣布等价。

### C — 有界 Graph 与四请求完整服务

两请求有界 Graph 已实现：默认最多 32 张、第二次遇到形状才捕获、LRU 驱逐；尚未完成四请求服务扩展与未知形状的冷延迟验收。正式 warm 对照排除新增 capture/JIT，不能视作冷请求延迟承诺。

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

## 5. “做得更好”的工程目标与晋升要求

以下数值是本阶段预设的工程目标，不是已测结果或用户承诺。评测负载、阈值和重复方案在正式运行前冻结；后续若调整必须留痕，不得看完结果再放宽门槛。

| 维度 | 目标与判定 |
| --- | --- |
| 正确性 | 先保持模型、精度与 greedy 契约，完整 tokens、logits/features 与有效状态检查通过；数值变化单独验收，不拿接受率代替输出质量 |
| 真实服务吞吐 | 主评测为 2/4 请求的不同 prompt、不同长度与接受率混合负载。相对当前优化服务，以 cohort 墙钟吞吐提升至少 10% 为阶段目标；低于目标如实报告，不冒称已超过 |
| 尾延迟 | 同负载、到达率和并发条件下，TTFT 与 TPOT p95 相对基线恶化不超过 5% 为初始晋升约束；吞吐增加但尾延迟不达标则保留实验开关 |
| 扩展效率 | 报告 2→4 请求总吞吐变化、各流速率与最慢请求完成时间，查明新增请求的串行草稿、padding、内存和同步成本；不以并发数本身证明扩展成功 |
| 可控资源 | Graph 数量与显存、候选状态和输出队列均有显式上限；长时取消/复用后状态可回收，资源水位不随完成请求数持续增长 |
| 自主后端 | 报告调度、状态、执行循环、算子与加载器的来源和剩余依赖；接口封装、独立运行、超越参考三项分别标记 |

性能至少做 5 次独立 cohort 重复，交错旧/新运行顺序，保留全部样本并按 cohort 计算配对差异和 95% 区间。吞吐收益区间下界须大于零，尾延迟需要足够请求样本；样本不足以判断时标为未建立结论，不把 token 当作独立请求样本。共享设备受干扰的运行保留原记录并按预先规则单列，不择优删除。

对照分两层：

1. **执行路径对照**：当前优化服务 vs 自有后端，保持相同 DFlash/E17、冻结的候选/验证轨迹和计时边界，定位数据路径收益；随后恢复在线动态决策，重新测服务结果与成本。
2. **系统对照**：自有完整服务 vs 固定版本 TensorFold，在同机、同模型、相同输入输出契约与可匹配精度下测量。原生 MTP 与 DFlash 不同须披露；系统速度差异不能归因于单个调度器或预测器。不能匹配的配置单列，不直接宣布领先。

优先争取三项设计增量：将 E17 的输入相关接受率与实测 batch 成本联合使用；让异长请求通过紧凑批次共享计算；让 DeepInfer 的准入和公平性约束控制合批与验证预算。各项需独立消融，复杂机制只有带来完整服务收益才晋升默认。

这条路线可以同时减少串行/补齐/调度开销和 TensorFold 依赖；实际吞吐收益仍需上述对照建立，不预报最终 tok/s。
