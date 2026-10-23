# Kaifra Decode Frontier：实验验证矩阵、排序与止损规则

> **状态：PROPOSED / design-only。** 对应 [Decode Frontier RFC](KAIFRA_DECODE_FRONTIER_RFC.md)。本文件没有启动实验、训练模型或修改运行时。阈值均为**拟议**，要在看新一轮 GPU 输出**之前**冻结，不能事后挑选获胜配置。  
> **当前事实基线（2026-10-22 更新）：** E8 rows8/full-vocab Q4 + DeepInfer MTPVarlen（仍是单请求、有界 greedy），E9 未采样开销和 Nsight GPU 活动分析，以及已合入 `main` 的 E10–E16：E10/E11 的负结果、E12 的专家成本离线结果、E13 的草稿缓存、E15/E16 的 opt-in 置信头预算。它们都不算本 PR 的新数据。

## 1. 选择原则：有限时间优先买信息

优先级不是按想象中的最高理论加速排，而是按**实验代价、可证伪性、架构依赖、与当前瓶颈的匹配度**排序。

| ID | 研究方向 | 当前建议 | 最便宜的第一步 | 不需要先实现 | 状态（2026-10-22） |
| --- | --- | --- | --- | --- | --- |
| R1 | Expert-footprint / cost-aware verify | **P0**，复用 [PR #1](https://github.com/EdwinWang831/spark-infer-q/pull/1) | 测真实 Router 每层专家并集与时间的相关性，验证是否优于 rows-only | 不必训练 Predictor，也不必修改 Scheduler | 第一步已做（E12）：真实并集 MAE −26.17%，上一轮并集 −2.42%；在线暂缓 |
| R5 | Tiny acceptance/cost predictor | **P1**，依赖 R1 和 [PR #4](https://github.com/EdwinWang831/spark-infer-q/pull/4) | 冻结历史特征，离线 replay 正确性和成本 | 不必先蒸馏大模型 | 接受率部分已做单请求在线（E15/E16）：相对滞后预算 +10.00% / +9.61%；没有等 R1 和 #4 |
| R4 | Grammar / retrieval / n-gram DraftProvider | **P1**，可独立推进 | 用人工可解释的 JSON/引用/代码补丁任务比较 draft 成本和接受前缀 | 不必先训练新的 Drafter | 未开始 |
| R3 | 2-request GPU batch / ragged verify | **P1**，系统底座 | Target-only 两请求各 1 行，一次共享 Forward 严格对照两个 solo | 不必先增加全局预算策略 | 未开始 |
| R6 | DFlash context projection cache | **P2**，长上下文方向 | 离线量化重复投影时长随 prefix 增长的斜率 | 不必先实现跨模型 KV 复用 | 已做到 2K（E13）：长输入 +6.27%，短输入 −0.31%，opt-in |
| R2 | GDN-aware diffusion draft tree | **P2 / 高风险研究** | CPU recurrent branch/fork/commit toy oracle | 不必先编写 tree GPU kernel | 未开始 |
| R8 | Portfolio + joint control | **P2/后期**，需要可靠成本曲线与 GPU batch | 在冻结的真实 trace 上离线枚举少量 legal modes | 不必现在写端到端自适应大框架 | 未开始 |
| R7 | Spark Target + RTX5070 Draft | **P3 / 高风险探索** | 量化每轮 Feature bytes、增量协议与实际链路 RTT/带宽 | 不必先跑全模型分布式推理 | 未开始 |

核心要求：**不要把受已知依赖约束的 R2/R8 当成当前 E8 推理路径已经具备的特性。**

## 2. 分层验收：数学正确、设备正确、整机收益是三回事

| 验收层 | 应回答的问题 | 最低证据 |
| --- | --- | --- |
| S0: Offline model / oracle | 使用可观测的输入能不能预测成本/接受率？是否泄漏了同一步 Target 的真值？ | 固定拆分（按 prompt/任务聚类）与完整离线 replay，存 feature 列表、标签口径、误差 |
| S1: Component | 新 kernel/状态处理是否正确且有微基准收益？ | 精确引用 FP32/BF16/Target baseline、完整输出和尾部保护、独立随机种子、SM121 多 shape |
| S2: Full greedy decode | 完整请求是否保持 Target 决策和有效状态正确？ | 完整输出 IDs、每次已提交 logits（若仍计算）、有效 GDN/QSA/PLE state、EOS/停止原因、取消/恢复 |
| S3: Complete request | 真正使用起来是否更快？ | 有控制的 SSE TTFT、完整耗时、有效 token/s；按任务配对、多加载重复；报告采样/探针开销 |
| S4: Concurrent serving | 多 Agent 总工作量是否变好，又有没有拖慢交互用户？ | 真正同一个 GPU Forward 的 batch 证据、TPOT/TTFT p50/p95、公平性、显存/统一内存压力、取消/重用 |
| S5: Generalization / novelty | 是一个可研究的机制还是单任务微调？ | 新冻结中英/代码/数学/长短混合输入、不同模型/硬件（如果声称通用）、最新 TensorFold/SGLang/相关论文消融 |

**不允许跳级宣传**：S0 命中率的收益不是 S3 的速度；S1 kernel 倍数不是 S3 的完整请求加速；S2 greedy 正确性不代表随机采样分布无损。

### 通用基线和测量契约

- 预先固定：checkpoint+tokenizer SHA、量化方案、上下文上限、Prompt/Thinking 模板、冷/热 Prompt 规则、KV dtype、接受/拒绝/EOS、目标设备温度、模型/后端版本。
- 记录：每轮 draft/verify/commit 的同步 Host wall、设备 Kernel 时间、读写字节**估算**与来源、接受长度、请求结束 token 数、短尾/graph 桶、峰值 allocated/reserved 与进程/系统占用。
- 不同输出导致不同生成工作量时，保留完整轨迹差异；相同输出的子集对照属于诊断，不取代完整任务集。
- CPU launch/sync 区间与 GPU kernel 区间重叠，不能求和当总延迟；E9 的 GPU 活动覆盖率不等于真实 SM/DRAM 利用率。
- Nsight Compute 无法读取硬件计数器（`ERR_NVGPUCTRPERM`）时，不使用未经验证的“带宽打满”结论；在既有权限内记录可执行的只读带宽探针，但缓存与 DRAM 真实流量仍可能不可区分。
- 采用按**请求/任务聚类**的 bootstrap 或多加载配对波动区间；不要把同一请求几百个 token 当成独立实验样本。
- 计量报告同时给出自然 EOS、定长 cap、中英文/代码/数学与不同上下文区间；不要只保留最快的任务。

## 3. 各方向的第一阶段实验卡（不执行，仅定义门槛）

### R1：专家并集成本模型（与 PR #1 共用一套证据）

**冻结变量：** Target 未改；先只记录不同验证宽度下各层 Top-10 expert ID、Router route 和对应真实 kernel/step 耗时。

**关键对照：**
1. Baseline-rows：`time ≈ f(verified_rows, context_bucket)`；
2. Baseline-shapes：`time ≈ f(rows, context_bucket, tensor shapes)`；
3. Expert-union：在相同数据切分中增加每层专家并集或专家 pack 字节的标签；
4. **Online-feasible**：只输入本轮 Verify 之前能取得的轻量特征；不能用当前请求真实 Target Router 标签做上线决策。

**初始 Go 条件：**以 PR #1 事前阈值为准（其草案：held-out MAE 比 rows-only 至少低 25%）；同时完整预测器的运行成本低于预期每步收益。若模型不能胜过简单 rows baseline，应记作负结果，保持原 MTPVarlen 宽度策略。

**独立价值：**若成本预测有效，可在不更改 Target 数值的前提下优化“相同预计有效 token 产出下的预计专家权重访问”。需证明实际性能，不能只证明 Router union 随宽度增长。

**结果（E12）：**真实并集的 held-out MAE 比 rows+context 低 26.17%，区间 12.36%–35.26%，点估计过了 25% 门槛但区间没有全部过；而且真实并集不是在线可得的特征。在线可得的上一轮并集只低 2.42%，没有过门槛。按上面的规则记为在线 No-Go，保持原 MTPVarlen 的成本曲线。

### R2：树形推测与递归状态分叉

**离线实验：**用可控的 toy recurrent layer 构建分支，测试 `fork(parent)`、分支状态推进、`commit(winner)`、`discard(reject)`，覆盖首节点拒绝、深层拒绝、共享前缀、EOS、长度边界与异常取消。

**GPU 实验门槛：**
- 不同子分支共享父 GDN/conv state，但子分支推进必须隔离；
- QSA attention/index 和 PLE 状态按祖先依赖更新，而不是仅复制一张注意力 mask；
- Tree 不能在未定义分支执行顺序与状态语义时“拼成一个普通 batch”；
- Target 的 greedy 输出和提交后的全部有效 state 与独立 serial reference 对齐；
- 成本必须比较 **相同 node budget 的 DDTree/CaDDTree、单条 DFlash Q4 chain、target-only**。

**Go 条件：**先达到全分支状态正确，再出现可复现的 full-decode 净收益；若只能提高 acceptance 却显著增加 target rows/time，则保留研究模式、不扩大实现。

### R3：真实多请求 GPU 执行

**最小实验矩阵：**
- 两个请求 each 1 row，统一 `stage → compute → per-request commit`，状态逐位对照两次独立执行；
- 两请求固定宽度 4/4，允许其中一个在第一 draft 拒绝；
- 4/2、7/3 等不等长验证，合法 ragged metadata；禁止错误把 padding 当作有效行；
- 一个请求 EOS/另一个继续；取消后复用 slot；图 capture/replay 在持有不同请求的条件下不能污染有效状态。

**目标不是第一个版本就变成大服务。** 先取得一个可被 profiler 证实的**真正的共同 GPU Forward**；仅逐请求交错没有有效共享权重执行，不算完成。

**Go 条件：**双请求严格的输出/状态验收，加上相对于两条 serial Forward 的总吞吐净收益；再进入 4/5 请求，测 p95 TPOT 和延迟公平性。若吞吐上涨但前台延迟显著恶化，不设为默认。

### R4：语法/复制草稿的合法性边界

**任务分组：**严格 JSON schema；含显式用户授权引用的笔记整理；补全结构化 Markdown；代码补丁；开放式问答对照。

**三条基线：**Target-only、完整词表 Q4 DFlash2、低成本 `GrammarDraft/NgramDraft` 或简单选择器；每个请求使用相同约束/生成契约。

- **普通 greedy：**复制、语法、检索给出的只是 draft，必须经 Target 验证才能保持原输出；
- **受约束解码：**如果 grammar 强制某个 token，则参考是“同一 grammar 的合法生成”，**不是** unconstrained Target 的输出；利用已知 token 做 chunk state advance 时必须执行完整合法状态推进；
- **隐私与授权：**索引只读取该请求/用户有权读取的输入，不能跨租户泄露引用和关键词。

**Go 条件：**特定类型任务完整请求出现可重复的优势，且索引/Tokenizer/Provider 切换时间受控；若优势只来自重复已知文本，不外推一般对话。

### R5：轻量预测器和在线决策

**输入切断规则：**严格记录每个特征在 `choose_width` 时是否真实可获得，包括 selector margin（只有调用已支付的 Drafter 后才能知道）、历史接受分布、上一轮已观测路由、上下文桶和内存水位。**Pre-draft 控制不能读取 post-draft 的 selector 分数。**

**基线：**冻结 fixed-k、原 DeepInfer `MTPVarlen`、简单 EWMA、只基于 rows 的成本模型、当前 TensorFold 的成本感知深度选择（以对应版本/模型可用性为前提）。

**Go 条件：**在新冻结任务集上的真实 decode 净收益**足以覆盖推理/同步/缓存成本**，并且按请求聚类后的不确定性允许得出可信的改进结论；否则优先采用简单模型。

**与 PR #4 关系：**先确认官方 BF16 arithmetic / FP32 控制相对我们的 DFlash Backbone 是否确实提高草稿接受率，再决定需要何种额外信号；不要同时更换草稿模型与控制器。

**结果（E15/E16）：**在线 Go 条件在单请求上满足：冻结新任务上相对滞后的 MTPVarlen +10.00%（E15，6 个任务）和 +9.61%（E16，20 个任务，按任务族 bootstrap 的耗时下降区间 7.38%–10.46%），特征、传输和预测都计入耗时。上面的基线里，fixed-k（E15 的固定七草稿）和 DeepInfer `MTPVarlen` 已对照；简单 EWMA、只基于 rows 的成本模型和 TensorFold 的深度策略还没对照。训练没有等 PR #4 的 fidelity 诊断，这和上一条的顺序不同；该诊断以后执行时，保持控制器不变。

### R6：长上下文 DFlash Cache

**试验:** 冻结 short/medium/long 的实际合法上下文长度，衡量 `ctx fc/norm`、每层 history K/V、concat 与 Selector 耗时占比；仅对已经 committed 的 Target features 使用 cache。

**Go 条件：**在被允许的上下文桶有稳定的草稿成本下降，最终 Target 输出保持正确，额外 cache 不导致内存准入/并发退化。该实验应等到研究执行器支持更长、被验证的上下文时再全面运行；当前 E8 的容量上限不能冒充长上下文支持。

**结果（E13）：**在 1024/1536/1920-token 输入上 decode +6.27%，四轮配对都为正，状态检查通过；12 个短任务 −0.31%。长输入桶满足 Go 条件，短输入不满足，所以缓存保持 opt-in。2K 以上还没测。

### R7：双 GPU / 双电脑划分

**先测量**：Spark→远端 drafter 的单轮与增量特征字节数、网络有效吞吐/RTT、远端完整草稿执行成本、失败回退。参考为 Spark 同机 drafter，无需先重分布 Target 权重。

**Go 条件：**在真实链路下，完整 `Draft+Transfer+Queue+Commit` 成本**持续低于**单 Spark drafter 或多请求场景中有可复现的并行重叠收益；若失败，记录成本下界即停止。不要先重新开发分布式权重运行。

### R8：后期联合控制

**禁忌：**同时修改候选树、Q4、Target kernel、Batch、控制器后宣称“算法 +30%”。

**正确步骤：**
1. 单独冻结并验证 R1/R2/R3/R4/R5/R6 各自可行路径；
2. 固定 KernelExecutor 时做策略 A/B；固定策略时做 KernelExecutor A/B；
3. 消融 `no Expert Cost`、`no Mode Switch`、`no Tree`、`no Memory Guard`、`no Cross-request Alloc`；
4. 新冻结工作负载，在 TTFT/TPOT p95、总有效吞吐、完整任务结束时间、显存峰值构成 Pareto 改进时才保留复杂控制；
5. 若没有可靠的净收益，选择最简单、最好维护的合规退化路径。

## 4. 如何理解“突破带宽墙”

目标不是绕开物理带宽上限，而是尽可能提高**每次硬件流量带来的正确计算结果**：

```text
1) 减少无价值的 Target 验证次数/分支（仍须合法验证）
2) 同一次权重读取让多个请求或多个被接受的 token 受益
3) 降低草稿发生拒绝时的历史状态重算
4) 合理选择量化的投影和草稿模式，而非假设更低比特总是更快
5) 只在实测还有带宽/计算余量时继续写新 Kernel
```

在 MoE 上，专家 union 与实际 DRAM 字节不是同一个概念；缓存命中、tile 重读、内存映射/页故障和路由变化都可能破坏简单模型。必须根据 GPU 计数器（若可用）判断。E12 已说明专家 pack 占用既不是 DRAM 流量，也不是它的下界；没有计数器时只能当指示值，不把估算称实测。

## 5. 交付组织与 PR 边界

这份 RFC 建议以后按下列**独立** PR 切分（**不是**本次自动创建这些 PR）：

| 顺序 | 提交主题 | 验收 | 状态（2026-10-22） |
| --- | --- | --- | --- |
| A | 利用 #1 的 Expert Union/专家占用采集进入离线 Cost Model | held-out 核对、与 rows-only 对照；不改 live decode | E12 已完成，在线 No-Go |
| B | 利用 #4 的 Drafter BF16 Fidelity/Acceptance 对照 | 新冻结输入上的 accepted-prefix 改进证据或明确负结果 | 未执行，[方案](DRAFT_FIDELITY_ACCEPTANCE_PLAN.md)已在 `main` |
| C | Dual-request Target-only GPU Batch | 一次共同 Forward，两个请求的状态/logits 完整对照 | 未开始 |
| D | Grammar/Ngram provider 实验 | 明确 unconstrained / constrained contract 与隐私边界 | 未开始 |
| E | CPU Tree Fork/Commit Oracle | 分支状态正确，尚不宣传 GPU tree | 未开始 |
| F | Selector/Cost Predictor A/B | 单变量预算对照、有效 token/s 与完整请求时间 | 接受率部分由 E15/E16 完成 |
| G | Long-context Drafter Cache | 随 prefix 的真实成本曲线，严格状态与内存校验 | E13 已完成到 2K |
| H | Portfolio/联合优化 | 独立结果已可信，完整系统多维 Pareto 报告 | 未开始 |

### 这次 PR 能声称什么？

- [x] 八个 Decode 创新方向被系统整理，有实现依赖、强基线、正确性门槛、失败条件；
- [x] 明确承认 DeepInfer、TensorFold、EcoSpec、EVICT、DDTree、CaDDTree 和 llama.cpp 的相关现有工作；
- [x] 把已有 PR #1/#4（均已合入）接为前置工作，避免重复开发成本模型/草稿 fidelity；
- [x] 保留 E8/E9 真实结果、E10/E11 负结果，以及 E12–E16 的后续结果（2026-10-22 更新）；
- [ ] **没有**任何新的 GPU benchmark / 模型推理优化 / 部署兼容性验收；
- [ ] **没有**创建 GPU Batch、Draft Tree、多设备或 Portfolio 的实际实现。

## 6. 推荐第一步

**最低风险验证（2026-10-22 更新）：**PR #1 的 Expert Union 观察已由 E12 完成，在线可得的特征没有过门槛，按规则止损。下一个最便宜、可证伪的第一步是 R4：在 JSON、引用和代码补丁任务上离线比较低成本草稿的成本与接受前缀。  
**可并行开发：**E5 DeepInfer Scheduler → 两请求共享 GPU Forward 的状态正确性原型（先 Target-only，不用自适应草稿）。  
**最高研究不确定性：**GDN-aware Tree 验证；应从 CPU toy state fork 开始，不要直接构建完整 GPU 树。

**只有证明净收益以后，才把这些组件组合成 Kaifra 的长期引擎能力。**