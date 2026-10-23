# Kaifra — 自适应本地推理引擎愿景（Vision RFC v0.1）

> **状态：PROPOSED / 长期愿景与研究假设，不是已完成的引擎，也不是性能发布。**  
> **技术定位：Kaifra 是基于 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 发展和深度改造的自适应推理引擎（DeepInfer-derived），不是从零重写 Scheduler。** DeepInfer 的 C++ Scheduler 与 MTPVarlen 是我们的控制面和投机预算基础；TensorFold 的 Qwen 执行/SM121 CUDA 组件是当前计算后端来源；DFlash2 是草稿架构来源。上游归属和许可证必须保持明确。  
> **项目关系：** Kaifra 为提议中的引擎/产品名称；SparkInfer-Q 继续作为已有研究仓库和实验代号。**本 PR 不重命名仓库、不修改推理运行时、不宣称实现了 Kaifra。**  
> **目标设备：** 第一阶段聚焦单台 NVIDIA DGX Spark（GB10/SM121），未来再研究跨平台后端。  
> **评估基线：** TensorFold、SGLang、vLLM，以及同一 Kaifra 执行器里的 target-only、native MTP、Q4 DFlash2。  
> **最后核对日期：** 2026-10-22，按 `main` @ `51828be`（E4–E16 已合入）。对应具体提交、测试和许可证请以文末链接为准。

## 1. 一句话的愿景

**Kaifra = Adaptive Inference Engine：把“算哪些 token、用哪种草稿、什么时候验证、如何使用缓存、怎样共享 GPU”变成一个受正确性约束且可被实测成本驱动的统一决策。**

我们的愿望不是做一个仅在某条固定 benchmark 上跑得更快的 Qwen fork，而是**以 DeepInfer 为内核延伸**，在它的 Scheduler/预算之上完成 Qwen 执行、DraftProvider、多请求状态与资源策略的专门适配，逐步构建能够稳定服务本地 AI 与多 Agent 工作流的 **Kaifra 自主维护运行时**。它是深度改造的独立产品/系统，不是原创于零的全新推理框架。

> 不只优化每秒输出多少 token，而是优化在实际资源和延迟约束下，一台机器能完成多少**有用的 AI 工作**。

如果这个目标实现，Kaifra 应该能以清晰的性能证据回答：
- 现在该使用 target-only、native MTP 还是 DFlash2？
- 当前请求值得生成多长的草稿，又值得验证多少行？
- 当第二个/第三个 Agent 进入时，谁应先 Prefill，谁应继续流式 Decode？
- 128GB 统一内存中的权重、KV、GDN、草稿缓存和映射工作集怎样共同限制并发？
- 哪种工作负载值得使用更长草稿、完整 Q4 词表头或增量 DFlash context cache？
- 在性能增益低于噪声或正确性不确定时，是否应回退到更简单的已验证实现？

## 2. 基于 DeepInfer，怎样做出真正自己的引擎

**当前已经明确的依赖谱系：** E3 使用 DeepInfer 原生 `MTPVarlen` 进行动态验证预算；E5 的 C++ `Scheduler.schedule` / `process_outputs` 已驱动 Qwen target-only 的有界请求生命周期。我们现在有真实的 DeepInfer 组件接线，并非只模仿其调度思想。不过 E5 尚未接入完整 DeepInfer Worker、真实 GPU batch 或 DFlash2 多请求执行。

**架构选择：继续基于/改造 DeepInfer 原生 C++ Scheduler 与动态验证机制，而不是另起炉灶重复写一个同类 Scheduler。** Kaifra 负责修复其 DeepSeek 专用假设、适配 Qwen GDN/QSA 状态和 TensorFold kernel、补齐服务/执行能力；如确需替换部分原生组件，应说明原因、量化收益和兼容边界。

“自己的引擎”**不要求从零重写所有 CUDA kernel**。现代引擎可以合法复用上游算子，只要自己真正拥有并维护运行时关键机制。

我们把独立性分为三层：

| 层次 | 定义 | 现在的位置 / 目标 |
| --- | --- | --- |
| R0：实验组合 | Python 实验循环 + 外部 Target kernel + 已有推测算法 | SparkInfer-Q 的 E3–E16 属于受控研究原型；尚非完整引擎 |
| R1：基于 DeepInfer 的可独立服务运行时 | **复用/改造 DeepInfer C++ Scheduler 和预算器**，由 Kaifra 维护 Qwen 请求/状态适配、Batch Plan、Prefill/Decode/Verify、提交、API、指标与回退；kernel 允许模块化复用 | **Kaifra MVP 的合理定义** |
| R2：独立可优化的系统 | R1 加上自己的执行计划成本反馈、多请求联合调度、增量 draft cache、跨后端模型/草稿适配与版本化性能证据 | 长期技术壁垒；不承诺必然实现或领先 |

**R1 即可合理地称为 DeepInfer-based 自主推理引擎**：既如实声明来自 DeepInfer，也由我们独立维护和发布 Qwen/SM121 运行时。独立性不等于“与所有上游毫无代码关系”，而是运行时能力、设计边界和维护责任。把外部 TensorFold Python/Zig 运行时整个嵌进产品、自己只包装 API，则独立性不足；把合法授权的 CUDA 算子做成 KernelAdapter，并由 DeepInfer-derived Scheduler 驱动我们自己的 Qwen 执行/状态/验证代码，才是更有意义的路线。

不必为了“原创”重复实现已经成熟的矩阵乘法。真正需要由我们承担的是：**请求如何变成 GPU 工作、哪些状态有效、何时将验证结果变成可见输出、为什么选择这种计算模式**。

## 3. 我们已有的事实：亮点与缺口都要保留

以下实验报告最初写在独立开发分支上，现已全部合入 `main`（2026-10-22 核对）；它们不因写入愿景而自动成为生产功能。

| 里程碑 | 已记录结果 | 必须保留的边界 |
| --- | --- | --- |
| E3 | DFlash2 + DeepInfer 原生 `MTPVarlen`；八行/七草稿在单请求 greedy 上通过状态检查 | 尚非完整多请求 Scheduler |
| E5 | 真正的 DeepInfer C++ Scheduler 与 Qwen target-only 生命周期桥接；限定两请求交错场景通过 | GPU 仍逐个请求执行，不是联合批处理或完整 Worker |
| E6 | Q4 完整词表草稿头 + 原生动态预算，在同轮 8 任务中 45.54 vs BF16 41.70 decode tok/s，约 +9.23% | Q4 只是**草稿输出头额外副本**；backbone/selector 保留 BF16；整体内存不因此缩小 |
| E7 | 同机充分预热：8 个自然任务 Q4 DFlash2 45.57 vs 原生 MTP 43.58 tok/s；全部 10 任务 45.62 vs 49.18 tok/s | **整体 Decode 未超过原生**；完整请求中位耗时之和 38.141 vs 18.791 秒；跨引擎输出 8/10 摘要相同 |
| E8 prompt-only | 保留同一 Decode 算术，将 prompt 按 8 token 分块，10 任务 prompt 耗时中位数之和 19.679→4.220 秒，约 4.66×；81 组对照通过 | **尚不代表最终 SSE TTFT / 端到端加速**；完整 Q4 生成集成另见下一行 |
| E8 完整生成集成 | 新的 Q4 SSE 执行器默认 8 行 prompt 分块；TTFT 中位数之和 19.69→4.23 秒，完整请求 37.92→22.67 秒（1.67×）；40 次完整生成 oracle 通过 | decode 45.91→45.39 tok/s（−1.14%）；E8 之后没有重测原生 TensorFold |
| E9 | 未采样请求中 target 验证占 decode 的 83.81%，草稿占 14.39% | GPU 活动覆盖率 91.76% 不是 SM 利用率 |
| E10/E11 | 专家核调参 holdout +0.08%；投影 MXFP8 完整生成 decode +0.62%；都没有改默认 | 说明调参空间有限，不等于已到带宽上限（硬件计数器返回 `ERR_NVGPUCTRPERM`） |
| E12 | 本轮真实专家并集让离线成本 MAE 降 26.17%；只用上一轮并集仅降 2.42% | 真实并集要验证后才知道，不能在线使用；同前缀预算诊断最多 +1.28%，不是提速 |
| E13 | DFlash 历史投影增量缓存：1K–2K 输入 decode +6.27% | 短输入 −0.31%，默认不变；4K 以上未测 |
| E14 | 在线 margin 预算 +0.065%，区间跨零 | 深度先验下 MTPVarlen 几乎总选同一宽度 |
| E15/E16 | 训练的七深度置信头：冻结新任务上相对滞后预算 +10.00%（E15）和 +9.61%（E16，20 个中英文任务）；E16 比旧 E15 头再快 3.18% | 单请求 greedy、opt-in；没有 4K 以上、多请求或默认切换 |

这些结果至少说明三件事：
1. 完整词表 Q4 DFlash2 具有实际工程价值；这是继续开发的支点；
2. 不能从局部草稿加速推断完整系统领先：Prefill、GPU 调度、Context Attention 和状态复制都可能成为决定因素；
3. 目前测得的较大 decode 提升来自草稿和决策层：E6 的 Q4 草稿头 +9.23%、E13 的长输入缓存 +6.27%、E15/E16 的置信头预算约 +10%；target kernel 调参（E10/E11）只有 +0.08%/+0.62%。这和第 1 节“决定算哪些 token”的定位一致。不过这些收益都是单请求上各自测得的，不能相加。

我们坚持用**相同输入、权重、精度、请求模式、预热和计时边界**做公平评估，同时保留语义/数值输出差异、吞吐波动与负结果。

## 4. 最值得畅想的四类能力

### 4.1 一个“推测算法组合”的引擎，而不是固定一种算法

Kaifra 的 DraftProvider 可以实现：
- **target-only**：草稿价值不足、资源紧张或者安全回退；
- **native MTP**：利用模型原生已训练的 MTP 层，支持正确的增量状态；
- **DFlash2 + full-vocab Q4 head**：多 token 并行草稿、Selector 保持高精度，适合其单位成本/接受率有优势的工作负载；
- **未来的新草稿**：只有验证过完整分布/状态契约的实现才可以加入。

ModeController **先在请求级或低频安全点**选择，不做每 token 无代价切换的假设；切换须计算模型/草稿常驻、图捕获、状态重建和预热成本。

**成功的情形：**混合模式相对这三种模式各自最佳的合理配置，在预先冻结的新任务分布中形成可复现的综合优势。  
**失败的情形：**选择器的训练成本、误判和模式切换成本超过收益；那就默认更简单的单模式实现。

### 4.2 多个 Agent 共享单台 Spark 的“有用吞吐”

场景：前台 Knotera 在流式讲解论文，后台 Agent 同时读取文档、构建知识结构、生成代码。

Kaifra 希望把这些任务映射为不同约束：
- **交互请求**需要较低的 TTFT、TPOT p95、持续输出；
- **后台工作**更重视总任务完成时间和批量吞吐；
- **长 Prefill**不能永久阻塞已有流，后台又不能被饿死；
- **内存增长**不能令已经运行的 Agent 状态失效。

跨请求预算器应决定**每批 GPU 计算里哪些请求参与、每个请求分到多少验证行、用哪个合法 CUDA Graph bucket**，并以真实结果闭环修正估计。

但**TensorFold 现在也有成本感知的跨请求预算与动态草稿深度**。差异必须来自更好的工作负载模型/联合决策/具体实现，而非声称“首创成本调度”。

### 4.3 专门为统一内存设备设计执行策略

DGX Spark 的 CPU/GPU 共享 128GB 物理内存，不应把 RAM 与 VRAM 算成两份。我们希望低频资源控制器观察：
- 权重常驻与 draft 副本大小；
- GDN recurrent/conv 状态与 QSA KV/index 的有效增长；
- N-gram 映射表和 NVMe/文件页访问的实际停顿；
- CUDA 图、workspace、上下文预分配和并发；
- 系统余量、内存压力阶段与临时峰值。

高压力时，不一定增加并发就会提高吞吐；模式切换或请求准入必须计算**真实端到端成本**。用户态引擎不能保证控制所有操作系统页面换入换出，不能宣称支持未经验证的状态换出。

### 4.4 从“模型服务”提升到“可测量的计算实验平台”

每次推理可以产出可匿名/脱敏的研究指标：
- chosen provider、宽度和图 bucket；
- 预测 vs 实际的草稿接受率；
- 有效输出 token / 完整请求时间；
- drafter/verify/commit/prefill 各阶段耗时、GPU/CPU 同步；
- 峰值内存与排队延迟，p50/p95 TTFT/TPOT；
- 性能回退或状态校验失败原因。

Kaifra 的长期资产应当是**可复现的正确执行、经过版本化的成本曲线和真正领先的负载域**，而不是仅仅拼接第三方项目。

## 5. 我们能在哪些方面实质性地超过 TensorFold？

下列都是**竞争假设**，不是已经取得的优势；不能把 TensorFold 的旧版 E7 对照当最新上游的能力边界。

| 方向 | 需要做出的差异 | 如何证明，而不是靠命名 |
| --- | --- | --- |
| Speculative Portfolio | 同一 Target Executor 下，在 MTP / DFlash2 / no-draft 间智能选择 | 新冻结混合任务上，比所有单模式基线更优，并计入切换成本 |
| Global Workload Scheduling | 接受率、图 bucket、交互延迟、内存准入和 Prefill 竞争一起决策 | 2–5 个请求的总吞吐、TTFT/TPOT p95、饥饿率优于 TensorFold |
| Spark Memory/Compute Co-design | 根据测得的模型/状态/映射内存压力调整请求准入与验证预算 | 不同上下文/内存压力的 Pareto 曲线优于对照，零状态污染 |
| DFlash Context Reuse | 将可复用历史 Target feature/K/V 投影增量缓存，避免每轮重算 | 更长上下文的 Proposer 成本下降，接受/拒绝/EOS 下 state 严格对照 |
| Quantization-Aware Drafting | 联合考虑 Q4/Q8/词表覆盖、草稿质量、成本与调度 | 最终 tok/s/质量/内存 Pareto 优于单纯缩词表或固定精度 |
| Stable Local Serving | 强约束 correctness、明确恢复/回退、可替换 kernel provider | 长时稳定性、取消竞态、并发、一致性与版本回归有证据 |

截至 2026-10-22 的进展：DFlash Context Reuse 已有 E13 的首个结果（1K–2K 输入 +6.27%，短输入 −0.31%，仍为 opt-in）；Quantization-Aware Drafting 只有 E6 的 Q4 草稿头，Q8 和混合精度未测；其余四项还没有实验。

**不要把“使用 DeepInfer”、“支持 cost-aware scheduling”、“支持 CUDA Graph”、“有 Q4 草稿头”单独写成原创算法。** 这些已有先例和上游归属。我们独立的机会在于**具体的新决策问题、新接口约束、新成本模型、新执行算法以及可复现实验结果**。

### 一个值得研究的新问题：资源约束下的联合优化

令 `m_i` 为第 i 个请求选择的草稿模式，`k_i` 为验证深度，`b` 为图 bucket，`a_i` 为请求准入/服务决定；探索：

\[
\max_{\mathbf a,\mathbf m,\mathbf k,b}
\frac{\sum_i a_i\,w_i\,\mathbb{E}[\mathrm{useful\ tokens}_i(m_i,k_i)]}
{\widehat T(\mathbf a,\mathbf m,\mathbf k,b,\mathrm{context},\mathrm{memory})}
\]

约束：
- 每个已选请求至少有合法的主模型 anchor；
- 任何草稿接受都不改变 Target 的合法输出/采样分布；
- GDN/QSA 状态只能提交有效前缀，Pending token 与 committed position 分离；
- 图 bucket、内存预算和 worker 能力均受实际验证范围约束；
- TTFT/TPOT 尾延迟、公平性和最低进度可被明确报告。

**这是一种研究方向，不等于数学形式本身有新颖性**。实际研究必须查阅同类论文、SGLang/vLLM/TensorFold/DeepInfer 最新代码，设计 ablation 才能评估算法原创性。

## 6. 怎样才算“创新”：明确四级主张

| 级别 | 什么证据足以支持 | 不允许的表述 |
| --- | --- | --- |
| 可复现集成 | Qwen + Q4 DFlash2 + 原生预算在真实 Spark 上跑通，正确性与速度记录 | 将 DFlash/DeepInfer/TensorFold 的算法和 kernel 说成自行发明 |
| 系统/工程创新 | 自己实现可独立部署、可替换的执行器与完整服务，复杂 workload 存在净优势 | 仅靠新名称宣称有竞争壁垒 |
| 新调度机制 | 新的联合决策算法、正确性约束、成本模型与严谨消融；超过最新强基线 | 仅从目标函数看起来新就宣称原创 |
| 研究论文/通用贡献 | 跨模型/硬件/负载、置信区间与开源复现，排除已有工作并解释失败条件 | 从单 Spark、八个提示推断普遍 SOTA |

**我们可以现在就承认已有工程研究贡献；算法原创性需要后续更强证据。** 与成熟引擎并存而非全部负载取代，也可以是有价值的结果。

## 7. 由现在走向 Kaifra 的三个阶段

### Horizon A：追平可用性（先把地基补齐）

- 正式冻结 E7 同机对照，以 E8 正确的分块 prompt 路径做完整 Q4 请求集成验收；不要把 prompt-only 4.66× 当端到端结果（进展：E8 集成已完成，完整请求 1.67×；E8 之后还没有重测原生 TensorFold）；
- 解决 Target Prefill 与同路径 Serial 数值/状态差异；引入优化验证与 Target CUDA Graph，维护 rejected-prefix 回退与状态一致性（进展：E8 的分块 prompt 沿用 decode 算术并通过对照；对照 FP32 参考的 [prefill 数值方案](PREFILL_REFERENCE_PLAN.md)未执行；Target CUDA Graph 未做）；
- 建立有状态单请求 API（streaming、cancel、EOS/max length、指标与回退）；
- 用同一执行器的 MTP/DFlash2/target-only 对照区分算法差异和运行时差异。

### Horizon B：自主引擎 MVP（最值得正式叫 Kaifra 的节点）

- **从 E5 已接通的原生 DeepInfer C++ Scheduler 出发**，保留可用的 Scheduler/BlockManager/MTPVarlen，扩展 Qwen 请求状态、执行计划与服务接口；不是再写一个同质 Scheduler；
- 通过清晰的 `ExecutorBackend` 接口管理外部 CUDA 算子，不依赖它们原来的 serving loop；
- 实现双请求 **一次真正的 batched GPU target forward**，而非仅交错两个串行请求；
- 接入可回退的固定长度与 DFlash2 动态预算，封装代码为自己的执行计划 / 生命周期；
- 经正确性和 API 实测后，发布 `Kaifra 0.x experimental`，仍可借用 TensorFold 算子。

**完成这层就可以称为我们独立维护的 DeepInfer-derived 系统级推理引擎**，但需明确上游贡献，无需虚构自己发明了 Scheduler 或 GPU 矩阵乘算法。

### Horizon C：证明为什么需要 Kaifra

- 2/4/5 多请求混合 Prefill/Decode，正确的 ragged/state 管理和公平性；
- 多模式组合决策，完整 warm/cold/steady benchmarking；
- Memory Guard、成本反馈、上下文相关 DFlash 缓存；
- 选定少数关键场景稳定超过最新 TensorFold/SGLang/vLLM，并客观列出不如对方的场景；
- 将能复现的算法与系统方法整理成技术报告/论文，分享设计决策和失败结果。

15%–30% 的特定混合负载优势可以作为**探索性的 stretch goal**，不应作为性能预测、默认承诺或宣传结论。没有可复现优势时保留成熟后端作为 fallback。

## 8. 开源策略、边界与署名

我们的自主架构不能靠模糊来源建立声誉。适配器、代码与模型权重要分别遵守各自许可和 NOTICE：

- [SparkInfer-Q](https://github.com/EdwinWang831/spark-infer-q) 目前 Apache-2.0；
- [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 原生预算和 Scheduler 是上游算法/实现，需保留其适用许可与声明；
- [TensorFold](https://github.com/ashhart/TensorFold) 从 0.6.0 起 Apache-2.0，较早版本及部分文件存在 MIT 历史与第三方 notices，应逐文件核对固定源修订；不能把 MiaAI-Lab 配方的 MIT 覆盖上游所有代码；
- [MiaAI-Lab DGX Spark 配方](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) 自有代码为 MIT，其 patch 的上游部分仍受上游许可证；
- DFlash2 草稿 checkpoint、Qwen/NVIDIA 模型权重及第三方内核的许可分别核对，商业部署需要另外评估对应条件。

**建议：**延续 Apache-2.0（是否独立新仓库后再决定），上游 kernel 用适配与清晰依赖边界，版权/NOTICE 按源文件保留。不要复制未经许可的私有内核、权重或构建产物。

## 9. Kaifra 名称只是工作名

**Kaifra** 简短、易记，可与 SparkInfer-Q 的实验时代区分。但项目名称的 GitHub 可用性不代表公司、域名或商标可用，也不构成法律意见。

初步公开信息在[澳大利亚官方 ABN 登记记录](https://abr.business.gov.au/ABN/View?abn=35137552508)中可查到 **Kaifra Pty Ltd**（澳大利亚商业实体），因此不能宣称名称在全球独占或完成商标清查。正式产品、域名、App/公司使用前，应针对目标市场及软件/AI 服务类别做独立商标、域名和开源生态冲突检索。

本 RFC **不重命名现有 GitHub 仓库、不更改 package/import、不更改许可证、不发布产品品牌**。所有命名操作单独讨论与评审。

## 10. 成功的愿景是什么样的？

理想情况下：

> 一个独立开发者带着一台 DGX Spark 和自己的本地知识/Agent 工作流，可以安装 Kaifra。它不需要懂 DFlash、MTP、CUDA Graph 或 KV/GDN：引擎会根据**可测量且安全**的条件决定计算方式，报告为什么这么选，并在收益不可靠时选择简单稳定的路径。

它不是“永远最快”的神话，而是一台能**为真实任务做出更好计算决策**的本地智能计算机器。

**是否值得作为独立引擎继续投入的判断标准：**至少具备自有运行时和可验证的 API/状态生命周期；能与 TensorFold 在同机新任务上公平比较；在某个有实际用户价值的工作负载域构成经重复验证的性能/延迟/资源优势。缺少这些证据时应继续作为 SparkInfer-Q 实验项目迭代。

---

## 可追溯的参考与后续交付

- [E5：原生 Scheduler / Qwen 的限定桥接](E5_ENGINE_BRIDGE_REPORT.md)
- [E6：完整词表 Q4 草稿头](E6_DRAFT_HEAD_REPORT.md)
- [E7：对照原生 TensorFold 的正式报告](E7_TENSORFOLD_MATCH_REPORT.md)
- [E8：分块 Prompt 的限定实测](E8_PREFILL_REPORT.md)；[完整生成集成](E8_INTEGRATION_REPORT.md)
- [E9–E16 的报告与未执行方案的索引](EXPERIMENTS.md)
- [此前的 Global Cost-Aware Scheduler RFC PR #2](https://github.com/EdwinWang831/spark-infer-q/pull/2)
- [TensorFold 当前独立成本预算与多请求分配](https://github.com/ashhart/TensorFold/blob/d1018b318fb51981c4a18f12e40269a4d3ee9713/zig/src/core/lanes/allocate.zig)；[按深度估算收益与成本](https://github.com/ashhart/TensorFold/blob/d1018b318fb51981c4a18f12e40269a4d3ee9713/zig/src/core/lanes/depth.zig)
- [DeepInfer 源码](https://github.com/deepseek-ai/DeepInfer)

**下一步的可实施工程契约、依赖顺序、性能门槛与放弃条件参见** [Kaifra 实施路线](KAIFRA_IMPLEMENTATION_ROADMAP.md)。