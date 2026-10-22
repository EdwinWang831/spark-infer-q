# Kaifra Decode Frontier：突破带宽墙的研究议程（RFC v0.1）

> **状态：PROPOSED / docs-only。** 这些是待验证的假设、候选设计与实验，不是新算法已经实现、性能已经取得或模型质量已获证明。  
> **研究底座：** DeepInfer-derived Scheduler / MTPVarlen + TensorFold-derived Qwen Target/CUDA kernels + Q4 DFlash2；长远目标是由 Kaifra 维护的独立推理服务。  
> **目标设备：** 单台 DGX Spark (GB10/SM121)，之后单独研究多设备；当前主要研究 `text-only / greedy / BF16 KV`。  
> **版本基点：** 原稿写于 `main` @ `9727154`（2026-10-21）；2026-10-22 按 `main` @ `51828be` 更新，E10–E16 已合入，R1/R5/R6 已有实测结果（见各节的“状态更新”）；名称 Kaifra 是暂定工作名。  
> **配套：** [验证矩阵和实验优先级](KAIFRA_DECODE_VALIDATION_MATRIX.md)。与 [Kaifra 引擎愿景 PR #3](https://github.com/EdwinWang831/spark-infer-q/pull/3) 及 [Global Scheduler PR #2](https://github.com/EdwinWang831/spark-infer-q/pull/2) 互补，**不替代它们**。

## 1. 研究问题：不是每个 Kernel 更快，而是每次 Target 成本带来多少正确输出

在单请求、低验证行数、大模型权重与统一内存带宽限制下，普通 Warp/Tile 微调不一定产生明显的 Decode 收益。我们想改变的不是 Target 的定义，而是**每次 Target 执行做多少有价值的验证、读取多少真正需要的专家权重、如何跨请求复用计算与内存**。

在严格的 Target 等价约束下，以完整服务的已提交（用户可见）token、总 GPU 时间/完整请求时间及尾延迟为主要目标，而不是只最大化候选长度或离线 accepted-rate。

一个用于讨论的目标式（不是新颖性主张）：

```text
decision = (provider mode, candidate path/tree, verify rows, request batch, memory state)
score    = expected useful committed tokens / predicted complete-round latency

constraints:
  target-correct greedy outputs / valid committed state
  verified kernel shape + GDN/QSA/PLE branch and request isolation
  memory/admission + latency/fairness bounds
  switching / draft / target / graph capture / synchronization costs all charged
```

注意 `expected useful committed tokens` 是**决策前可用特征的条件期望**，不能使用本轮 Target Router 真值或实际接受前缀“预知未来”。Offline-oracle 只能报告上界和模型误差，不能用作线上对照成绩。

### 为什么现在讨论这件事

| 已提交证据 | 数据 | 含义 / 边界 |
| --- | --- | --- |
| [E6 Q4 draft head](E6_DRAFT_HEAD_REPORT.md) | 同任务单请求 Q4 动态 45.54 vs BF16-full 41.70 decode tok/s（+9.23%） | 草稿输出头的优化有效；不是整模量化或完整服务领先 |
| [E7 对照原生 MTP](E7_TENSORFOLD_MATCH_REPORT.md) | 自然任务 DFlash2 45.57 vs native MTP 43.58；10 任务整体 45.62 vs 49.18 tok/s | 单请求总体尚未领先原生 |
| [E8 完整分块 Prompt](E8_INTEGRATION_REPORT.md) | 同任务请求耗时之和 37.917→22.665 秒（减少 40.22%）；Decode 未提升 | Prompt 路径已有关键改善，剩余 Decode 需要单独攻坚 |
| [E9 Nsight Decode](E9_DECODE_PROFILE_REPORT.md) | 正常未采样数据 Target Verify 占 83.81%；三个 GPU 样本活动区间 91.76%；5,492 验证行对 3,348 已提交行 | 活动率不是 SM/DRAM 利用率；2,144 未提交行不是可无损直接省掉的 39.04% 时间 |
| [E10 专家核调参](E10_TARGET_KERNEL_REPORT.md) | holdout 缓存扰动复核最佳 Warp/prefetch 约 +0.08%，未升级默认 | 该搜索空间局部收益有限；不等于证明已触带宽上限 |
| [E11 投影](E11_PROJECTION_REPORT.md) | 完整生成 Decode 45.445→45.727 tok/s，+0.62%；部分任务退化 | 微基准局部 2× 不能直接外推端到端；候选未成为默认 |
| [E12 专家感知验证](E12_EXPERT_VERIFICATION_REPORT.md) | 本轮真实专家并集让离线成本 MAE 降 26.17%；只用上一轮并集仅降 2.42%；同前缀预算诊断最多 +1.28% | 真实并集不能在线使用；诊断不是 decode tok/s |
| [E13 草稿上下文缓存](E13_DRAFT_CACHE_CONFIDENCE_REPORT.md) | 1K–2K 输入 39.73→42.22 tok/s（+6.27%）；短输入 −0.31% | opt-in，默认不变；4K 以上未测 |
| [E14 在线 margin 预算](E14_CONFIDENCE_BUDGET_REPORT.md) | +0.065%，区间跨零；各轮的宽度序列相同 | 深度先验下 MTPVarlen 几乎不随输入改选宽度 |
| [E15](E15_CONFIDENCE_TRAINING_REPORT.md)/[E16 置信头](E16_BILINGUAL_EXPANSION_REPORT.md) | 冻结新任务上相对滞后预算 +10.00% / +9.61%；E16 比旧 E15 头再快 3.18% | 单请求 greedy、opt-in；没有多请求或 4K 以上 |

E10 的 Nsight Compute 硬件计数器因 `ERR_NVGPUCTRPERM` 未获得，**不能认定已证实达到 273 GB/s 的硬件带宽上限**；也不能把 E9 的 GPU 活动覆盖率等同于带宽利用率。以上数值仅对应其报告的设备、版本、任务与测量口径。

## 2. 八个有想象空间、但可被证伪的方向

### R1 — Expert-Footprint-Aware Verification（MoE 专家并集成本感知）

**大胆假设：**两组同样大小的候选、相近的预计接受长度，若路由到的专家更重合，验证可能少读一些分散权重，并获得更好的有效 token/秒。

- **成本模型：**在每一层分别统计当前验证行实际命中的 `top-k` 专家集合，累加各层的唯一专家数量、权重 pack 字节与路由密度。不同层的同名专家 ID **不能合并为同一个专家**。
- **第一步只观测：**从已执行 Target Forward 得到真实路由与时间线，检验 `rows`、`expert_union_per_layer`、`context`、`routing locality`、`verify latency` 的相关性。真实路由可以作为回归标签，**不得作为同一步线上提前选择候选的输入**。
- **第二步才预测：**尝试用 `DFlash selector margin`、当前候选 token、上一轮路由、上下文摘要等**验证前可获得**的特征预测边际专家访问成本。预测器的 FLOPs、常驻内存和 GPU/CPU 同步都计入成本。
- **第三步联合预算：**用预测 `cost(k)` 替换或补充 DeepInfer MTPVarlen 中仅按宽度插值的成本曲线；先保留相同接受率先验，做**单变量**宽度选择 A/B，再研究候选树。
- **决策门槛：**Held-out 成本预测必须优于 rows-only；完整 Decode 在相同输出契约下优于已校准的固定深度和原始 MTPVarlen，否则不升默认。

**已有工作/避免重复：** [EcoSpec](https://arxiv.org/abs/2607.12696) 已提出预测专家重用减少验证成本；[EVICT](https://arxiv.org/abs/2605.00342) 已提出 MoE 推测树提前裁剪。因此单独“算专家并集”不具新颖性。仓库 [PR #1](https://github.com/EdwinWang831/spark-infer-q/pull/1)（已合入）**已有**专家占用核算（原称带宽核算，E12 之后改为占用速率）、专家并集预算计划和 Host-only 参考代码。本 RFC 的 R1 将其作为前置工作，不创建第二套 `moe_cost.py` 或重复代码。

**状态更新（2026-10-22）：**[E12](E12_EXPERT_VERIFICATION_REPORT.md) 完成了第一步。本轮真实专家并集让 held-out 成本 MAE 降 26.17%（prompt bootstrap 区间 12.36%–35.26%，点估计刚过 25% 门槛，区间没有全部过）；按第二步的思路只用上一轮 width4 的并集，仅降 2.42%。在固定接受率先验下，即使直接知道真实 action 成本，同前缀预算诊断也只有 +1.277%。所以在这些上下文长度上，只改成本模型的预算收益很小；第二、三步按[方案](MOE_AWARE_BUDGET_PLAN.md)的停止规则暂缓，除非出现明显更好的验证前特征。这个 +1.277% 是在固定先验下算的，换成 E16 的置信头后需要重算。

### R2 — GDN-Aware Diffusion Draft Tree（带递归状态的成本感知验证树）

**大胆假设：**同一个 DFlash2 block 不只交给 Target 一条草稿路径，而是保留几个“接受概率高 / 新增验证成本低”的分支；通过更高概率命中的分支减少完整 Target round 次数。

- DFlash2 可给出不同位置的候选信息，但现在的验证主要沿着 Selector 选出的一条路径；可离线构造固定 `node budget` 的 Top-N 树，研究 `expected committed tokens / (draft + target)`。
- **关键难点在 Qwen Flash-Next 的 GDN recurrent/conv 与 PLE 状态**。普通 Transformer 的树形祖先 attention mask **不能直接代替递归状态的分支隔离**：从同一父节点伸出的两条分支，应拥有相同父状态的独立复制/分叉，然后各自推进；所有 rejected 分支应被丢弃。
- 先做 CPU toy recurrent-state 树执行器与 Serial oracle：`fork(parent_state)`、`advance(branch token)`、`commit(winning prefix)`、`discard(other branches)`，检查每个叶子的状态等价。再做单 GPU 小型 Target 多分支实验；在 GPU 批处理路径具备分支级隔离前，绝不宣称“一次 Forward 树验证已经实现”。
- 只在 Tree 验证的真实 GPU 成本曲线与预计接受收益都比单链好时升级；不能只因为 tree 更宽、平均接受长度增加就认定它更快。

**已有工作：** [DDTree](https://arxiv.org/abs/2604.12989) 已用 DFlash 扩散分布构建树；[CaDDTree](https://arxiv.org/abs/2606.01813) 已做树结构和预算的成本联合优化。我们的可能研究价值应在 **GDN/PLE/QSA 严格状态语义、SM121 真实树验证执行和专家成本** 的联合问题，且需查新和消融，不能声称发明树式推测解码。

### R3 — Ragged Multi-Request GPU Verification（多个 Agent 一次 Target Forward）

**大胆假设：**在多请求情况下，同一批 GPU 读取的 Dense 权重和部分共享专家权重可以服务不同请求的不同验证行数，使整机有效吞吐增加。

- 使用 E5 真正的 **DeepInfer C++ Scheduler / BlockManager** 作为起点，不重造同质 Scheduler；当前 E5 仅 CPU/GPU 交错模拟**逐请求** Forward，尚非真正 Batched Target Execution。
- 第一门槛：两个 target-only 请求，各 1 行，一次 GPU Forward，同时保持每请求独立 GDN/QSA/PLE/N-gram 有效状态、位置、EOS、取消/重用；数值与两条独立 Serial 请求逐行对照。
- 其后依序加入：相同 verification width → 不同宽度 `ragged` → padded CUDA Graph 桶（不能让无效行写污染）→ Prefill/Decode 混合；每一级独立验收。
- Batch 后的专家并集可能带来数据复用，也可能扩大访存与延迟；必须同时报告**总有效吞吐、每请求 TPOT p95、排队时间、峰值内存和公平性**。对只有一名用户的绝对 Decode TPS 不保证提升。
- 与 PR #2 的 Global Cost-Aware Scheduler RFC 形成依赖关系：**先验证真实 GPU Batch，再添加复杂预算器**。否则调度很聪明但 Kernel 还是逐请求执行。

**对 TensorFold 的定位：**当前 TensorFold 已有跨请求预算、多请求执行和分块 Prompt，不能把“支持 Batch”写成首创。价值在 Kaifra 的真实约束、混合模式和被验证的新执行方式。

### R4 — Grammar / Retrieval / Copy Draft Portfolio（低成本结构与复制草稿）

**大胆假设：**针对 Knotera 的结构化笔记、JSON、代码补丁和长篇资料引用，部分输出可直接从语法或当前授权上下文生成廉价草稿，不必每个候选都通过 BF16 DFlash Backbone。

- `GrammarDraft` 从输出约束产生候选；`NgramDraft` 从**当前请求有权访问**的 prompt/history 检索字符串或 token span；`DFlashDraft` 解决真正不确定的开放文本。
- 以 Request/Chunk 粒度选择 `provider`，清晰计入扫描索引、Tokenizer 对齐、热启动及路由成本；避免每 token 切换造成额外同步。
- **普通 unconstrained Decode** 下，复制文本只是草稿，必须被 Target 检验，不能因为它来自上下文就跳过 Target，也不能把旁路复制包装成“无损采样”。
- 只有**明确改变解码契约、语法约束唯一确定下一个 token**的模式，才可以探索通过一次批量状态推进（chunked-forward）而非每个 token 单独自回归输出；保证约束语义、GDN/QSA/PLE state 与正确 EOS 一致，不能外推到自由生成。
- 不应从其他用户或隐私上下文共享未经授权的 n-gram 索引。

**已有工作：** [llama.cpp speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) 已支持多种 n-gram 检索与 DFlash，原创性不在“提出 n-gram 草稿”。此前讨论的 [Copy-as-Decode](https://arxiv.org/abs/2604.18170) **已被作者撤稿**（arXiv 原页面明确标注 withdrawn）；不得将它当作已验证的性能先例。本研究独立验证结构化合约与成本。

### R5 — Tiny Acceptance & Cost Predictor（很小的控制器，而不是更大的 Draft）

**大胆假设：**比起再训练一个更大的 Drafter，用很小的模型预测接受率、单位 Verify 成本与模式切换价值，可能更划算。

- 输入仅限预算决策**之前**的合法信号：过往 conditional acceptance、DFlash selector logits/margin、当前请求长度、目标行数、已验证的内存与成本桶、上次可观测路由。
- 输出先只支持 `k∈{0,2,3,4,6,7}` 或原生支持的宽度集合；不发明未被 Kernel 证实的宽度/图桶。
- 必须对比：DeepInfer 原生 `MTPVarlen`、冻结 fixed-k、简单 EWMA/规则模型、强基线 TensorFold cost-aware depth。模型推理、特征提取、CPU↔GPU 传输成本均进入完整 step 时间。
- 用冻结的校准集训练/调参，新任务按 prompt 做 cluster bootstrap；不要把同一请求的不同位置随机拆成训练与测试，避免轨迹泄漏。
- 若小模型未能稳定超过简单规则，就直接采用规则，不为“AI 调度器”增加额外复杂性。

**已有工作：** DeepInfer/TensorFold/EVICT 均有成本和接受率自适应机制，不能仅凭 ML 预测器结构宣称创新。仓库 [PR #4](https://github.com/EdwinWang831/spark-infer-q/pull/4) 已规划“BF16 草稿数值差异是否伤害接受率”的诊断，先拿到该证据，再决定是否值得训练新的接受率控制器。

**状态更新（2026-10-22）：**接受率部分已在单请求上实测，没有等 PR #4 的 fidelity 诊断。[E14](E14_CONFIDENCE_BUDGET_REPORT.md) 的分桶 margin 规则没有确认的提速；[E15](E15_CONFIDENCE_TRAINING_REPORT.md)/[E16](E16_BILINGUAL_EXPANSION_REPORT.md) 训练了七个条件 logistic 头（F0 25 个特征、F1 137 个特征），按 prompt 和语义组拆分，在冻结的新任务上相对滞后的 MTPVarlen 预算 +10.00% / +9.61%，特征、传输和预测都计入了耗时。还缺：成本预测部分、模式切换价值，以及和简单 EWMA、TensorFold depth policy 的对照。[草稿 fidelity 方案](DRAFT_FIDELITY_ACCEPTANCE_PLAN.md)仍未执行；以后执行时不要同时换草稿和控制器。

### R6 — Incremental DFlash Context Projection Cache（增量上下文复用）

**大胆假设：**当 DFlash 每轮重算历史目标特征的 FC/context K/V 时，长上下文的重复投影可能持续增加成本；仅为新增且已提交的历史 token 计算投影并缓存，有机会降低 Proposer 成本。

- 当前 `spark_infer_q/dflash2_block.py` 的路径每次由已有 `features` 重新做 `fc/hidden norm`，并在五层草稿中投影历史 K/V；`torch.cat` 和历史特征拷贝也可能随 context 增长。
- 设计每请求自己的缓存版本 `(request generation, committed prefix length, feature revision, DFlash layer, key/value layout)`；只允许已提交前缀产生的特征加入持久缓存。拒绝候选或取消/槽位复用后必须丢弃无效尾部。
- 先测试等价性：长上下文的 cached/uncached hidden、Top-16、Selector 提案、完整目标输出/已提交 logits/GDN/QSA 状态；若 FP32/BF16 舍入差异改变提案，记录并做接受率+速度复核，不能宣称 bit-exact。
- 记录增加的 cache 字节、驻留/峰值与命中时间；短 prompt E9 中 Drafter 只有约 14%，不要想象缓存能让短请求整段 Decode 翻倍。
- 与 [H-Spec](https://arxiv.org/abs/2609.24197) 的草稿侧 KV 缓存研究对照，不能声称自己第一个发现草稿缓存开销。

**状态更新（2026-10-22）：**[E13](E13_DRAFT_CACHE_CONFIDENCE_REPORT.md) 实现了这个缓存（`CachedContextDraft`，opt-in）。1024/1536/1920-token 的三个输入上 decode 39.73→42.22 tok/s（+6.27%，含首轮构建和追加），四轮配对都为正，输出、committed logits 和状态检查通过；12 个短任务 −0.31%，所以默认不变。1024 容量的缓存约 25 MiB。4K 以上要先处理 DFlash 的 2048 滑窗和绝对位置。

### R7 — Spark Target + RTX 5070 Drafter 异构双设备实验

**高风险设想：**把成本昂贵的 Target Verification 保留在 GB10 Spark，另一台 RTX 5070 8GB 显存 + 64GB 主机内存的电脑运行低成本的 Drafter 或参考 Predictor。

- 决定是否值得尝试的条件不是另一台 GPU 有算力，而是 `远端 drafter 实际时间 + target feature 传输/拷贝 + 双机队列/同步` **严格小于** Spark 本地 drafter 的相同成本；任何模式转换/状态同步时间都计入。
- 每轮 DFlash 读取多层 Target Features，**天真的历史全量特征搬运会随着上下文增长**。必须先测增量 Context Cache、传输字节与链路实测带宽/RTT，而不是假定可以无成本地发送整段历史。
- 8GB 显存是否容纳完整草稿、代码/权重是否兼容异构 CUDA、远端失败怎样回退，都需要独立证明；更不能假定同请求 Draft 与 Target Verification 能无依赖地完全重叠。
- 优先考虑**跨不同请求的流水**，而非强行在同一轮交叉依赖下并发。
- 该方向目前属于 P3 探索，不作为 Kaifra MVP 或单台 Spark 速度目标的先决条件。

### R8 — Kernel-Aware + Portfolio Controller（真正的统一联合决策）

**终极愿景：**把 R1 的专家成本、R2 的树形结构、R3 的 Batch 以及 R4/R5 的多草稿模式放进同一个**可审计、可回退**的低频/逐 round 决策器。

在第 `t` 轮，只使用 `t` 时刻已经可观测的信号，估计：

```text
P(accepted_prefix | provider, proposed candidates, history)
  vs.
T(drafter + target_verify + commit + graph + cache + transfer | context, expert-cost estimate, batch)
  vs.
SLO(TTFT/TPOT p95), memory guard, fairness, fallback
```

算法可先从枚举少数合法模式/宽度开始，而不是直接训练复杂黑箱。**生成完整 DFlash 八行之后再缩小验证行数，不会返还已支付的 Drafter 费用**；做 pre-draft 与 post-draft 两阶段决策时必须区别已付与尚未付的成本。

这不是“我们发明 Cost-Aware Scheduler”。[TensorFold allocator](https://github.com/ashhart/TensorFold/blob/main/zig/src/core/lanes/allocate.zig)、[depth policy](https://github.com/ashhart/TensorFold/blob/main/zig/src/core/lanes/depth.zig)、EcoSpec、EVICT、CaDDTree 已有重叠先例。我们的研究问题是**在 GB10 + GDN 状态 + Q4 DFlash/原生 MTP + 多请求条件下，是否存在低开销且更优的联合决策与执行契约**。

## 3. 先后依赖：不要把多个改变堆在同一个 A/B

```text
E8/E9 固定执行器与正确性 oracle
   ├── #1 Expert Union / 专家占用基线 → R1 代价预测（E12：离线完成，在线暂缓）
   ├── R5 置信头预算（E15/E16：单请求在线 +10.00% / +9.61%）
   ├── #4 Draft BF16 fidelity / 真实接受率 → R2 假设（未执行）
   ├── E5 DeepInfer Scheduler 桥接 → R3 2-request GPU batch → Global portfolio
   ├── Q4 DFlash + long-context baseline → R6 incremental cache（E13：2K 以内完成）
   ├── R2 CPU toy branch-state → small GPU state fork → tree verify
   └── Grammar/n-gram 无训练原型 → full decode → 确定性约束 batch state
                      ↓
         R8 联合决策（仅融合已经单独验证有益的模块）
```

**建议单请求近期主线（2026-10-22 更新）：** R1 离线阶段已由 E12 完成，在线阶段暂缓。R5 按 E16 报告的下一步，先验证小头进入常驻实际执行入口后的收益，并用独立的外部任务来源检验泛化；R6 先解决 4K 以上的滑窗。R4 低成本草稿组合还没开始，仍是近期主线；R2 先做 CPU 状态分叉理论/正确性原型。  
**建议系统工程主线：** R3 双请求真实 GPU Batch。  
**长线高风险：** R2 tree、R7 双设备；R8 统一优化放在真实成本数据出来之后。

每条方案的具体仪表、Baselines、正确性门槛、止损条件与交付物见 [验证矩阵](KAIFRA_DECODE_VALIDATION_MATRIX.md)。

## 4. 研究成果、工程创新与原创算法要分开表述

1. **复现/集成**：我们已在 Spark 上运行 Q4 DFlash + DeepInfer 预算与 TensorFold-derived Target，性能收益必须如实引用 E6/E7/E8 以及 E13/E15/E16。
2. **独立工程能力**：由 DeepInfer-derived Scheduler 驱动我们的 GPU 执行与状态生命周期，经 API、批量状态/取消和长时正确性检验，可作为 Kaifra 衍生引擎的工程贡献。
3. **算法创新**：需要具有明确增量的新决策信息或正确执行算法、严格查新、消融和强基线（TensorFold/DeepInfer/SGLang/vLLM），而非重新命名 EcoSpec 或 CaDDTree。
4. **性能声称**：不同设备、输入、模型、版本、精度及输出轨迹的吞吐不能直接相加或排名。精确 greedy 回归属于同一路径的证明；随机 sampling 需要独立分布正确性（proposal/correction/RNG）验收。
5. **许可证与归属**：本仓库 Apache-2.0，TensorFold 0.6.0 后主许可证 Apache-2.0、早期及部分文件另有 MIT notices；DeepInfer、DFlash 及模型权重各自遵守原许可。研究阅读不等于可以在商用引擎中无条件重分发第三方二进制/模型权重。

## 5. 相关论文与实现：查新入口（截至 2026-10-21）

| 工作 | 已提出的能力 | 我们不能重复宣称的“创新” |
| --- | --- | --- |
| [EcoSpec, arXiv:2607.12696](https://arxiv.org/abs/2607.12696) | MoE 专家重用/边际验证成本的推测草稿选择 | “首次提出专家并集成本感知” |
| [EVICT, arXiv:2605.00342](https://arxiv.org/abs/2605.00342) | 训练自由的 MoE 草稿树自适应裁剪 | “首次研究裁剪低收益验证行” |
| [DDTree, arXiv:2604.12989](https://arxiv.org/abs/2604.12989) | DFlash 的 block diffusion 候选树 | “首次用 DFlash 构建验证树” |
| [CaDDTree, arXiv:2606.01813](https://arxiv.org/abs/2606.01813) | 成本感知地联合选择 diffusion 树和预算 | “首次以 throughput 选树预算” |
| [H-Spec, arXiv:2609.24197](https://arxiv.org/abs/2609.24197) | 减少 drafter-side KV cache 的混合目标上下文注入 | “首次尝试规避 Drafter 历史 KV 成本” |
| [llama.cpp/speculative](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) | n-gram、DFlash 和多种 speculative provider | “首次做低成本检索/复制草稿” |
| [TensorFold/lanes](https://github.com/ashhart/TensorFold/tree/main/zig/src/core/lanes) | 已有成本感知、多请求草稿深度及分配 | “首次做跨请求 Cost-Aware” |
| [Copy-as-Decode, arXiv:2604.18170](https://arxiv.org/abs/2604.18170) | 曾提出结构化复制 + parallel prefill | **该 arXiv 版本已撤稿**，不能按有效结果引用 |

相关结果主要是各自模型、硬件和论文测试条件下的预印本/实现，不能移植为 Spark 上的性能预测，也不能将未复现数字用于产品宣传。每项研究启动前还应再次查新。

## 6. 现在唯一可负责任的承诺

我们会继续优先验证“**同样带宽条件下，更少重复成本、更高有用输出**”这个研究命题；对任何失败结果也保留实验记录。**R5（E15/E16）和 R6（E13）已有受限的单请求实测；其余方向在新的 GPU benchmark 之前，加速百分比都是未知的。**

下一步从最便宜、最可证伪的实验开始；不要直接写一个宣称实现八种策略的大引擎。