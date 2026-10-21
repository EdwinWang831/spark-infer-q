# Kaifra 实施路线图：从 SparkInfer-Q 实验原型到自主推理引擎

> **状态：设计提议，不包含任何实际实现或新的 GPU 运行结果。**  
> **基础路线：基于 DeepInfer 原生 C++ Scheduler / MTPVarlen 进行 Qwen 与 Spark 深度适配，不从零实现新的 Scheduler；TensorFold 组件作为可替换执行后端，DFlash2 作为草稿提供者。**  
> 此文档与 [Kaifra Vision RFC](KAIFRA_VISION_RFC.md) 同 PR，落实“怎样才算自己的引擎”。  
> 先保留 `spark-infer-q` 仓库与包名，不启用自动迁移或重命名。

## 1. 工程目标与不做什么

### 成功的 Kaifra MVP（最低完整定义）

一台 DGX Spark 上，**在 DeepInfer C++ Scheduler / MTPVarlen 底座上由我们维护和扩展** Qwen 请求状态、任务队列、GPU Batch、推测验证与执行契约，并驱动同一份 Qwen Target（可复用外部获许可 CUDA kernel），提供基本服务 API：

- 单模型加载、显式 checkpoint/量化/后端版本；
- Text-only greedy streaming、EOS/max_tokens、取消、限额/拒绝；
- 分块 Prefill、target-only decode、正确缓存和 GDN/QSA 状态管理；
- 可选 Q4 DFlash2 DraftProvider，拒绝后保证 Target 参考输出/状态正确；
- 完整 decode/prefill 与端到端 SSE/HTTP 指标，错误安全回退；
- 可以不启动 TensorFold 原来的 serving 服务，也不必运行 DeepInfer 原本针对 DeepSeek 的完整 LLMEngine/Worker 进程；**仍然复用并扩展 DeepInfer C++ Scheduler、BlockManager（逻辑配额）和 MTPVarlen**，Kaifra 自己适配 Qwen GPU KV/GDN/PLE 状态。

**不要求**自研所有 CUDA kernel，不要求从第一个版本支持全模型、vision、随机采样、长上下文或多用户，不要求一开始超越 TensorFold。

**仅包装 TensorFold 现成服务 API ≠ 自己的执行引擎。** 最简单的可验收区别：**DeepInfer-derived** 调度计划通过 Kaifra 自己的 Qwen 适配器驱动 device state 的生命周期、speculative verify/commit 和 API；GPU kernel 只是可替换的执行 provider。我们不主张 DeepInfer Scheduler 是自行发明。

## 2. 理想结构（先契约、后优化）

~~~mermaid
flowchart TD
    API["Kaifra API / stream"] --> R["Request Runtime"]
    R --> DS["DeepInfer C++ Scheduler + BlockManager"]
    DS --> AD["Kaifra Qwen Adapter / Admission + Cancellation"]
    AD --> CP["DeepInfer MTPVarlen + Kaifra Cost / SLO Policy"]
    CP --> BP["Batch & Phase Planner"]
    BP --> EX["Kaifra Executor"]
    EX --> KP["Kernel Adapter: TensorFold-derived, later alternatives"]
    EX --> DP["Draft Providers: none / native MTP / Q4 DFlash2"]
    KP --> ST["Qwen GDN + QSA state manager"]
    DP --> ST
    ST --> OUT["Verified output + metrics"]
    OUT --> CP
~~~

只有三条必须先清晰隔离的接口：

1. **DeepInfer-based RequestRuntime**：以原生 C++ Scheduler 的 `schedule/process_outputs` 和逻辑 BlockManager 为起点，扩展 Qwen 请求身份、等待/运行/完成/取消、输出与故障隔离；不重造同质 Scheduler，也不在其中直接操作 GPU 权重；
2. **ExecutorBackend**：以批次/图 plan 调用 Prefill、Decode/Verify、Snapshot、Commit、Restore、Release；外部 kernel/weight loader 可注入；
3. **DraftProvider**：根据 Target 已提交特征和 pending token 给出 proposal，表明 block 大小、量化、token map、候选分布和状态开销；始终由 Target 进行合法验证。

### 可写入代码的接口草案（*并非已实现*）

~~~python
class ExecutorBackend(Protocol):
    capabilities: Capabilities  # verified flags, not optimistic promises
    def prefill(self, requests, chunks) -> PrefillResult: ...
    def verify(self, requests, candidate_tokens, lengths) -> VerifyResult: ...
    def commit(self, request, verified, keep_rows) -> None: ...
    def snapshot(self, request) -> StateHandle: ...
    def restore(self, request, handle) -> None: ...
    def release(self, request) -> None: ...

class DraftProvider(Protocol):
    provider_id: str
    max_drafts: int
    def preflight(self, target: ExecutorBackend) -> Capabilities: ...
    def propose(self, request, capture, anchor, limit) -> Proposal: ...

class KaifraEngine(Protocol):
    def submit(self, prompt, settings) -> RequestHandle: ...
    def step(self) -> list[OutputEvent]: ...
    def cancel(self, request) -> None: ...
~~~

接口参数在实现 PR 中依据真实 Qwen 状态结构细化。**不要直接把 DeepSeek V4.1 的 cache layout、TensorFold 内部对象或 PyTorch Tensor 生命周期写死在公共协议里。**

### 正确性不变量（所有优化都必须满足）

- 已向客户端输出的 token、经过 Target forward 的 token、Target 已提交的 position 是不同概念；
- correction/bonus 可以是已输出但尚未处理的 pending token，不能提前推进 GDN/QSA position；
- 拒绝 token 留在临时计算区，不能污染 GDN recurrent/conv、QSA KV/index、PLE 或 proposer context；
- `commit(keep=0)` 不能盲目调用底层不支持的 API；零保留应丢弃临时状态；
- request_id、slot、generation、dispatch epoch 对齐；cancel 后旧 GPU 完成事件无权覆盖复用 slot；
- CUDA Graph 捕获和 padded rows 不得写入其他请求有效状态；
- 随机采样仅在 proposal q、accept/correction、RNG 和分布等价单独验证通过后才能宣布支持；
- 任何预测控制器不能访问未来接受结果，hindsight/oracle 只能在离线报告中出现。

## 3. 和已经提交的 E5/E6/E7/E8 工作衔接

| 现有证据 | 可复用资源 | 当前尚缺 |
| --- | --- | --- |
| E5 Scheduler Bridge | 原生 C++ schedule/process 生命周期、两请求取消/复用检查 | 真正的 batched GPU forward、Worker/异步与 DFlash 接线 |
| E6 Draft Head | 可注入 Q4Backend、full vocab ID、同路径 state oracle | 正式引擎中的 Q4 proposer 生命周期与多请求常驻 |
| E7 Native Match | 充分逐任务预热的同机对照脚本和 Q4 DFlash2 与 MTP 差距 | 与同一执行器 MTP 对照、可比长输出性能 |
| E8 Prompt-only | 2/4/8 token 块与 serial 的 logits/features/state 检查；8 行 prompt 速度已单测 | Q4 完整 SSE 集成/独立多次测量及 CUDA Graph |
| 先前 PR #2 | Global Cost-Aware Scheduler 的控制面提案 | GPU 批处理、成本模型和运行时接入 |

**重要依赖顺序：**当前 `main` 还没有合并 E5–E8 的开发代码。落实时应先把相关开发分支整理成审查清楚、依赖明确的 PR；**不要在旧 main 上重新发明 E5/E6 的实现或把原始大结果文件随意复制进 vision PR。**

## 4. 分阶段实施，严格的 Done 定义

### K0 — 整理证据与锁定可重复的基线

**交付：**冻结相同 SM121、checkpoint、模型/草稿权重、Tokenizer、E7 客户端、原生/候选 binary hash、预热计划与输出差异。保留原生 MTP 和 Q4 DFlash2 的对照。

**验收：**历史结果可追溯，回归测试能识别数值与输出变化；训练/测试切分不混淆。优先落实 E8 完整生成（包括 TTFT/完整请求），其 prompt-only 4.66× **不替代**端到端性能结果。

### K1 — 基于 DeepInfer 的 Target-only Runtime

**交付：**沿用 E5 已证明可用的原生 DeepInfer C++ Scheduler / BlockManager 逻辑，构建由 Kaifra 维护的 `KaifraEngine` Qwen 请求状态机/Target KernelAdapter、bounded chunked Prefill、serial Decode、stream、EOS/max_tokens、cancel、权重加载与清理。

**验收：**同路径目标 Serial 逐 row logits、有效 GDN/QSA/PLE state 连贯；输入长度上限、非空响应、请求取消/复用、错误和资源回收正确。**运行时继续使用原生 DeepInfer C++ Scheduler，但不依赖 TensorFold 的 HTTP server 或原版 DeepSeek 专用 DeepInfer Worker 常驻。**

### K2 — DFlash2/Q4 接入并测量执行路径

**交付：**`DraftProvider=DFlash2Q4`，完整词表投影、selector 和提案前缀；target-only/fixed K/dynamic K 可切换；每个 provider 明确 checkpoint/precision/block contract。

**验收：**至少 K=0/1/3/7、全接受、首/中/末拒绝、EOS、max_tokens，在每次 reject 后的下一轮继续生成可对照；随机采样仍明确 unsupported。报告 per-stage draft/target/commit、动态预算开销和完整请求速度。与单模式策略/普通 Decode 比较。

### K3 — CUDA Graph 与逐 token 同步去除

**交付：**静态工作区、可验证的 CUDA Graph bucket、减少 `.tolist()` / `.cpu()` / host fence 的执行路径；适配器公开实际的 capture 支持状态。

**验收：**每个允许的 bucket 的 logits/状态/输出回归；被禁止的形状能可靠走 eager fallback；明确区分首次图 capture 和稳定 replay、批次变化。避免在未覆盖的 GDN/QSA 稀疏分支上强启 Graph。

### K4 — 真正双请求 GPU 联合验证（通向全局调度的硬门槛）

**交付：**两个请求的一次 Target batch forward，正确持有各自 GDN/QSA/PLE 和拒绝后 rollback；先无草稿，再相同 K，最后不同 K 的 ragged/合法 bucket。

**验收：**同请求 solo vs mixed token + logits + 有效状态对照；EOS / cancelled / slot reuse / graph padding / 低高接受混合无污染；报告总吞吐与 per-request TPOT p50/p95。仅两个请求交错但逐个 GPU forward **不算完成**。

### K5 — 成本感知的跨请求调度

**交付：**测过的 batch bucket 成本、按请求历史接受率、内存准入与公平性限制；比较固定 K、TensorFold 已有成本策略、DeepInfer `MTPVarlen`、Kaifra 候选。

**验收：**新冻结 2/4/5 请求混合负载中至少存在可复现的完整任务/吞吐优势，不得显著恶化交互 TTFT/TPOT p95 或状态正确性。否则保留简单策略为默认，复杂策略撤到研究模式。

### K6 — Speculative Mode Portfolio + Spark Resource Policy

**交付：**Native MTP / DFlash2 Q4 / no-draft 的受控低频选择；对切换时延、常驻、warmup、重建 state、图缓存收费；内存水位和前台/后台 SLO 的组合优化。

**验收：**不以仅比较错误的单路 fixed3 作为胜利证据，而要超过独立校准选定的最佳单模式；每个模式正确性、低收益 fallback 和 workload 分布偏斜须披露。开关成本过高则退回按请求选模式。

### K7 — 长时稳定性、API/发布与研究评估

**交付：**可安装/复现的服务，合法许可证通知，版本化 benchmark harness、真实诊断报告、长上下文/多请求/压力/恢复测试。

**验收：**不夸大模型质量和广泛兼容性；目标为一个负载域可明确优于最新 TensorFold/SGLang/vLLM，而不是声称全部负载无条件领先。给出可靠性、成本与回归趋势。

## 5. 对比 TensorFold 的“公平竞争规则”

TensorFold 的**当前**代码已经有：
- `zig/src/core/lanes/allocate.zig`：跨请求概率和成本预算；
- `zig/src/core/lanes/depth.zig`：按接受率和完整 round 时间调草稿深度；
- `zig/src/families/flashnext/cuda_lanes.zig`：多请求 CUDA 执行和分块 Prefill；
- `zig/src/families/flashnext/cuda_round_graph.zig`：有限 decode graph 的 capture/replay 支持（不是所有路径保证）；
- 性能/内存基线会随版本发生变化，E7 原生固定配方不等于最新 upstream。

因此比较必须标记相同硬件、checkpoint revision、实际内核来源/二进制 SHA、KV/草稿精度、图开关、并发数、上下文、缓存/预热协议和完整请求输出。优先使用**同一 Target 计算与相同输出**的微基准隔离优化，再用真实服务完整请求衡量最终体验。

主指标：

| 指标 | 重要性 |
| --- | --- |
| 全任务有效输出吞吐和完整请求耗时 | 主成功标准 |
| TTFT / TPOT p50、p95（按请求，不是跨任务耗时中位数之和） | 交互质量与公平性 |
| 同等输出长度与不同输出轨迹的独立报告 | 防止工作量变化假加速 |
| verified token、有效 GDN/QSA state、EOS、取消 | 不能以正确性换性能 |
| 模型/草稿/缓存/graph resident bytes 与峰值、OOM | 统一内存成本 |
| draft/target/commit/context/profiling/sync 时间 | 成本归因 |

必须保留“实验不支持假设”的负结果；如果预算控制器与校准固定长度持平，应优先发布简单实现。提出的 15%–30% 多请求目标只能当 **stretch target**，不是预计会拿到的加速值。

## 6. 建议的前五张工程 PR（本文件只是规划，不包含实现）

| 次序 | PR 目标 | 预期改动 |
| --- | --- | --- |
| PR-K1 | `kaifra/runtime` DeepInfer-derived 请求状态与状态契约 | 在 E5 C++ Scheduler 桥接上构建 API/生命周期、单测/回归与 no-kernel fake backend，保留上游归属 |
| PR-K2 | `kaifra/executor` Target-only 路径 | Qwen kernel adapter + Prefill/Decode、有效 state 对照 |
| PR-K3 | `kaifra/spec` Q4 DFlash2 | provider 接口、K 档位、拒绝正确性 + E7/E8 复现 |
| PR-K4 | `kaifra/graphs` & batched verify | 静态 GPU buffers、Graph 尺寸、双请求 state 隔离 |
| PR-K5 | `kaifra/scheduler` | 真实成本表、跨请求预算、公平性/降级、对照报告 |

这里只是将来建议的**模块名称/PR 名称**，并非实际已经创建的路径、提交或 PR。初期应尽量把每个 PR 限制在可独立验证的一层，不要将仓库一次性大改名。

## 7. 原创性与停止条件

- **工程创新**：在明确承认 DeepInfer-based 架构和上游版权的前提下，当我们维护完整的 Qwen runtime、state lifecycle、batch + correct verification，并有真实用户负载优势时，能够有根据地称为独立维护的衍生推理引擎；不必自研全部 kernels 或重新发明 Scheduler。
- **算法创新**：仅在新联合决策机制、数值/状态约束或缓存算法被清晰形式化、查新、消融并超过最新强基线之后主张；成本感知/动态草稿不是首次提出。
- **中止/缩减条件**：GPU batch/状态长期无法正确支持、优化收益不能超过变更噪声、维护成本明显高于可用收益时保留 TensorFold serving + SparkInfer-Q 可验证研究模块，不为“自研”牺牲可靠性。
- **名字**：Kaifra 是工作名。先不改仓库、产品发布、公司或域名，完成目标市场和商标检索再决定。

**最先值得完成的一步：**保留并整理 E8 分块路径的正确性与真实端到端实验，然后**继续扩展现有 E5 DeepInfer C++ Scheduler 桥接**，使其驱动 Kaifra 自己的 Qwen Target-only 执行/状态服务，逐步替换测试 harness 中的单请求生成循环。