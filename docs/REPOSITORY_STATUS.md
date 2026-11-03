# Repository map / 仓库分支与成果导航

2026-11-02 update: E38–E58 (prefill, 256K long context, KV compression and memory), previously only on `codex/prefill-expert-reuse`, are now on `main` as well. This map describes what is on `main` and what remains on an independent research branch. Experimental results retain their original workload and timing boundaries; merging a result does not promote its candidate to the default service. A themed index of every experiment is in [docs/README.md](README.md).

2026-11-02 更新：原先只在 `codex/prefill-expert-reuse` 上的 E38–E58（prefill、256K 长上下文、KV 压缩与内存）已汇总到 `main`。`main` 汇总已完成验证的代码与证据；独立研究分支继续保留。合入负面实验是保存结论，默认服务不因此启用其候选。全部实验的按主题索引见 [docs/README.md](README.md)。

| Location / 位置 | Contents / 内容 | State / 状态 |
| --- | --- | --- |
| `codex/engine-v01` | Kaifra v0.4.1, four 128K requests and optional compact features / 统一服务四请求 128K、紧凑特征与较快 prefill | [Usage / 用法](ENGINE_V05_USAGE.md), [report / 报告](ENGINE_V05_REPORT.md), [PR #23](https://github.com/EdwinWang831/spark-infer-q/pull/23) · Matched complete wall −18.67%, serial long GPU path / 默认镜像／512 行，显式启用单缓冲／1024 行 |
| `codex/engine-v01` | [128K profiling and QSA screen / 长 prefill 剖析与 QSA 候选](ENGINE_V06_REPORT.md) | Complete wall −1.99%, below 3% gate, research only / 未晋升，线上 v0.4.1 已恢复 |
| `main` | E0–E58 research records, including E26 cancellation; E33 exact screened head, E34/E35 rejected candidates, E36 exact prefill and E37 profile / 已完成研究与脱敏证据 | Consolidated; per-experiment limits still apply / 已汇总，沿用各实验边界 |
| `main` | K1 target-only runtime → K2 dynamic DFlash → K3 Graph/E33 profile → single-request overhead reduction / 原生调度运行时演进 | Explicit opt-in APIs/profiles / 显式启用 |
| `main` | Joint Target, two requests sharing one target forward / 双请求联合 Target | Target-only API; distinct from K4 DFlash service / 独立于 K4 服务研究 |
| `main` | E38 paired expert reuse, E39 CTA loading and ownership recovery / 专家复用、CTA加载及观测回调收尾 | Both component cost gates failed; independent research records / 两候选成本门槛失败，独立研究记录 |
| `main` | E40 owned prefill executor with TensorFold operators / 自主 prompt 执行模块及算子接口 | Explicit opt-in target-only / dynamic DFlash integration; see [usage](E40_PREFILL_EXECUTOR_USAGE.md) / 显式接入，外部算子依赖仍保留 |
| `main` | [E41](E41_PREFILL_GRAPH_REPORT.md) optimized Graph/E33/W8 prefill handoff / 完整优化链路验证 | Exact 1,536-token input, partial commits, cancel/reuse and HTTP smoke pass; prefill gain, no new decode gain / prefill提速，decode无新增收益 |
| `main` | [E42](E42_WIDER_PREFILL_REPORT.md) explicit1024-row prefill / 更大分块 | Long-prefill wall −5.80%, exact checks pass; default512 retained / 长输入收益，默认保留512 |
| `main` | [E43](E43_PREFILL_PIPELINE_REPORT.md) instruction-only screen / 指令流水筛选 | No gain; rejected, early32MiB allocation increase remains recorded / 不采用并保留审计异常 |
| `main` | [E44](E44_MIA_PREFILL_TILE_REPORT.md) Mia shared-input expert tile / 专家tile适配与公开接口 | Opt-in; long-prefill wall −6.97%, full-model/public-hook and current-baseline audit pass / 显式启用，当前基线审计通过 |
| `main` | [E45](E45_LONG_CONTEXT_REPORT.md) bounded draft history /4K QSA and Graph / 有界草稿历史与4K | Opt-in;99.61MiB feature ring, no speed gain; no1M/paged KV/multi-request DFlash / 精确与隔离门槛通过 |
| `main` | [E46](E46_256K_REPORT.md) direct256K / 单请求256K实跑 | 262080 input tokens,230.4s, retrieval3/3, exact state/Graph;81.41GiB peak / 未测双请求256K |
| `main` | [E47](E47_KV_DUAL_REPORT.md) compressed KV + two256K states / 双256K容量 | 4.13GiB/request cache,82.63GiB peak; original precision gate failed,225.8s output gap / 非服务晋升 |
| `main` | [E48](E48_PREFILL_FAIRNESS_REPORT.md) two256K prefill/decode fairness / 公平调度 | A gap225.42→0.842s, completion453.03→241.60s; full state exact, total wall unchanged / 显式target-only，INT8精度门槛仍未过 |
| `main` | [E49](E49_LONG_DYNAMIC_REPORT.md) long-output adaptive DFlash / 动态长输出与四状态容量 | BF16 single-request decode16.36→30.81 tok/s; full state exact; four256K capacity stop90.14GiB >88GiB / 未运行四路长请求 |
| `main` | [E50](E50_OWN_OPTIMIZATION_REPORT.md) own long execution and eight-state accounting / 自主链路与八路预算 | BF16 native decode32.14→34.49 tok/s, exact full state; FP8 eight-state106.77GiB forecast / 显式启用；有界原始索引仅CPU模型，尚无八路GPU结果 |
| `main` | [E51](E51_BOUNDED_INDEX_REPORT.md) bounded raw QSA index / 有界原始索引 | Saves0.744GiB/request, decode flat; opt-in / 显式启用 |
| `main` | [E52](E52_DECODE_PIPELINE_REPORT.md) decode expert pipeline / decode专家流水 | Both cost intervals cross zero; rejected / 不采用 |
| `main` | [E53](E53_LONG_CONCURRENCY_REPORT.md)/[E54](E54_REQUEST_DRAFT_CACHE_REPORT.md) two concurrent256K adaptive requests / 双256K并发 | Outputs/state match isolated runs; E54 restores strict parity; eight-slot forecast104.06GiB >88GiB / 未运行八路 |
| `main` | [E55](E55_RESIDENCY_REPORT.md)–[E58](E58_NATIVE_CONVERSION_REPORT.md) residency inventory and INT4/FP8 KV / 常驻盘点与KV压缩 | INT4 eight-way forecast92.82GiB >88GiB; FP8/INT4 fail the3% logit gate; defaults unchanged / 未采用 |
| `codex/prefill-expert-reuse` | Source branch of E38–E58 / E38–E58 的来源分支 | All its commits are on `main`; its owner decides whether to keep it for further work / 提交均已在 main，是否继续使用由负责人决定 |
| [K4 research, PR #19](https://github.com/EdwinWang831/spark-infer-q/pull/19) | `codex/k4-batched-runtime`: joint verification, async SSE and dispatch wait fix / 双请求动态验证、异步服务及等待修正 | Independent draft; existing worktree retained / 保留现有工作区 |
| [Self-evolution design, PR #15](https://github.com/EdwinWang831/spark-infer-q/pull/15) | `claude/magical-fermi-tc55jr`; local worktree branch `codex/pr15-e33-rsi-plan` / 自演化设计 | Design under review, not an implemented speed result / 设计评审中 |
| `archive/pr6-confidence-proposal` tag | Original closed PR #6 branch tip / 已关闭预测器提案的原分支端点 | History retained; later E15–E25 results live on main / 归档保留，不将旧提案混入当前实现 |

K4's remote base `codex/k4-runtime-overhead` is temporarily retained because PR #19 still targets it. Its commits are also on `main`; it is a dependency reference, not a second active workstream. The K4 owner can update to main and remove this base when ready, without this cleanup changing that worktree or its unmerged code.

K4 仍以远端 `codex/k4-runtime-overhead` 为 PR 基线，因此暂留这个依赖引用；该基线成果已在 main。它不是另一条开发路线。本次不改写 K4 或设计工作区，不重放 GPU 实验，也不删除模型、原始实验归档和未提交文件。

## Navigation / 导航

- Runtime / 运行时：[K1](K1_RUNTIME_USAGE.md)、[K2](K2_DFLASH_SCHEDULER_USAGE.md)、[Graph](K3_GRAPH_RUNTIME_USAGE.md)、[E33 profile](K3_E33_RUNTIME_USAGE.md)、[overhead report](K4_RUNTIME_OVERHEAD.md)。
- Target batching / 联合执行：[Joint Target](JOINT_TARGET_EXECUTION_REPORT.md)。
- Prefill / 提示处理：[E36](E36_PREFILL_NUMERICS_REPORT.md)、[E37](E37_PREFILL_PROFILE_REPORT.md)、[E40 executor](E40_PREFILL_EXECUTOR_USAGE.md)、[E41 Graph validation](E41_PREFILL_GRAPH_REPORT.md)、[E42 chunk1024](E42_WIDER_PREFILL_REPORT.md)、[E44 expert tile](E44_MIA_PREFILL_TILE_REPORT.md)、[E45 bounded history /4K](E45_LONG_CONTEXT_REPORT.md)、[E46 direct256K](E46_256K_REPORT.md)、[E47 compressed KV / two256K](E47_KV_DUAL_REPORT.md)、[E48 fairness](E48_PREFILL_FAIRNESS_REPORT.md)、[E49 adaptive long output](E49_LONG_DYNAMIC_REPORT.md)、[E50 own optimization](E50_OWN_OPTIMIZATION_REPORT.md)、[E51 bounded raw index](E51_BOUNDED_INDEX_REPORT.md)、[E52 decode expert screen](E52_DECODE_PIPELINE_REPORT.md)、[E53 long concurrency / rejected strict policy gate](E53_LONG_CONCURRENCY_REPORT.md)、[E54 private draft cache / strict dual adaptive parity](E54_REQUEST_DRAFT_CACHE_REPORT.md)。
- Long context and memory / 长上下文与内存：[E55 residency / INT4 KV](E55_RESIDENCY_REPORT.md)、[E56 INT4 precision](E56_KV_PRECISION_REPORT.md)、[E57 affine KV](E57_AFFINE_DEFERRED_KV_REPORT.md)、[E58 native conversion](E58_NATIVE_CONVERSION_REPORT.md)。
- Rejected candidates / 已否决候选：[E34](E34_DEVICE_SCREEN_REPORT.md)、[E35](E35_MOE_OVERLAP_REPORT.md)。
- Full experiment index / 完整实验索引：[themed index / 按主题索引](README.md)、[chronological log / 时间顺序记录](EXPERIMENTS.md)。

For new work, branch from current main, keep one branch per active workstream, and remove a completed branch only after its commits are reachable from main or an explicit archive tag. Historical plans, source hashes and result files stay intact.

后续新工作从最新 main 开始，每条在做的路线保留一个分支。已完成分支在确认 main 或明确归档标签可达后清理；历史方案、源指纹和结果文件保持不变。
