# Repository map / 仓库分支与成果导航

2026-10-24 consolidation. This map describes what is on `main` and what remains on an independent research branch. Experimental results retain their original workload and timing boundaries; merging a result does not promote its candidate to the default service.

2026-10-24 整理：`main` 汇总已完成验证的代码与证据；独立研究分支继续保留。合入负面实验是保存结论，默认服务不因此启用其候选。

| Location / 位置 | Contents / 内容 | State / 状态 |
| --- | --- | --- |
| `main` | E0–E37 research records, including E26 cancellation; E33 exact screened head, E34/E35 rejected candidates, E36 exact prefill and E37 profile / 已完成研究与脱敏证据 | Consolidated; per-experiment limits still apply / 已汇总，沿用各实验边界 |
| `main` | K1 target-only runtime → K2 dynamic DFlash → K3 Graph/E33 profile → single-request overhead reduction / 原生调度运行时演进 | Explicit opt-in APIs/profiles / 显式启用 |
| `main` | Joint Target, two requests sharing one target forward / 双请求联合 Target | Target-only API; distinct from K4 DFlash service / 独立于 K4 服务研究 |
| `codex/prefill-expert-reuse` | E38 paired expert reuse, E39 CTA loading and ownership recovery / 专家复用、CTA加载及观测回调收尾 | Both component cost gates failed; independent research records / 两候选成本门槛失败，独立研究记录 |
| `codex/prefill-expert-reuse` | E40 owned prefill executor with TensorFold operators / 自主 prompt 执行模块及算子接口 | Explicit opt-in target-only / dynamic DFlash integration; see [usage](E40_PREFILL_EXECUTOR_USAGE.md) / 显式接入，外部算子依赖仍保留 |
| `codex/prefill-expert-reuse` | [E41](E41_PREFILL_GRAPH_REPORT.md) optimized Graph/E33/W8 prefill handoff / 完整优化链路验证 | Exact 1,536-token input, partial commits, cancel/reuse and HTTP smoke pass; prefill gain, no new decode gain / prefill提速，decode无新增收益 |
| [K4 research, PR #19](https://github.com/EdwinWang831/spark-infer-q/pull/19) | `codex/k4-batched-runtime`: joint verification, async SSE and dispatch wait fix / 双请求动态验证、异步服务及等待修正 | Independent draft; existing worktree retained / 保留现有工作区 |
| [Self-evolution design, PR #15](https://github.com/EdwinWang831/spark-infer-q/pull/15) | `claude/magical-fermi-tc55jr`; local worktree branch `codex/pr15-e33-rsi-plan` / 自演化设计 | Design under review, not an implemented speed result / 设计评审中 |
| `archive/pr6-confidence-proposal` tag | Original closed PR #6 branch tip / 已关闭预测器提案的原分支端点 | History retained; later E15–E25 results live on main / 归档保留，不将旧提案混入当前实现 |

K4's remote base `codex/k4-runtime-overhead` is temporarily retained because PR #19 still targets it. Its commits are also on `main`; it is a dependency reference, not a second active workstream. The K4 owner can update to main and remove this base when ready, without this cleanup changing that worktree or its unmerged code.

K4 仍以远端 `codex/k4-runtime-overhead` 为 PR 基线，因此暂留这个依赖引用；该基线成果已在 main。它不是另一条开发路线。本次不改写 K4 或设计工作区，不重放 GPU 实验，也不删除模型、原始实验归档和未提交文件。

## Navigation / 导航

- Runtime / 运行时：[K1](K1_RUNTIME_USAGE.md)、[K2](K2_DFLASH_SCHEDULER_USAGE.md)、[Graph](K3_GRAPH_RUNTIME_USAGE.md)、[E33 profile](K3_E33_RUNTIME_USAGE.md)、[overhead report](K4_RUNTIME_OVERHEAD.md)。
- Target batching / 联合执行：[Joint Target](JOINT_TARGET_EXECUTION_REPORT.md)。
- Prefill / 提示处理：[E36](E36_PREFILL_NUMERICS_REPORT.md)、[E37](E37_PREFILL_PROFILE_REPORT.md)、[E40 executor](E40_PREFILL_EXECUTOR_USAGE.md)、[E41 Graph validation](E41_PREFILL_GRAPH_REPORT.md)。
- Rejected candidates / 已否决候选：[E34](E34_DEVICE_SCREEN_REPORT.md)、[E35](E35_MOE_OVERLAP_REPORT.md)。
- Full experiment index / 完整实验索引：[EXPERIMENTS](EXPERIMENTS.md)。

For new work, branch from current main, keep one branch per active workstream, and remove a completed branch only after its commits are reachable from main or an explicit archive tag. Historical plans, source hashes and result files stay intact.

后续新工作从最新 main 开始，每条在做的路线保留一个分支。已完成分支在确认 main 或明确归档标签可达后清理；历史方案、源指纹和结果文件保持不变。
