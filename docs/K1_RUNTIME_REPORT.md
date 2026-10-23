# K1：原生 Scheduler 驱动的 Target-only 运行时

2026-10-23。E5 的真实 C++ Scheduler 桥接已扩展为可复用的 `KaifraEngine`，包含 Token/文本准入、分块 prompt、逐步已提交输出、EOS/长度结束、取消与 slot/generation、请求状态回收和有界历史。[事前方案](K1_RUNTIME_PLAN.md)、[用法](K1_RUNTIME_USAGE.md)。

请求计划及逻辑块仍由 DeepInfer 原版 Scheduler / BlockManager 决定；TensorFold Target 负责实际 forward 与 GDN/QSA/PLE 状态。新增运行时没有启动 TensorFold HTTP 服务，也没有复制一套新的调度算法。

## 已完成的真实 GPU 验收

使用已有常驻 session，同一 Target / DFlash 对象，加载计数1。固定 DeepInfer revision 与核心扩展 SHA 与 E5 相同。capacity256、最多两请求、token_budget32、BF16 KV、greedy；三个受控英文算术/JSON/中文提示，输出上限1～48。没有重新加载权重。

| 场景 | 结果 |
| --- | --- |
| 单请求 EOS，prefill_rows=1 | 通过 |
| 单请求 EOS，prefill_rows=8 | 通过 |
| 中文与 JSON 双请求交错 | 通过，GPU 仍逐请求执行 |
| max_tokens=1 | 只提交 prompt 状态，输出 token 保持 pending |
| 等待时取消、同 ID/slot 复用 | 取消没有新增输出，generation 递增 |
| 首次输出后取消、同 ID/slot 复用 | 取消不再追加 token，旧句柄被拒绝 |
| 非法 token/长度与满槽拒绝 | 拒绝正确，取消后回收 |

共核对 **233 行完整 logits、91 次有效状态**，精确匹配同路径逐 token reference。每组结束都无活跃 device state、无历史 completed/event 累积、原生调度队列为空，36/36 可用逻辑块全部归还。所有非取消输出与 reference 前缀一致；终止设备 position 等于 prompt 长度加输出长度减1。没有把 pending 的最后输出 token 提前计入设备状态。

原生上下文重复打开/关闭及嵌套借用 Gloo 进程组检查通过；借用者关闭不会销毁拥有者的 group，显式修改的环境变量恢复。常驻 Target/DFlash 对象和权重保持原样。

## HTTP 与本地验证

`kaifra_serve.py` 提供单模型、loopback、一次一请求的 SSE 入口，以及健康检查和显式取消。SSE 输出包含新增 token IDs、累计文本快照、终止原因与准入后的墙钟指标；不是完整 OpenAI 协议。断线回收、显式取消、繁忙拒绝、输入拒绝、Unicode 快照和隔离状态已有本地 HTTP 测试。

真实模型 HTTP/SSE 检查另行排队执行；报告在该检查完成后补充最终结果。独立 CLI 加载流程沿用原 loader，本轮不为重复验证启动另一个大模型进程。

基于当前 main 的本地完整测试：200项，其中12项因依赖缺失跳过，其余通过。回到 main 基底后，GPU 封存的所有源码 SHA 逐一复核一致。

## 失败与修正

保留两次失败，不并入通过计数：

1. v1 的原生上下文关闭恢复了原先未设置的 NEXTN 环境；重新打开时把已缓存为0的兼容扩展误判不兼容。修复判断后，重复打开和嵌套借用检查通过。该失败在实际调度前发生。
2. v2 的单请求输出/状态检查通过，但验收发现原生 running 队列仍含 terminal 项。运行时在最后请求完成后追加有界 drain 清除原生收尾账本，再报告 idle。修复版七组完整重跑通过。

另一次 HTTP 检查的提交预检发现 worker 正在执行其他实验，未派发 GPU 任务；随后按正常队列排在该实验之后，没有中断或覆盖它。

## 结论边界与下一步

这是 K1 的 Target-only 运行时进展。DFlash2、Q4 草稿、MTPVarlen 预算与该原生 Scheduler 运行时仍未统一；下一步按 K2 接入已有组件，验证拒绝后继续生成及状态一致性。跨请求 GPU batch、整模 Graph、异步流水、GPU 页映射和长上下文仍待实现。

这里的 reference 与候选使用同一外部 Target 算术；通过证明接线和状态连续性，不是独立整模数值正确性证明。三个短提示不代表一般用户负载，本轮不宣称任何性能提升。已有 prefill 数值与 drafter fidelity 问题继续保留。
