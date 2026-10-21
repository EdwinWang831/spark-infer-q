# E7：完整词表 Q4 DFlash2 与原生 TensorFold 同机对照

日期：2026-10-21（America/Los_Angeles）。本轮使用[预先冻结的方法](E7_TENSORFOLD_MATCH_PLAN.md)，主对照为本机已核验的 MiaAI-Lab 配方安装态原生 MTP 与我们的完整词表 Q4 DFlash2＋DeepInfer 原生动态预算。四个加载阶段全部完成，200 次正式计时、100 次完整预热（其中含40次候选非串行 oracle）通过归档校验。

## 版本与计时口径

- 原生 ELF SHA256：`cb2eb4e21e2031cad267ad646c80170df894f76b426ebddcba0895628682185d`；320 项 SM121 AOT 清单 SHA256：`f87c8e05dd76a68c843cc9f3c823217eeee93f59f49732233b106ab271459189`，与 E4 相同。Mia 配方固定为 `656879fd752749bbd11a6e1a065a326c00940911`，不是对其最新分支的性能宣称。
- 同一 NVIDIA FP4 checkpoint revision `b9ce18a6887e18c089fdea8509a33e63429a3300`；BF16 KV、纯文本、单请求 greedy、同 prompt token IDs、thinking 开关与输出上限。原生 prompt-cache/repeat-cache 关闭；context admission1017，内部/candidate capacity1024。
- 原生保留此前默认 MTP 配置：最大六草稿、累计概率阈值0.4、product streams5。候选为完整词表 Q4，BF16 backbone/selector；成本与先验冻结自 E6 独立校准，不加入 INT8，不按原生结果调参。E7 只驻留这一种草稿头，E6 四头共同驻留的旧速度不直接进入本轮汇总。
- 十任务包含八条 E6 中英自然任务与两条 E4 的 256-token 长输出控制，属于已见回归输入。五模式各四次，共200次正式计时。加载阶段为 native A / candidate B / candidate B / native A，独立卸载、重载，每个阶段各计时两轮。每任务、每模式先完整预热；不剔除正式首轮。
- 同一 loopback SSE 客户端测 wire TTFT 和完整请求时间。Backend decode 按实际 completion token 数减一及报告的 TPS 统一计算，排除首个输出；预热、加载与独立 oracle 不计入。逐任务耗时中位数求和后按各引擎实际 token 数汇总，不平均 TPS。完整请求时间包含实际 prompt 路径，不能用 decode 替代。

## 正确性和解释边界

候选每个阶段每个任务均有 serial oracle，fixed7 / dynamic7 各一次独立已提交 logits 和有效状态核对；计时输出与本路径串行结果完整 IDs 对比。八条 E6 输入还须与 E6 的完整输出 IDs 一致。原生每任务完整 serial / MTP 预热，并核对预热及计时的正文、推理、计数、停止原因和 token 摘要。其 API 只有截断为12位的 token SHA，缺少完整原生 IDs、logits 和 state；不能宣称跨引擎逐 token/数值全等。

跨引擎输出差异与客观答案必须随速度一起报告；相同输出子集只作诊断，不替代全任务主表。两条 capped 代码/散文只评价非空前缀。目标 prefill/serial 数值差异、eager BF16 草稿官方 fidelity 仍是既有边界；本轮比较的是实际执行路径，不是仅切换 MTP/DFlash2 架构的因果实验。原生 accepted/drafted 可含 EOS 后的内部草稿，接受率计数不直接与候选横比。没有多请求吞吐、长上下文、随机采样或完整 DeepInfer 引擎的结论。

## 实测结论

[完整数据](../results/e7/2026-10-21-tensorfold-match.json)。以下为每任务四次耗时的中位数，随后按实际输出 token 加权汇总；TTFT 为各任务 wire TTFT 中位数之和，不能当作单请求延迟。

| 输入组 | 原生 MTP decode tok/s | Q4 fixed7 | Q4 dynamic7 | 原生完整请求耗时之和 | dynamic完整请求耗时之和 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 自然完成8任务 | 43.58 | 44.32 | 45.57 | 8.835 s | 22.408 s |
| capped256两任务 | 53.60 | 45.22 | 45.66 | 9.956 s | 15.734 s |
| 全部10任务 | 49.18 | 44.86 | 45.62 | 18.791 s | 38.141 s |

自然任务候选 decode 比原生高4.56%，但全任务低7.23%；长输出控制中原生更快。候选 dynamic 比自己的 fixed7 高1.69%。四次全任务重复候选/原生 decode TPS 比为0.940、0.938、0.918、0.908；不能宣称整体反超。原生两个加载阶段存在速度变化，候选较稳定，只有两次加载，不能外推统计显著性。

全任务 TTFT 中位数之和：原生 MTP **1.653 s**，Q4 dynamic **19.775 s**。候选完整请求约为原生2.03倍；prompt路径是明确的优化重点。TTFT包含tokenization、状态初始化、执行及SSE传输，不能把它全部解释为GPU prefill。当前prompt仍逐token，尚无分块prefill提速结果。

候选动态选择宽度3/4/6分别28/361/637次。所有候选计时输出匹配自身serial完整IDs；40次独立非串行oracle的已提交logits和有效状态一致。原生MTP与自身serial摘要10/10一致，跨引擎8/10一致。`zh_prose`为原生63、候选62 tokens；`length_prose`双方256但正文不同。五类客观检查均通过，散文没有独立质量评分，截断控制仅评价非空前缀。相同摘要不能证明完整token/数值全等，也不能用于替代全任务主结果。

容器cgroup记录OOM/OOM-kill均为0，四阶段worker均结束。本轮不是E5调度桥或完整DeepInfer引擎性能；下一阶段单独验证分块prompt，不混入INT8或预算重新校准。
