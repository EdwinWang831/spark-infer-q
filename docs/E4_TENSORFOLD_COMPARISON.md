# TensorFold 对照：速度趋势、贡献归属与验证缺口

核对日期：2026-10-20（America/Los_Angeles）。对照仓库为 MiaAI-Lab 的 [Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)，固定在 `656879fd752749bbd11a6e1a065a326c00940911`；我们的实测固定在 `69e7d7ae52d30a6b502e297b40cc5dd13750917e`。

**结论：我们相对自身基线的进步有实测支持，但尚未证明比该 TensorFold 配方更快。较长 DFlash2 草稿的价值已经得到支持；在线动态预算相对合适固定长度的优势尚未建立。当前贡献主要是跨框架适配、局部成本决策实验和严格的状态验证，而不是原创 SM121 内核或首次动态草稿算法。**

本次进行了公开源码/基准方法审阅，并重新核算已有实验数据，没有启动对方服务或运行同机交替性能对照。因此不把公开数字列成可直接排序的排行榜；E4 实测仍未完成。

## 公开性能与我们的实测

双方默认主模型使用相同 NVIDIA FP4 checkpoint revision `b9ce18a6887e18c089fdea8509a33e63429a3300`，均面向单台 GB10 DGX Spark，保留主模型 top-10 routed experts。不要混入对方 optional INT4/top-5 或其他 checkpoint 的速度。

| 条目 | 对方公开数据 | 我们已有数据 | 能得到的结论 |
| --- | --- | --- | --- |
| 单请求，sparkDash | 散文 46–55 tok/s；代码 67.9 tok/s；每请求 400 token | 没有相同输入、相同协议的结果 | 不能给出相对加速比 |
| 单请求，bench_streams | Zig 散文 49.8、代码 50.9 tok/s；较早 Python 对照 49.5/57.1；请求上限 256、三次中位数 | 八任务混合汇总：fixed3 38.10、fixed7 41.50、dynamic7 43.10 tok/s | 数量级相近，但没有领先证据 |
| 多请求 | sparkDash 五请求汇总：散文 132.7、代码 149.7 tok/s；另一 bench_streams 五请求为 198.2/205.5 | 当前单请求；没有多请求对照 | 我们尚无并发性能证据；两个工具的对方数字也不能混为同一测量 |
| 冷 prompt prefill | sparkDash 4k 约 2665、131k 约 2197 tok/s | prompt 逐 token 走 decode 算术，未建立可比 prefill 基线 | 对方在已公开功能和验证范围上明显更完整 |
| 上下文与服务 | 默认窗口 262144，公开约 195k needle 和混合长请求测试；API、vision、prompt cache、共享序列池 | 分配容量 1536，最长已测有效位置 278；无部署服务 | 不能从我们的 decode 数据推出长上下文或服务可用性 |

对方公开来源：[README 性能和条件](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/656879fd752749bbd11a6e1a065a326c00940911/README.md#L39-L165)、[bench_streams 实现](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/656879fd752749bbd11a6e1a065a326c00940911/tools/bench_streams.py)。我们的来源：[八行实测报告](E3_WIDE_BLOCK_REPORT.md)及[完整结果](../results/e3/2026-10-20-wide-block.json)。对方数字是作者报告，本次没有独立复现。

对方 README 的 bench_streams 引言写 thinking off，随后表后说明又写 thinking on。脚本传 temperature=0，但请求没有显式 enable_thinking，配方默认 THINKING=1。实际对照时必须检查实际请求/服务设置，不能自行选定一种解释。对方 68 ms 等 TTFT 使用重复 prompt 缓存，不能与冷 prompt TTFT 比较；README 已明确说明这一点。

我们的 43.10 是 `394 / 各任务中位 decode 秒数之和`，不是八任务 tok/s 的算术平均，计时排除 prompt 和首个 pending 输出。输入混合中文、英文、极短答案、代码和 thinking，全部自然 EOS。单个 clamp 任务动态为 75.91 tok/s，而解释任务只有 21.62 tok/s；挑前者与对方 67.9 宣称领先，或挑后者宣称全局失败，都不公平。两个 thinking 任务合计贡献 273/394 个计时 token，约 69.3%，汇总对任务构成敏感。

还存在 KV 精度差异：对方默认 FP8 KV；我们的宽度实验为 BF16 state/KV。其公开说明 FP8 与 BF16 约 98.8% top-1 一致率。不能把所有差异都归因于调度；也不能在没有对照时断言 FP8 导致了多少速度差。

## 是否存在好的发展趋势

有，但应使用同轮对照支持，而不是把不同实验的比值串起来：

| 实验 | 同轮观察 | 解释 |
| --- | --- | --- |
| 三草稿局部两级策略 | fixed3 38.55 → 两级 39.30 tok/s，+1.95%；仅启停更慢 | 新策略已有小幅净收益，幅度和覆盖有限 |
| 独立缓存/EOS 对照 | 原始 38.80 → 缓存+EOS 39.29 tok/s，+1.26%；缓存单独没有建立稳定收益 | 这是另一实验，未与学习策略组合 |
| 八行主实验 | fixed3 38.10 → fixed7 41.50 → dynamic7 43.10 tok/s | 长度扩展带来更大的改进；动态较 fixed3 +13.14%，较 fixed7 +3.86% |
| 附加校准固定长度对照 | 固定六草稿 42.20，dynamic7 42.21 tok/s | 动态相对固定六草稿仅 +0.0306%，未建立可靠额外收益 |

八行网络只验证前三个草稿已达到 39.76 tok/s，说明一部分收益来自 proposer 布局/候选变化；并非 13.14% 全来自动态预算。动态当前实际选择 2–7 个草稿，没有在这批计时中选零/一草稿；不能宣布低接受阶段启停问题已经解决。

八行主实验每个重复编号的总耗时均优于 fixed7，但附加静态六草稿对照并非每轮都赢。解释任务动态仍慢于 fixed3 和 serial。八行及缓存实验复用先前八个测试提示，不属于新的泛化测试。下一步的价值在于把这些受控进步放进公平的 TensorFold 基线，当前不能宣称从 39 到 43 的跨实验差值就是稳定、可部署的趋势线。

## 对方已经具备什么

源码明确显示对方不是只用固定长度：

- 默认 MTP 最大六个草稿；按当前草稿的 temperature-1 概率停止更深链。
- 默认使用累计概率乘积阈值 0.4；活跃请求数超过配置阈值后切换逐草稿置信度 0.5。配方把请求数阈值设为五，对默认并发范围使用累计规则。
- MTP 按深度逐步运行；第一草稿总是提出，更深草稿可以提前停止。源码也支持固定阈值/无提前停止及更深的上限，不存在“三草稿就是 TensorFold 的架构上限”这一结论。
- 接受前缀提交、拒绝后 GDN 状态恢复、批量验证、CUDA Graph 支持及 serving 体系已有实现；我们的 target 执行和 commit 正是在复用 TensorFold Python 实现。

依据：[配置](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/656879fd752749bbd11a6e1a065a326c00940911/scripts/config.sh#L87-L95)、[Stop/draft 实现](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/656879fd752749bbd11a6e1a065a326c00940911/patches/0005-engine.patch#L1738-L1785)、[served hybrid](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/656879fd752749bbd11a6e1a065a326c00940911/patches/0005-engine.patch#L1979-L2007)。审阅该固定版本的配方/patch 未见 DFlash2 或 DeepInfer MTPVarlen 接入；这不等于已经完成全领域新颖性检索。

## 我们的新增贡献应怎样表述

| 工作 | 本项目的具体新增部分 | 已有证据与边界 |
| --- | --- | --- |
| DFlash2 与 Qwen target 的跨框架适配 | 连接真实 target 中间特征、已有 DFlash2 权重、四流 hyperconnection 表示、selector、pending token 和 accepted-prefix 提交；支持四/八行 proposer | 工程集成贡献；DFlash2 模型和算法来自上游，不能称为我们发明；官方 FP32 对照通过，BF16 fidelity 未解决 |
| 根据实际耗时选择验证预算 | 原生 DeepInfer MTPVarlen 接入单 Spark/Qwen 路径，结合实测行宽成本、七深度条件接受率及滞后更新，比较“预期输出/时间” | 区别于对方置信度阈值规则，但目标和原生算法来自 DeepInfer；尚未胜过校准好的固定六草稿 |
| 两阶段局部决策 | 草稿前预测是否值得付出 drafter 成本，草稿后校准预期可输出步长，再按成本选验证长度 | 本项目新增的 greedy 策略实现；独立训练/校准/测试，三草稿实验净收益约 1.95%；不是首次提出 gating 或校准，七草稿版本未实现 |
| 系统性的正确性与归因实验 | 完整 logits/特征、有效 GDN/QSA 状态和拒绝后继续生成检查；拆开四/八行布局、草稿长度、动态、校准固定长度及独立 EOS/缓存对照 | 八行 44 个组合、272 次计时和 56 次 oracle 运行通过；这是可信工程/实验贡献，不代表独立 target 质量、随机采样或完整引擎已验证 |

最稳妥的项目定位是：**在 DGX Spark 的 Qwen 混合注意力模型上，将 DFlash2 并行草稿与 DeepInfer 成本预算连接起来，并验证不同草稿/验证长度下的状态正确性和净收益。** 当前可以称为有差异的系统集成与实证研究；还不足以声称原创通用算法、首次 SM121 专家内核或对 TensorFold 的性能领先。

特别需要归属清楚：保留 BF16 激活的 NVFP4 专家内核是 TensorFold 原实现的编译/执行验证；我们没有重新设计内核。44 个组合的验证是我们的工作，底层 accepted-prefix commit/replay 不是我们发明。八行来自现有 checkpoint 配置，解锁本探针原先四行的限制，不是突破该权重固有上限。CSF 加载桥接是适配工作，CSF 格式/解码器也来自既有实现。

## 最值得优先补的实验

优先完成 E4 同机对照，再扩大策略搜索；不要继续用不同输入的公开峰值指导优劣结论。

1. 固定双方源码、主权重、chat template 和 GB10 时钟/功耗设置；记录实际设置，不为追求峰值擅自改动。第一组使用相同 BF16 KV、text-only、单请求、显式 greedy 和 thinking 开/关；另设 FP8 KV/vision 默认服务配置为产品对照。
2. 在同一台机器分时交替运行 TensorFold 默认原生 MTP、DFlash2 固定六/七草稿、DeepInfer 动态预算及各自 serial。草稿模型及其状态内存属于各方案成本，均应包含；不能把一种模型的 MTP 模块当成另一种 proposer。
3. 使用新冻结的长短文本、数学、真实代码、JSON/工具任务；固定长度吞吐和自然 EOS 延迟分开报告，统一实际输出 token、停止标记和计时边界。旧八任务保留为回归，不当新测试。同时报冷 prompt TTFT、decode、prefill、端到端时间和请求间波动。
4. 跨 Zig/Python 路径先核对数值与质量，不能假定逐 bit 输出一致：对方已有约 98.9% teacher-forced next-token 一致率的诊断，我们也有未解决的 prefill/BF16 差异。不同输出可能改变速度，因此还要记录 token 序列、输出长度和任务正确性。
5. 单请求确认后扩至多请求和长上下文；再决定将 DFlash2 接入成熟 serving 路径，还是继续独立引擎。完整 DeepInfer scheduler、多请求预算分配、prefill/算子改进均仍在研究范围，不应只围绕 MTP。

如果成本预算在这些对照中仍与固定长度打平，工程默认应采用更简单且经验证的静态/阈值策略，把动态机制保留为研究分支。若能在独立任务分布、低接受阶段或多请求竞争下得到稳定优势，才有依据将在线自适应提升为核心创新主张。
