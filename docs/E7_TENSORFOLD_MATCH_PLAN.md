# E7：当前 Q4 DFlash2 与原生 TensorFold 同机对照

2026-10-21，产生 E7 GPU 输出前冻结。比较本机此前核验的 MiaAI-Lab 配方安装态 TensorFold（与 E4 同一二进制/AOT），和 E6 完整词表 Q4＋原生 DeepInfer 动态预算候选。不是最新 upstream 排名，也不重新调原生默认参数。

主对照 native MTP vs candidate Q4 dynamic7。诊断保留 native serial、candidate serial、candidate Q4 fixed7。使用 E6 八条中英任务，再加 E4 原有两条长散文/代码 256-token capped 控制，**共十任务、五模式、每模式四次，200 次正式计时**。这些是已见回归任务，不称新质量测试集。自然 EOS 与 capped 分开；不删失败或较慢任务。两条长输出控制若提前 EOS，照实保留。

两套模型串行驻留，四个独立加载 session 按 native A / candidate B / candidate B / native A。每 session 每任务先完整运行所有该引擎模式一次，再计时两轮；native 模式交替，candidate 三模式循环（四轮不称完整三阶 Latin square）。预热和独立 oracle 不进入计时；不剔除正式首轮。完整控制 prompt/repeat cache 为零、greedy、BF16 KV、context admission 1017（buffer/candidate capacity1024）、同一 NVIDIA FP4 checkpoint revision、同一 thinking 开关/模板/输出上限。原生默认六草稿、累计阈值0.4/product streams5保持 E4 设置；candidate 为 E6 固定七草稿或动态0..7。只加载完整 Q4 草稿投影，不再共同驻留 E6 其他三头。

候选成本曲线/接受先验从 E6 两条独立校准冻结，不使用 E7 的原生输出调参；本轮不加入 INT8。十任务的 HF prompt IDs 在启动完整权重前准备，native `/tokenize` 与它逐项核对。候选八条 E6 输出还要对照此前完整 serial token ID；每 session 每任务的两种非 serial 模式独立核对已提交 logits/最终有效状态，合计40次。计时输出按各自 serial 校验；原生 API 只有 token SHA/正文/计数，没有完整 IDs、logits 或 state，此边界明确保留。

共同 loopback SSE 客户端测 wire TTFT、first-to-last 与完整请求时间；backend decode 单独记录并统一排除首个输出 token。检查 native runtime 的 TPS/count/time 口径，不把累计 accepted（可能包含 EOS 后草稿）直接与候选接受率横比。每任务取四次耗时中位数，按实际输出 token 加权汇总，不平均 TPS。输出差异、停止原因、客观答案、逐任务速度、各 session/重复波动必须随总表公布；相同输出子集只作诊断，不能替代全任务成绩。跨引擎数值不同仍是已知边界，不把实际路径速度比解释为只切换草稿架构的纯算法收益。

必须报告：完整集/自然集/capped 集各模式 decode TPS、TTFT、完整请求耗时、实际 token 数、峰值内存、OOM/失败、prompt ID 相等性、原生 MTP/serial 一致性、跨引擎输出摘要/正文差异、五类客观答案检查。先验与代码/二进制/配置哈希随证据保存。结果只能回答这台 Spark、这些任务、当前安装态下的单请求表现；不外推长上下文、并发吞吐、随机采样或完整 DeepInfer 引擎。
