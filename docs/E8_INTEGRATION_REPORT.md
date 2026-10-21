# E8完整Q4动态生成：分块prompt接入结果

2026-10-21。[冻结方法](E8_INTEGRATION_PLAN.md)、[完整数据](../results/e8/2026-10-21-integrated-prompt.json)、[接口/运行说明](PROMPT_USAGE.md)。状态 **PASS_INTEGRATED_BOUNDED**。新的`e8_ours.py`完整Q4 SSE执行器默认8行prompt；原E7冻结源保持不变。

## 配对结果

单次独立完整加载内，对同一组E7十条已见输入，新旧prompt路径按任务/重复交替AB/BA。每条任务两个动态模式各四次，共80次正式计时。每任务/模式完整预热，关闭prompt cache；target仍NVFP4，KV及draft backbone/selector为BF16，完整词表Q4草稿头、E6成本/先验和E7动态decode算法不变。

以下是每任务四次中位数再求和，不是单请求延迟；decode按实际输出token数减一加权。两模式的全部输出IDs一致。

| 指标（10任务） | 逐token prompt | 8行prompt | 变化 |
| --- | ---: | ---: | ---: |
| 真实wire TTFT之和 | 19.688 s | **4.225 s** | 加速4.66×，耗时减少78.54% |
| 完整请求耗时之和 | 37.917 s | **22.665 s** | 加速1.67×，耗时减少40.22% |
| 内部prompt耗时之和 | 19.678 s | **4.215 s** | 加速4.67× |
| 加权decode token/s | 45.91 | 45.39 | -1.14%，没有decode提速证据 |

| 输入组 | 逐token完整请求 | 8行完整请求 | 完整请求加速 |
| --- | ---: | ---: | ---: |
| 自然完成8任务 | 22.300 s | 10.474 s | 2.13× |
| capped256两任务 | 15.617 s | 12.191 s | 1.28× |

四轮完整请求加速比为1.689×、1.652×、1.687×、1.668×；wire TTFT加速4.608×–4.728×。收益在本轮四次重复中一致，主要来自prompt。decode每轮均未改善，保留约1.14%的整体退化；动态宽度分布也不同（rows1宽度3/4/6为33/357/634次，rows8为36/255/730次）。原有运行成本反馈仍然启用，尚未隔离decode微退化与宽度变化的原因，不以此宣称预算优化。

## 正确性与计时边界

每任务先完整运行serial rows1参考，再运行serial rows8、fixed7 rows8、dynamic7 rows1和dynamic7 rows8。共50次完整预热，其中40个完整生成对照检查 **3,348行已提交decode logits** 和最终有效GDN/QSA/PLE状态；全部精确一致。所有完整输出IDs及prompt IDs回归E7结果。五类客观答案检查通过；散文仍无独立语义评分，两个capped任务仅是256-token前缀。

80次正式请求全部检查完整IDs。每次SSE响应完成后再独立检查最终有效状态，不将GPU状态比较混入wire计时；80次均通过。之前[E8 prompt本体实验](E8_PREFILL_REPORT.md)另行覆盖81组完整逐行prompt logits/五层draft特征/块边界状态及八步延续，不重复计入本轮40个生成对照。

SSE wire TTFT包含tokenization、状态初始化和实际prompt执行；完整请求包含decode、传输及响应整理。加载约434秒、oracle、预热和请求后状态检查均排除。单个Q4草稿头与完整target共同驻留，CUDA峰值allocated约72.96 GiB、reserved约73.40 GiB；OOM/OOM-kill均为0。计时结束后模型worker已退出。

## 使用与限制

`prompt_state`复用目标的`prefill=False` decode算术和完整commit路径，默认每块8行，支持短尾块；不会启用此前有数值差异的独立prefill算法。新SSE执行器默认8行，`--prefill-rows 1`可恢复逐token路径。接入和结果留在`codex/e8-prefill`实验分支，未合main。

这是一次加载内的四次配对重复、十条已见短上下文输入。没有新的原生TensorFold同轮计时，不能据此宣称已反超原生；E7原生结果仅为历史参照。也未验证多请求吞吐、长上下文、随机采样、官方BF16草稿fidelity或完整DeepInfer Worker/异步引擎。下一性能优先点是进一步降低剩余prompt开销并研究target验证/decode与预算反馈；INT8另做受控对照。
