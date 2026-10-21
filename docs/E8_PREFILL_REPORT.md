# E8：保留decode算术的分块prompt结果

2026-10-21。[预定方法](E8_PREFILL_PLAN.md)、[完整数据](../results/e8/2026-10-21-chunked-prompt.json)。状态 **PASS_PROMPT_ONLY**；现有默认路径未改。

## 结果

同一台SM121 Spark、完整48层NVFP4 target、BF16 KV、capacity1024，使用已有`WideTarget`的`prefill=False`。保留decode算术，每次提交全部块内输入，捕获五层draft特征。加载291.33秒，不计性能。

| 每块token数 | 十任务prompt耗时中位数之和 | 加权prompt token/s | 相对逐token |
| --- | ---: | ---: | ---: |
| 1 | 19.679 s | 23.43 | 1.00× |
| 2 | 11.166 s | 41.29 | 1.76× |
| 4 | 6.659 s | 69.23 | 2.96× |
| 8 | **4.220 s** | **109.23** | **4.66×** |

输入共461 tokens，复用E7十条prompt；每任务每宽度先完整预热，再以循环改变的顺序各计时四次，共160次。每任务取四次中位数再求和；不是单请求耗时。8行方案耗时减少78.55%。四轮分别汇总的8行加速比为4.573×、4.728×、4.730×、4.635×，本轮趋势一致；不作跨硬件或统计显著性外推。

## 正确性和范围

十条完整prompt加第一条prompt的1..17 token前缀，共27条×2/4/8三种宽度，**81组对照全部通过**。在每块边界检查所有行的完整logits、五层draft特征及有效GDN/QSA/PLE状态，与逐token执行精确一致；每组又核对八步greedy target-only延续的logits、特征、有效状态与pending token。短尾块已覆盖，不只检查末尾argmax。

主机同步计时包含状态初始化、特征捕获和context拼接，排除tokenization、网络/SSE、模型加载、oracle和后续延续。没有加载Q4草稿模型，也未执行完整投机生成；不能把109.23 prompt token/s当作decode速度或wire TTFT。E7原生TTFT还包含其他开销，不能将本轮4.22秒与原生1.65秒直接组成新的公平服务对照。

CUDA峰值allocated约70.54 GiB，reserved约71.77 GiB；容器OOM/OOM-kill计数为0。未启用CUDA Graph、独立prefill算术、INT8或新预算调参。模型权重及第三方二进制不提交。

本轮证明了现有正确算术下的prompt提速空间。下一步将8行prompt路径接入完整Q4动态生成，保持E6先验/成本固定，核对完整输出和已提交状态，再配对测量wire TTFT、完整请求与decode；届时才评估是否缩小原生服务差距。
