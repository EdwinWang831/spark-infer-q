# E48：双256K请求的Prefill/Decode公平调度

2026-10-25。**在有长prefill等待时插入decode机会，首请求最长输出停顿从225.42秒降至0.842秒，完成时间从453.03秒提前至241.60秒。** 两请求总耗时453.24→452.27秒，基本持平；这是并发响应延迟改善，没有证实通用吞吐提升。

[事前方案与预热修订](E48_PREFILL_FAIRNESS_PLAN.md) · [显式用法](E48_PREFILL_FAIRNESS_USAGE.md) · [脱敏证据](../results/e48/2026-10-25-prefill-fairness.json)

## 改动与边界

`SchedulerBridge`和`shared_graph_engine`新增默认关闭的`prefill_interleave`。显式开启时严格要求单rank、两个请求槽、NEXTN=0。使用原生DeepInfer Scheduler已有的interval2轮询和暴露的`decode_step`阶段：存在等待prefill和已就绪decode时，一个prompt计划完成后让出一次decode机会，然后再推进prefill；没有就绪decode时继续prefill，没有等待prompt时保留原生decode进度。

原生C++仍决定请求、行数、队列、逻辑块与停止条件。这里调整轮询阶段，不修改MTP的schedule/process计数。GPU工作仍串行执行；没有加入GPU batch、多请求adaptive DFlash或HTTP服务支持。模型算术、缓存格式和默认设置保持原配置。

## 同期双256K配对结果

同一常驻模型、INT8 KV、两份私有状态/Graph及公共串行scratch，chunk1024；logical capacity262144、physical262160。每臂同时准入两个不同的262080-token输入，输出上限64。A输出24 tokens、B输出27 tokens，均遇EOS，各自三个合成检索键全部答对。

| 指标 | 原生prefill优先对照 | 显式交错候选 |
| --- | ---: | ---: |
| A首token时间 | 224.950秒 | 223.891秒 |
| A最长相邻输出停顿 | **225.422秒** | **0.842秒** |
| A完成时间 | **453.029秒** | **241.596秒** |
| B首token时间 | 450.228秒 | 450.570秒 |
| B完成时间 | 453.205秒 | 452.236秒 |
| 两请求总wall | 453.237秒 | 452.273秒 |
| 有decode等待时连续prompt计划上限 | 256 | 1 |
| 峰值CUDA allocated | 88267132928 bytes，82.21GiB | 相同 |
| 最低系统可用内存 | 15391379456 bytes，14.33GiB | 15416020992 bytes，14.36GiB |

A最大停顿减少99.63%，完成时间减少46.67%；B首token时间增加0.076%，总wall减少0.213%。事前要求A停顿至少减少90%、A更早完成、总wall及B首token退化不超过5%，全部通过。

这是**一组完整长输入配对**，先对照后候选；不是多轮统计收益，也不采用E47历史计时作为对照。两臂均从submit开始到drain及最终同步计时，无中途诊断暂停。模型/Graph/算子准备在计时外，完整状态指纹在计时后计算。每个请求的TTFT/完成时间从其提交起算，总wall包括两次提交及最终同步；原生事件用于重新计算输出停顿。

这里限制的是连续prompt计划数量。实际等待仍取决于单个1024行prompt计划成本，不承诺固定毫秒级deadline。A较早完成，也没有消除B等待长prefill的约450秒首token时间。

## 正确性与生命周期

两臂每请求的输入/输出token SHA、终止位置，以及完整已提交KV、scale、QSA index/pool、有效GDN recurrence/conv和PLE状态指纹一致。输出也与E47同输入结果一致；A与B使用不同输入和检索键。

两臂共用32个预先capture的私有状态Graph桶，每臂实际decode replay49次（A23、B26）；正式计时内Graph/head fallback均0，kernel源码哈希不变。没有重载模型或下载权重。

实际固定版本原生Scheduler搭配确定性host backend完成六种场景的两策略对照：混合长度、early EOS、late admission、取消waiting、取消prepared prompt、取消prepared decode。12个case的输出、同ID代际复用、逻辑块回收通过，无空设备计划。真实GPU另完成两次短请求预热、四次AB/BA正式短pair、两次late-admission pair，以及三个阶段的取消/复用检查；输出和完整状态检查全部通过。

本地完整测试376项：357通过、19跳过。CPU测试校验阶段控制、默认行为与范围限制，以及事件计时、位置/工作量、Graph replay和证据拒绝条件；GPU证据来自常驻worker作业。

## 保留失败与回收审计

首次`study-v1`在第一条短预热中因`timed head JIT changed`停止，尚未进入256K测试。`screen.prepare(rows)`没有提前注册所有按候选并集大小特化的subset kernel。修订在所有请求计时前预热subset counts1..128；允许warmup补充kernel登记，所有正式计时仍要求哈希不变且fallback为0。原失败及独立清理审计保留，不重标通过。

| 阶段 | 作业 | 结果 |
| --- | --- | --- |
| 原生pilot | `510-e48-pilot-v1.json` | PASS_E48_PILOT |
| 原始预热 | `512-e48-study-v1.json` | FAIL，保留原错误 |
| 失败回收审计 | `513-e48-failed-audit-v1.json` | PASS_E48_AUDIT |
| 完整正式研究 | `514-e48-study-v2.json` | PASS_E48_STUDY |
| 最终独立审计 | `516-e48-audit-v1.json` | PASS_E48_AUDIT |

同一worker PID46235、load_count1，权重对象不变；原始算子、no_grad、进程组和环境恢复，实验模块释放。最终CUDA allocated回到77393408000 bytes的本轮稳定基线。没有empty_cache、服务停止或提高分配限额。这个后续稳定基线审计不抹去E43历史32MiB分配差额。

## 证据复核

原始归档位于忽略目录`results/raw/e48/2026-10-25-final/`，459个文件、1464751 bytes，无权重；归档SHA256为`575a3a6c545ee7c7f50c01f7386288084f5677a42bccc7e93100714b82b4dd34`。已提交JSON包含脱敏报告、源码/作业哈希、计划和事件。取得归档后可离线重新导出：

```bash
python3 scripts/e48_export.py \
  --raw results/raw/e48/2026-10-25-final/worker \
  --out results/e48/2026-10-25-prefill-fairness.json \
  --pilot-version pilot-v1 --pilot-job 510-e48-pilot-v1.json \
  --study-version study-v2 --study-job 514-e48-study-v2.json \
  --audit-version audit-v1 --audit-job 516-e48-audit-v1.json
```

导出器检查阶段来源链、常驻身份、失败与回收、当前库源码、外部源码/二进制、实际工作量、事件时延、状态指纹和延迟门槛，结果为`PASS_E48_EVIDENCE`。

## 当前结论与下一步

E48修复了E47已观察到的长prefill饿死decode问题，在这组双256K target-only研究中保留输出和状态。**E47原始raw-logit L2精度门槛仍失败**，INT8与调度候选均不晋升默认；本轮不能证明BF16等效、自然长文质量、随机采样或每请求1M支持。

下一优先级是独立长上下文精度/质量校准，再验证长请求HTTP响应与取消恢复。多请求DFlash、真实GPU batching、物理分页和超出checkpoint262K位置配置的质量仍需各自证据。
