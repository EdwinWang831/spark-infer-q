# K4 同负载扩展性能复核

**2026-10-24更新：主要轮询损失已定位并修复，见[等待修复复测](K4_WAIT_FIX_REPORT.md)。** 同期优化single/修复joint短组40.30→45.75，cap128组37.78→39.00；真实SSE短组32.57→45.00。仍保留显式研究入口，未完成广泛性能验收。以下为修复前的历史诊断，数据不与后续轮次拼接。

**修复前：功能链路通过，性能尚未达标。** 上一轮 31.09→38.58 token/s（+24.1%）只比较新 backend 内部的 split/joint，缺少已经优化好的单请求基线，不能证明双流扩展收益。本轮补上同负载对照，三组混合请求的联合路径均慢于原优化单流顺序完成两条请求。PR 保持显式研究入口，不应晋升默认路径；先修接入开销，再讨论 K5。

[公开证据](../results/k4/2026-10-23-matched-scaling.json)由 `k4batch_scaling_export.py` 重新检查；原[功能/SSE 报告](K4_BATCH_RUNTIME_REPORT.md)保留作为历史受控结果。

## 匹配协议

保留 E17 NumPy、W8/Q4 DFlash2、E33 精确头、当前草稿后动态预算和自然 EOS。四个模式使用完全相同的提示、输出上限、最终 tokens 和每请求完整预算/接受轨迹：

- `optimized_single`：现有优化单流引擎，各请求独立完成，合计 decode token 除以合计 decode 时间；这是顺序处理两请求的容量基线，不是两条流同时运行。
- `batch_single`：新 backend 每次只准入一个请求，隔离迁入本身的损失。
- `split_pair`：新 backend 同时准入两请求，但逐请求发起 Target ticket。
- `joint_pair`：同样两请求共享一次 Target forward。

所有请求在开始 decode 计时前恰好输出一个 token，程序对此逐个断言。计时从首 token 后开始，包括第一轮草稿、预测/预算、验证、commit、原生记账、事件和最终回收；预填充、准备 Graph、诊断均排除。每组各模式两次计时，逐次拒绝新 head JIT 形状。保留完整 token/E17 决策一致断言，没有为了扩展性能改变预算或强制固定草稿。

这是 token API 直接计时。它与上轮 SSE 不是同一边界：新 direct driver 轮询间隔0.5ms，上轮 worker 为1ms，因此也不能把二者的差额全部归因于 HTTP。

## 实际结果

单位为合计 decode token/s；single 模式是两请求顺序时间的加权总速率。

| 负载 | 原优化 single | 新 backend single | 新 split pair | 新 joint pair | joint 相对原优化 single |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原短英中，32/32输出 | 38.16 | 30.09 | 27.83 | 33.65 | -11.83% |
| 原长短混合，20/3自然输出 | 56.13 | 44.90 | 41.85 | 49.28 | -12.20% |
| 同短提示、cap128，57/128实际输出 | 38.50 | 28.66 | 27.85 | 30.55 | -20.64% |

回退在没有 HTTP 的 direct driver 中依然存在。新 backend 单请求已慢约21%～26%，因此不能把问题只归结为 DFlash 尚未双请求合批。原单流“约48”来自不同八任务的均值，不是每个输入都应该达到的值；本轮相同输入下，代码任务原单流较快，中文任务本身较慢。

## 重叠不足与有利控制

原短 pair 总共18次 Target forward，其中7次联合、11次单独；长短 pair 为1次联合、3次单独。两组只有8/22次联合，且较慢中文请求的短组31个 decode token 要18轮，平均每轮1.72个输出。输出上限提高到128后也没有消除长尾：两个请求输出57/128 token，10次联合后仍需55次单独 forward。不能将提高 cap 自动称为稳定双流。

另补两条相同提示的控制，确保从首 token 到结束每次 Target forward 都有两个请求，既测试高接受的代码，也测试较低接受的中文。它是有利的重复输入控制，不代替不同请求的实际混合负载。

| 全程重叠控制 | 原优化 single 顺序总速率 | 新 joint 总速率 | 变化 |
| --- | ---: | ---: | ---: |
| 同代码提示两请求 | 72.74 | 97.69 | +34.30% |
| 同中文提示两请求 | 31.03 | 40.73 | +31.24% |

联合 GPU 计算本身可以带来收益；混合负载中，进入单请求尾部后，新 backend 的开销会损害总体结果。这里记录的是 forward 数量，不把它伪称为已经测得的各阶段 wall-time 占比。

## 分段诊断与已排除的原因

独立的诊断运行使用修正后的 API 与 CUDA event 包围测量，计时阶段从 decode cohort 才启用。API 总时间有嵌套，不能相加；CUDA stream 时间可能包含主机未及时提交造成的空档，也不是纯 kernel 时间。初版剖析误包含 prefill、只累加了最后一个 single 请求的 API 阶段；它的剖析数据不采用，正式吞吐计时仍有效。

短组诊断中，原 single 的 `propose` 合计约272ms，新 backend single 约573ms；原 `run_predictions` 约1280ms，新 backend 包围 execute 约1442ms。较长输出组原 single `propose` 约821ms，新 backend single 约2071ms。这些是独立、带仪器的样本，不能将全部额外时间直接归因于 GPU 草稿算术。

额外做两个短上下文 × 四种草稿调用方式 ×16次均衡重复：原 provider、新请求包装每次新 handle、包装重用 handle、只在已证明缓存不读写的短上下文中共享未用 buffer 的复制 worker。全部草稿 token、E17 概率和 q 一致，各变体 median 都在约10.4～11.1ms；每次新建约50.12MiB的五个缓存分配没有产生数倍的稳定开销。**因此缓存复制不是当前数倍 `propose` 时间增加的主要解释。** 这个试验没有在生产共享可变缓存，也没有测 Target 与草稿交替后的缓存/调度行为。

修复前观察到、当轮尚未完成 kernel 归因的开销机制：

- DFlash 按请求串行，且每个 proposal 后仍有全设备同步。
- 新 Target 将 q 补齐到1/2/4/8；原两组125个实际行要计算140个物理行，多12%。原单流捕获精确宽度1～8。
- verify/head/commit 多阶段由主机推进，并存在固定轮询等待。

当轮最主要的待查问题是**为何相同草稿计算与新 Target 交替时会变慢**；后续[事件等待诊断](K4_WAIT_FIX_REPORT.md)已确认固定sleep的因果影响，并以kernel/CPU观察佐证。独立草稿控制已排除简单的类/精度/权重错误和缓存分配解释，当轮尚无证据将损失全部归因于某个 CUDA kernel。后续已补充 Target→draft 调度/同步和实际 kernel 证据；singleton/tail效率仍需继续改善；不以改预测器、改预算或上线更复杂全局调度掩盖问题。

## 回收与复现

实际 GPU 作业250、251、252、253全部由同一 resident 串行执行，PID319、load_count=1。最终255资源检查通过，Gloo/环境恢复，没有活跃 K4 对象，模型仍READY，allocated保持77,695,397,888 bytes；延续上轮已披露的96MiB stream库缓存，没有新增模型加载或清空共享缓存。

```bash
python3 scripts/k4batch_stage.py unique-scaling unique-scaling.json k4batch_scaling_validate.py --graphs
python3 scripts/k4batch_stage.py unique-profile unique-profile.json k4batch_scaling_profile.py --graphs
python3 scripts/k4batch_stage.py unique-overlap unique-overlap.json k4batch_overlap_validate.py --graphs
python3 scripts/k4batch_stage.py unique-draft unique-draft.json k4batch_draft_probe.py
python3 scripts/k4batch_scaling_export.py results/raw/k4-scaling results/k4/2026-10-23-matched-scaling.json
```

每次必须换唯一版本/作业名；正式结果只有两次重复，属于诊断。两轮脚本修订分别封存，raw保留最初源码和未采用的剖析，公开 JSON 的 `performance_gate_passed=false` 不会将旧 split/joint 内部改进误判为扩展性能过关。

完整 raw 封存 `evidence.tar` SHA256：`2300b49d9652660cd1c2bf3413bb428fe43ee2cdae7c198a2337a5c0055a63d4`。
