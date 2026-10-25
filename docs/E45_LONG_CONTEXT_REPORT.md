# E45：有界草稿历史与完整 4K 请求验证

2026-10-25。**4K 正确性与生命周期门槛通过；没有证实新的速度收益。**
本轮为“每个并发请求支持 1M tokens”打基础：将 DFlash 原始特征历史改为有界窗口，
保留 target 完整历史，并验证此前未覆盖的 >2K QSA/Graph 路径。
尚未实现 1M 模型执行、分页/量化 KV 或多请求 DFlash。

[事前方案](E45_LONG_CONTEXT_PLAN.md) · [脱敏证据](../results/e45/2026-10-25-bounded-4k.json) ·
[使用方式](E45_LONG_CONTEXT_USAGE.md)

## 实现与边界

- `FeatureWindow` 为每个请求独立持有 2040 行镜像环形缓存；只写 target 已提交的五层特征。
  两份镜像使有序后缀可直接借用连续视图，读取时无需拼接；视图有效到下一次 append。
- `RollingContextDraft` 移动并复用投影、K、V，RoPE 使用绝对位置；请求切换时重置所有者。
  `context_start + retained_rows == target.pos`，预测器和预算控制器继续收到全局位置。
- target KV、QSA index、GDN、PLE 保留完整历史。拒绝的草稿不进入已提交特征或状态。
- 草稿使用共同的最后 2040 行，加 8 行草稿块不超过训练窗口 2048。
  **这是显式的 proposal 后缀变体，不宣称等价于 SGLang 对各 query 分别处理的完整 sliding mask。**
- `optimized_target(session, capacity=4608)` 和 `optimized_provider(...,
  history_window_rows=2040, max_request_context=4608)` 显式启用。默认仍为原路径。
  本轮 physical4608/logical4352，输入最长4096，输出上限64。
  构造器允许8192只是未来实验入口，不代表已验证8K。
- prefill 使用 E44 Mia tile、chunk1024；decode 使用 Graph/E33、W8 MLP/Q4草稿头、
  BF16 attention/selector、E17 NumPy 与 E29 成本。>2K 预测器/成本保持冻结，没有重新校准。

## 完整数值与缓存检查

同一份常驻真实权重，native rows8 对照自主 prefill：English64、boundary2049、mixed3073、
code4096，完整末行 logits、全部9282行五层特征、所有有效 GDN/QSA/PLE 状态逐位一致。
prefill 与 decode 的 QSA scratch 均实际启用。

每条前缀验证 width1–8，奇数完整提交、偶数半提交：共32窗口、148行完整logits，覆盖全部
16种 width/parity 桶，实际 Graph replay，无 fallback。这些是完整模型对照，并非选定层检查。

使用真实4096行 target 特征，在位置2040/2041/2048/2056/3073/4096检查滚动缓存。
原始后缀、投影、K、V、hidden、proposal 与置信信号，均与相同绝对位置的完整重建参考逐位一致。
窗口 start 分别0/1/8/16/1033/2056，存储始终104,448,000 bytes。

## 调度停滞：失败记录与修复

首个完整请求 study-v1 在 code4096 超过原有100,000步上限。原因不是上限太小：
DeepInfer 预先分配完整 prompt 的逻辑块表，首个1024行块之后，使用257/275个块，
超过其 `GPU_CACHE_PREFILL_WATERMARK=0.9`；余下 prompt 仍在 waiting，却无法继续 prefill。
3073-token 请求占用较少，没有触发这一问题。

桥接层将原有计数池向上取整扩为 `ceil(original_blocks / 0.9)`，让所有已准入槽位在接近
容量时仍有水位余量。这些 block ID 只做 host 逻辑计数，**没有额外分配物理 GPU KV**，
没有改变上游算子、采样或原生调度决策算法。
这一逻辑计数修复对桥接层统一生效；有界草稿历史仍需单独显式开启。

真实 pinned C++ Scheduler + 确定性 host backend 的回归先复现旧池：第一块之后连续7步空转。
新池306块完成4096及4288-token单请求，新池610块完成两个4288-token请求，另检查
capacity2048/输入1984。四组均零空转、输出正确、所有逻辑块回收。完整模型随后通过4K请求。
study-v2 为 oracle/dual 设置有限、按工作量计算的步数；没有放宽数值门槛。

## 请求测量

比较两种**相同 proposal 语义**：full 臂保留全部原始 target 特征、给 drafter 相同绝对位置后缀；
bounded 臂只保留镜像环。不是“4K无窗口DFlash”与窗口的比较，也不是与原生TensorFold的排名。

四条输入，每臂一次预热，再四轮AB/BA：8预热、32正式请求。全部输出、事件位置、完整
接受/宽度/概率/head轨迹、特征后缀和最终有效状态一致，并与单独 target-only greedy oracle一致。
正式计时没有新增 capture、head JIT 或 Graph/head fallback。

四次均值，单位毫秒；decode 使用四次耗时之和计算吞吐：

| 输入 | 输出tokens | TTFT full→bounded | 完整请求 full→bounded | 残差decode full→bounded |
| --- | ---: | ---: | ---: | ---: |
| English23 | 64 | 108.45→106.95 | 1095.90→1107.21 | 63.80→62.98 tok/s |
| Chinese23 | 51（EOS） | 107.30→108.38 | 1758.16→1757.83 | 30.29→30.31 tok/s |
| Mixed3073 | 64 | 2010.05→2052.17 | 2688.86→2719.02 | 92.81→94.47 tok/s |
| Code4096 | 64 | 2607.28→2592.07 | 3273.43→3260.73 | 94.57→94.22 tok/s |

相同972个输出tokens/臂，完整wall35,265.38→35,379.17ms（增加0.32%）；
相同956个首token后tokens，残差decode **60.001→59.972 tok/s（−0.049%）**。
本轮没有预设速度晋升门槛，不把小幅双向波动视为提速。残差decode为token API完整wall减
native首事件TTFT，含host边界开销；HTTP分词/客户端传输不在这个计时中。

四条输入中观察到的草稿接受/已观察草稿分别47/56、20/44、55/55、55/55；两臂完全相同。
两条长输入是重复/截断的工程输入，100%接受率和约94 tok/s不能推广到自然长文或1M。
没有对比更长上下文的原始无窗口proposal接受率，也没有完成长文检索/多跳推理质量评测。

## 内存与 1M 目标

五层特征宽12800、BF16，原完整历史为每token25,600 bytes；镜像环固定
`2 × 2040 × 12800 × 2 = 104,448,000 bytes`（**99.61 MiB/请求**）。
这是原始草稿特征缓存，**不是整个模型或整个KV的大小**，不包含投影、scratch、权重和target状态。

当前版本首块就分配整个镜像环：English/Chinese终态完整历史原本仅约2MiB，
mixed3073原本76.56MiB，code4096终态原本101.54MiB。因此4K仅略省，短输入反而多占内存；
它的意义是消除后续随上下文无限增长的这一项，默认未开启。

以下为布局推算，**不是1M模型实测**；1M按1,048,576计算，target无原生MTP缓存：

| 1M时每请求的项目 | 推算 |
| --- | ---: |
| 原始完整DFlash特征 | 25 GiB |
| 新镜像特征环 | 99.61 MiB |
| 当前BF16 target KV + QSA原始/池化index | 27.75 GiB |
| 两个1M请求的上述target缓存 | 55.5 GiB |

target配置为48层，36层GDN/12层full attention、KV heads2、head_dim256；
QSA index_dim128、压缩比4。BF16增长项为
`12 × (N × (2 × 2 × 256 × 2 + 128 × 2) + ceil(N/4) × 128 × 2)`。
target张量约69.85GiB，尚有DFlash、GDN、Graph/prefill scratch、PLE映射页与系统占用。
所以当前BF16布局不能作为单Spark多并发1M的可行方案，CPU搬移也不会增加统一物理内存。

下一阶段优先做**有真实物理页映射的target KV管理，以及受控KV压缩**，逐步测8K/32K，
再扩大到128K/256K；同时保留QSA index和GDN状态的完整预算。分页解决分配/复用，
量化解决每token容量，两者都要接入attention/Graph并验证拒绝回滚、取消与请求隔离。
当前checkpoint的`max_position_embeddings=262144`，1M还需要单独证明位置扩展和长文质量。
多请求DFlash预算/批处理与现有K4路线协调，不能用本轮target-only隔离代替该项验证。

## 生命周期、审计与复核

4096-token请求的waiting/prepared-prompt/prepared-decode取消及同ID复用通过；
真实中文HTTP prompt2359 tokens，SSE24token/length与1token/cancelled通过。
两个准入的mixed3073/code4096 target-only请求逐个对照独立oracle，输出和终态一致。
设备执行仍串行，不是GPU batching、多请求DFlash或并发吞吐测量。

一直复用worker46235/load_count1，没有模型重载、下载或清理共享cache。
失败后的独立审计473、成功后的独立审计476均回到本轮稳定基线：
**allocated77,393,408,000→77,393,408,000 bytes**。原算子/no_grad、group/env及实验模块释放通过。
E43早先32MiB未解释差额仍保留，不用本轮稳定基线覆盖它。

原始归档448文件、1,440,402bytes，SHA256
`f2637075cc19a07674347711ffd67bf9dc082416e3993f3b34372fa662e40592`，
位于忽略目录`results/raw/e45/2026-10-25-final-v2/`，包含pilot、失败study/审计、成功study/审计、
各自冻结源码/job、外部源码、配置和实测binary，不含权重。

```bash
python3 scripts/e45_export.py \
  --raw results/raw/e45/2026-10-25-final-v2/worker \
  --out results/e45/2026-10-25-bounded-4k.json \
  --pilot-version pilot-v1 --pilot-job 470-e45-pilot-v1.json \
  --study-version study-v2 --study-job 474-e45-study-v2.json \
  --audit-version audit-v1 --audit-job 476-e45-audit-v1.json
python3 -m unittest discover -s tests -q
```

导出器核对源/job/外部依赖/当前库指纹、失败记录、匹配工作量、数值/QSA/Graph/缓存、
oracle/HTTP/生命周期/双请求边界和最终内存。公开JSON不含私有路径、对象ID或token IDs。
本地352项测试：333通过、19项可选依赖跳过；另对真实证据的数值、QSA、缓存与逻辑块
回收字段做篡改拒绝检查，379条本轮涉及的文档链接通过。

后续更新：[E46](E46_256K_REPORT.md) 按用户要求直接完成单请求256K实跑，未逐级停在8K/32K；
本报告的下一阶段段落保留当时规划，当前优先级改为KV压缩和多请求工作区复用。
