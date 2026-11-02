# K2：DFlash2 与原生 Scheduler 推测解码闭环

以下为分阶段封存的验收记录；最新范围见文末“移除固定模式”。

2026-10-23，基于 K1（PR #13）增量开发，不等待新预测器训练、不修改主模型权重。

固定 DeepInfer revision/扩展 SHA 与 E5/K1 一致。新增显式 max_drafts=7 原生上下文，退出恢复环境及 Python 缓存值；不启用 DeepSeek Worker/GPU sampler。原生 `schedule` 生成请求和宽度，适配器拼接 pending token 与原生 metadata 中的草稿，Target 验证并提交正确前缀，随后按原生顺序 `process_MTPInfo` → `process_outputs` 发布下一轮草稿和本轮输出。每个 schedule 均配对一次 process，包括分块 prompt、取消和空计划。

## 模式与范围

- K=0 沿用 K1；K=1/3/7 使用原生 Scheduler 空成本曲线的固定草稿长度路径。
- lagged/confidence 使用原生 Scheduler 内部 MTPVarlen；先限定单请求，明确注入单请求测得的成本曲线。每种宽度都计入已付出的完整草稿成本，含 width1。不把单请求曲线当作真实联合 batch 曲线。
- confidence 复用冻结 E16 线性预测器、特征与 Q4 DFlash2，不训练新模型。按原生 N-2 总预算/N-1 分配约束发布条件概率，不声称等价于旧的本轮 post-draft 控制器或重获历史约10%收益。
- 固定长度模式覆盖双请求交错；执行仍逐请求、同步、greedy。草稿 context 只追加 Target 已提交特征；单 owner 的草稿投影缓存在请求切换时显式重建。
- 原生模型长度为逻辑准入边界，runtime 额外预留14个位置用于原生最大验证/草稿 lookahead；同时检查 Target/DFlash 实际容量。终止 token 保持 pending，不提前处理。

## 执行前验收

1. 本地无 GPU 契约测试与 K1 回归。
2. 真实 C++ Scheduler + CPU toy Target：固定 K=1/3/7，全接受、首/中/末拒绝后继续、EOS/长度、取消（waiting/prepared prompt/prepared decode）、slot/generation、双请求、动态宽度选择与原生计数、逻辑块和 context 回收。CPU toy 不作为整模 GPU 证据。
3. 复用已有常驻 Target/DFlash、同一 PID/load_count；新源独立封存，旧实验源不覆盖。GPU 上英文、中文、JSON/代码受控提示，固定 K=1/3/7、lagged、confidence 对逐 token serial 的完整输出、已提交 logits、有效 GDN/QSA/PLE state 检查。另覆盖两请求固定 K、max_tokens=1、取消和复用。测试结束清理全部请求和原生槽位，恢复上下文，worker READY。
4. 通过核心检查后进行真实 loopback SSE 多 token 输出与取消检查；K1 测试保持通过。
5. 做小规模同输入、同运行时、交替顺序的 decode 墙钟对照，计入草稿/特征/预测/原生调度/验证/提交，不混入加载、prefill 或 oracle。受控样本只报告工程回归，不声称一般性能提升、最新 TensorFold 排名或多请求吞吐优势。

任何失败停止、保留原因；修复后新封存明确重跑。保持设备权重常驻，不为验证重载。随机采样、整模 Graph、异步流水、跨请求 GPU batch、完整 KV 页映射与长上下文仍属后续工作。


## 2026-10-23 E17 本轮动态预算补充验收（GPU 前封存）

`adaptive` 保留当前 E17 unweighted_f1 Python 线性预测器及原有 BudgetProbe 二进制。使用本轮累计概率同时作为 accept_rates/prev_accept_rates，剩余输出上限裁剪候选长度，context<768 选择 short 成本，否则 long；每个候选长度包含完整 draft/feature/predict 成本。原生 Scheduler 保守预留 NEXTN=7，通过其公开 `update_metadata_for_sampling_zos` 最终确定 q，再交给原生 process_outputs。宽度更改不会改变最大预留或 inflight 计数。限定单请求。首轮草稿仍在首 token 发布前预计算，因此 TTFT/decode 边界与旧循环不同，必须同时报告总请求时间。

新增真实 CPU finalization 各拒绝位置测试；GPU 三提示独立执行旧 E17 循环，逐轮比较完整宽度序列、输出，并继续对 serial 检查 committed logits/state；在 adaptive 模式重跑长度1及三种取消/复用、真实 SSE/取消。小规模计时同轮对比 none/fixed7/confidence/adaptive。旧 E16 lagged 结果作为历史配置证据保留，不代表现有组合收益。


## 2026-10-23 移除固定模式（复验前封存）

按用户指示，K2 API/CLI 删除 fixed1/fixed3/fixed7，默认 adaptive；运行模式只保留 none（K1 对照）、lagged、confidence、adaptive。所有推测模式按预算选择验证宽度，最大七个候选草稿只是候选上限。移除固定模式的当前测试、双请求推测路径与文档入口；历史比较数据按原配置保留并标为历史。

更新后的CPU suite检查动态预算/分块/拒绝/停止/取消及最终q；GPU suite保留三个动态模式、独立旧循环逐轮宽度对齐、oracle边界/停止/取消；在线suite只对比none/confidence/adaptive并检查SSE。新封存重跑，结果与旧配置证据分开记录。K1 target-only双请求不受本次模式移除影响。
