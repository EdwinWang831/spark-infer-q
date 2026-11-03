# K4：草稿 attention 缓冲、query CUDA Graph 与 Target 分解

本轮为既有 batch DFlash 增加独立可选的缓冲复用和 query Graph 执行，保留 E17、Q4/W8、E29 动态预算及当前 greedy 契约。完整服务仍为双请求，默认不变。性能结论以本报告的同期完整 SSE 对照为准，不将草稿组件改善或此前 prefill/核放置收益相乘。

公开数据：[2026-10-24-draft-query-graph.json](../results/k4/2026-10-24-draft-query-graph.json)。此前批量草稿相对逐请求草稿的数值差异，仍见[原报告](K4_OWN_BATCH_DRAFT_REPORT.md)；本轮比较的是同一 batch 草稿内部三种执行方式，不能抹去原限制。

## 1. 实现边界

`BufferedBatchedDFlash` 保留原 batch 草稿实现作为默认和数值参考。attention 的请求数、实际最大长度、matmul/softmax 维度与计算顺序不变，BF16 K/V 通过广播直接复制到 FP32 GQA 工作区，替代每层的 BF16 拼接、repeat 与转换中间分配。BF16 到 FP32 转换精确；CPU/GPU 检查还验证最终 attention/hidden，而非只检查布局。

每种请求数只保留最近一个精确长度的 attention 工作区，五层复用；长度变化时重新分配，未实现可变长度的融合 attention kernel，也没有去掉每层全部 context 拷贝。padding 每层清零，避免伴随请求长度变化后读到旧尾部。context 缓存仍按带 generation 的 handle 隔离，追加行数只在五层完成后发布。

query Graph 每层拆为两段：输入归一化/卷积/QKV/RoPE，以及 attention 后输出投影/卷积/MLP。最后归一化并入第五层后段。两种请求数各10张 Graph，单请求尾段使用原3维 attention 路径。Graph 输入只绑定自有 ids、positions 和 attention 工作区，不绑定某个请求的 context/cache 地址；所有 context 投影和 attention 保留原精确长度，selector/Q4 head 也不在本轮 Graph 中。

每个片段使用独立的 Graph pool，避免不同片段覆盖仍存活的 residual/输出。输出沿用“下一次 forward 前有效”的借用契约。服务在 owner 所用 stream 上预热两种请求数后才接收请求，启动成本单列。捕获或执行异常会隔离 graph owner；不发布部分追加的 cache.rows，不自动重试已经失败的 owner。关闭要求先释放请求，完成设备工作后再清理 Graph 与工作区。

没有改变模型权重、量化、E17 特征/NumPy 实现、动态预算目标或引入固定草稿模式。仍使用显式 W8/Q4 provider 和原有低层兼容算子；不是完全移除 TensorFold。

## 2. 三组完整服务对照

所有组共用 compact Target、E40 chunk128/native budget128、ready-decode 交错、performance CPU owner、同一模型与精度、E17 和原 E29 控制。三组分别为原 eager batch 草稿、仅复用缓冲、缓冲加 query Graph。成本表未重校准；三个组保持同一旧控制，不将其称为批次最优预算。

短输入 cap32、cap128 和短流先输出后提交长 prompt 三组，每组每方式五个独立 cohort，轮换顺序，共45个正式 SSE cohort。每种负载先预热全部方式，再完成该负载的五轮，减少不同负载占用有界 Target LRU 引起的抖动。staggered 到达由观察到首个输出触发，不用固定 sleep。所有正式输出对照 gold greedy tokens；相同直接诊断中的验证宽度序列也必须相同。正式计时内不允许新增 draft/Target capture 或 exact-head JIT；每次冷形状观察立即保存，固定最多补跑六次，第一组无捕获/JIT的结果才进入正式记录，不按速度选择。全部预热与晚形状记录单列。

完整请求吞吐从最早请求提交到全部完成，包含 prefill、HTTP 和回收。SSE cohort 从最早首输出到全部完成，扣除每请求恰好一个首 token，可能包含伴随请求 prefill。两者都不是两条流各自速率之和，也不能与单流约48 tok/s直接比较。每组每方式仅十个请求，小样本 p95 不提供生产尾延迟保证。

| 两请求负载 | 原 eager 完整请求 | 仅缓冲完整请求 | 缓冲+Graph完整请求 | 原 eager SSE cohort | 缓冲+Graph SSE cohort |
| --- | ---: | ---: | ---: | ---: | ---: |
| 短输入 cap32 | 40.84 | 41.47 | 41.42 | 42.89 | 44.78 |
| 混合 cap128 | 39.09 | 38.48 | 40.13 | 39.88 | 40.79 |
| 短流输出后长输入到达 | 26.61 | 27.05 | 27.46 | 27.13 | 28.02 |

单位token/s。Graph相对原eager完整吞吐的配对cohort bootstrap95%区间：
- 短输入 cap32：+1.41%，区间[-0.02%, +2.94%]。
- 混合 cap128：+2.68%，区间[+0.59%, +5.56%]。
- 短流输出后长输入到达：+3.20%，区间[+1.33%, +5.78%]。

短组尚未建立稳定的Graph收益；三个负载均未达到预设的+10%服务目标。仅复用缓冲的收益也不一致：短组+1.54%，cap128−1.55%，staggered+1.67%。Graph相对仅缓冲的完整吞吐分别−0.12%、+4.30%、+1.51%，因此不能把缓冲与Graph各自的局部收益相乘。

cap128第一个提示的TTFT小样本p95由232.47→310.51ms，另一个提示321.69→309.15ms；按提示分组存在回退，不能宣称延迟全面改善。该组TPOT p95分别19.77→18.71ms、35.58→35.80ms。只有五个cohort/每提示五个请求，不将其解释为长期尾延迟界限。默认不晋升。

同组平均累计草稿墙钟801.83→689.27ms，verify3583.24→3570.49ms；草稿省下的时间受整个验证阶段占比限制。以上为同轮同期对照，不与此前44.00/39.73 SSE cohort等不同轮次绝对数字计算提升率。

| 草稿组件，7轮中位ms | 原 eager | 仅缓冲 | 缓冲+Graph |
| --- | ---: | ---: | ---: |
| 短混合 | 14.082 | 14.181 | 13.354 |
| 长混合缓存 | 15.930 | 15.976 | 15.305 |
| 单请求尾段 | 9.522 | 9.210 | 8.388 |

这些是独立草稿组件数据，排除Target和服务；长context用重复特征合成，不能用它预报真实长请求吞吐。20个query片段的单独启动预热耗时89.54ms，排除模型加载/Target/头部准备。组件最终attention缓冲约43,384,832 bytes；不是Graph私有pool或完整服务总显存。

正式45组均无新增draft/Target capture、head JIT、eager Target forward或Graph驱逐；Target最终30张，保持32张上限。15组预热和1组发生Target捕获的冷样本全部另存，未丢弃其记录。

## 3. 数值、生命周期与资源

服务诊断通过1,386 logits行、1,386 feature行、366次有效状态与24个prompt交接检查；全部45组正式输出一致。组件覆盖短/长不等长 context、只剩一个请求、初次与重复调用、增量追加、同位置更换 anchor、伴随请求内容隔离，以及长 context 的 K/V 前缀。hidden、proposal IDs、完整 selector/E17 signals 和有效缓存与原 batch 路径逐项精确比较。合成重复的真实 prompt 特征只用于长缓存组件探针，不能代替真实服务输入。

故障注入在第三层追加 K 投影时失败；此前有效缓存前缀保持一致，追加行数未发布，后续调用被隔离。真实 GPU 生命周期复用14项既有场景，包含 NEXTN0/7、自然EOS/cap、取消/survivor/同ID新generation、在途fence、关闭超时、native guard、故障后新owner。CPU/Torch 实际执行11项，包含不等长 padding、缓存追加、代次与预算发布检查。本地267项，249通过、18项因可选环境跳过；其中3项新 tensor 测试在远端 CPU/Torch 已实际执行。

全部 GPU 工作经共享 resident 队列串行，不重载/停止模型、不清空共享缓存、不修改其他实验。最终审计410无活跃批次对象，原生Gloo/NEXTN与主线程CPU0～19 mask恢复，二次GC后allocated稳定为78,030,942,208 bytes；worker PID319/load_count1并回到READY。raw包SHA256为 `a784b2612c2e7ef4c661c4e65096683f27bd2e9abc3c6eba53c49545222802b2`，完整Graph/工作区与资源数据保存在公开JSON。审计后的共享进程变化不能归因于本轮。

393 CPU runner 曾把测试方法名中的 `skipped` 当成跳过，错误拒绝已经成功的8项测试；改为匹配 unittest 结果摘要后，398完整执行两套共11项并通过。392/394/399组件、396服务、395 Target分解和397生命周期为较早快照，均保留，不与最终406/409/407/404混合。403在首组正式测量仍有新增Target捕获，触发无capture门槛而失败；原先一次补跑不足以处理second-sight捕获。409按上述明确的负载分组与有界补跑规则重测，保留全部冷样本，不将403称为运行时正确性失败。最终版本还补充启动预热的临时handle清理和Target诊断接受前缀后的状态核对；先前服务数据不用于择优替换最终结果。

## 4. Target 内部耗时

这是独立的固定候选 Target 诊断。使用真实短/长 prompt 状态，固定3+5、8+8及两个 singleton 形状；每个形状先核对完整 logits/features 和接受前缀后的有效状态，再对已捕获 Graph 运行五次。数据不包含 draft、E17、prefill、head、commit 或 staging，因此不当作完整服务提升。

CUDA event 记录互不重叠的模型执行区间；PLE 内的投影不再重复计入投影，MoE 内部调用也整体归 MoE。事件节点本身有开销，区间可能包括流上空隙，不能称为逐 kernel 活跃时间之和。它用于确定下一轮采样优先级，不能直接把某项归零推算最终 tok/s。

| 固定验证形状 | 模型循环总区间ms | MoE | 输入+输出投影 | HC | GDN前/中/后合计 | QSA attention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (3, 5) | 55.58 | 26.76 | 15.18 | 9.04 | 2.15 | 1.44 |
| (8, 8) | 64.18 | 33.68 | 15.91 | 9.40 | 2.60 | 1.55 |
| (3, 0) | 42.09 | 16.25 | 14.53 | 8.65 | 1.32 | 0.36 |
| (0, 8) | 51.53 | 23.65 | 14.91 | 8.81 | 1.71 | 1.47 |

3+5行的MoE约占模型循环48.2%，输入/输出投影27.3%，HC16.3%；8+8行三者合计约92%。这些区间仍包含仪器事件开销及可能的流空隙，并未证明某个具体MoE内核的瓶颈。后续先细分路由/专家计算/数据搬运，再选适配DeepInfer的SM121算子或自有实现。

## 5. 复现与后续

在既有准备环境、已加载模型和唯一新版本/作业名下运行：

```bash
python3 -m unittest discover -s tests -q
python3 scripts/k4batch_stage.py unique-component unique-component.json k4batch_draft_graph_component.py --graphs
python3 scripts/k4batch_stage.py unique-service unique-service.json k4batch_draft_graph_service.py --graphs
python3 scripts/k4batch_stage.py unique-life unique-life.json k4batch_draft_graph_lifecycle.py --graphs
python3 scripts/k4batch_stage.py unique-target unique-target.json k4batch_target_breakdown.py --graphs
python3 scripts/k4batch_draft_graph_export.py results/raw/k4-draft-query-graph results/k4/2026-10-24-draft-query-graph.json
```

导出器验证最终作业、各冻结 manifest、正式运行源码/归属文件、所有配对样本、无capture/JIT、数值/生命周期与资源审计。原始作业和工作区保留；不要覆盖旧快照或用同名不同定义重排队。

下一步应按 Target 分解继续核对实际 kernel/内存行为，再选择 MoE、投影及 HC 的具体优化；已有联合预算须在新后端上重新校准并独立验收。四请求完整链路、随机采样、更广长度/到达率和完全自主 kernel/加载器仍未完成。
