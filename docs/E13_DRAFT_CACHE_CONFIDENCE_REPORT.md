# E13：1K–2K 输入的草稿缓存有 decode 收益；接受率信号值得在线验证

2026-10-21。[预先冻结的方案与形状修正](E13_DRAFT_CACHE_CONFIDENCE_PLAN.md) · [缓存用法](DRAFT_CONTEXT_CACHE_USAGE.md)。

## 实际生成结果

Target、DFlash 和完整词表Q4草稿头始终使用同一常驻worker的原权重，**没有重新加载或下载**。缓存wrapper只新增历史FC/K/V缓冲；Target路由、算子及量化不变，预算仍是原生DeepInfer MTPVarlen，接受率新规则尚未接入在线。

| 实际输入条件 | 原路径 decode | 缓存 decode | 变化 |
| --- | ---: | ---: | ---: |
| 12个新任务，prompt38–413 token | 41.1417 tok/s | 41.0158 tok/s | **−0.31%，未证明收益** |
| 1024/1536/1920-token输入，三个长度案例合计 | 39.7316 tok/s | 42.2219 tok/s | **+6.27%** |

吞吐为相同输出token数除以各任务decode耗时中位数之和。每个请求最多生成96 token，遇EOS提前结束；不是所有任务完整自然结束，更不是自由长文质量评测。decode计入首次缓存构建、逐轮追加、draft/verify/commit及Python循环，**不含prompt、状态复制、网络或oracle**。不能称为完整请求/TTFT提速，也不能与此前不同任务的45 tok/s直接排名。

短输入正式96次计时、24次另行逐committed-logit oracle通过；长输入24次计时、6次oracle通过。全部120次正式结果的输出IDs与最终live state对照serial通过。长输入最终committed位置1119/1631/2015。原与缓存均共享同一个已处理prompt状态；长prompt本身没有逐token完整重做prefill oracle。

### 千token案例

| prompt长度 | decode因子（原耗时/缓存耗时） | 吞吐变化 |
| --- | ---: | ---: |
| 1024 | 1.01555 | +1.55% |
| 1536 | 1.07888 | +7.89% |
| 1920 | 1.09421 | +9.42% |

四轮合计配对decode因子为1.05463、1.08367、1.05388、1.06614，均改善。按全部配对计时计算的耗时降幅为6.06%，三组长度条件bootstrap区间约2.04%–7.99%。**三个案例共享合成重复背景模板**；该区间只描述这些条件，不代表三个独立语义任务的泛化保证。仍需真实长文任务、多轮上下文、更多重复及4K以上验证。

缓存带来的成本变化也改变了预算和执行工作：24次长输入计时共提交1140行，原路径验证2034行、缓存验证2014行；draft主机阶段累计6.192→4.814秒，verify22.020→21.663秒。因此+6.27%是这个组合执行器的decode变化，不能全归为同工作量的单kernel效率。

短输入配对耗时改善点估计−0.317%，按prompt bootstrap区间约−1.973%–+1.059%。低于256的请求走原路径，额外wrapper与运行波动仍可能造成小差异。只保留opt-in，不升级全局默认。

[短输入数据](../results/e13/2026-10-21-cache-integrated.json) · [1K–2K数据](../results/e13/2026-10-21-long-cache-integrated.json)。

## 数值与状态：初版失败及修正均保留

初版64→65行追加，投影组件relative L2最高约0.431%，hidden差异2.013%，超过预设1%门槛。该例草稿路径虽相同，仍判失败。合成形状诊断显示Torch BF16 GEMM会随行数改变舍入，旧前缀在64→65时也可能改变，不能直接假设每一行数值独立。

未放宽门槛。修正为context<256回原路径、≥256才缓存，追加不足4行pad到4行。修正后15组64/256/384/512/768及+1/+4检查、9组1024/1536/1984及+1/+4检查中，FC、五层K/V、hidden与草稿路径均bit exact。**仅覆盖这些点，不能宣称任意长度的DFlash hidden普遍bit exact。**

另在2000行Target前缀后，对8行teacher-forced执行核对全部logits/features，并分别commit1、4、8行，与serial live state一致。实际生成对照进一步核对greedy提交状态。4K及以上还需要DFlash2048滑窗和绝对位置处理、Target容量与状态验证；本轮最多2K，不外推到8K/32K。

缓存1024容量约25.12 MiB，2048容量约50.12 MiB，不包括Target状态。worker生命周期的累计CUDA peak allocated约73.891 GiB、reserved74.281 GiB；不是只属于本实验单job的峰值。

[微基准、失败与状态数据](../results/e13/2026-10-21-cache-probes.json)。完整微基准区分cold build、纯reuse和append4，不能把纯reuse因子当生成收益。

## 接受率信号：预测改善，尚无在线策略速度结论

从已经算出的selector分数提取选中路径的top1-top2边差，作为主要proxy。使用6个新的校准prompt确定各depth的median分桶和Beta(1,1)平滑，6个held-out prompt仅评估。首次拒绝之后、EOS/cap之后的条目不补失败标签；matched EOS计成功。

| 预测方法 | held-out Brier（越低越好） | log-loss |
| --- | ---: | ---: |
| depth-only平滑先验 | 0.19316 | 0.57806 |
| depth+edge-margin二分桶 | **0.15716** | **0.46734** |
| 仅随过去step更新的depth先验 | 0.18773 | 0.57297 |

487个held-out条件观察标签中，margin模型相对固定depth先验Brier下降18.64%，按prompt bootstrap区间9.91%–30.01%；相对在线更新的depth先验下降16.28%，区间10.02%–26.35%。这是**接受概率预测的误差变化，不是接受率或吞吐提高18.64%**。分数是16候选内的proxy，不是真实全词表概率。

这轮depth1–5分别观察196/129/71/50/41个测试标签，depth6/7没有校准数据，不能据此升级更大草稿预算。只在完整草稿之后可用；不得用这些信号冒充pre-draft输入，或在草稿已生成后退还draft费用。

路由/信号采集在单独的412-step观察执行中，额外draft核对、路由复制和CPU提取都不混入缓存正式计时。这轮观察的verify wall有40–659ms的大幅波动，二级成本回归held-out MAE约56–59ms，远不够稳定；即使加入首候选margin后出现小幅改善，也不作为有用成本预测或在线收益证据。采样期间整机有较高后台CPU/I/O负载，未停止其他服务，不能只凭一次观察把波动归因于某一组件。接受率标签和serial正确性仍有效。

[接受率及探索性成本数据](../results/e13/2026-10-21-confidence.json)。

## 交付与下一步

- `CachedContextDraft`为可选共享权重wrapper，记录新请求、追加、回退及窗口约束；默认draft未改。实际长输入结果支持继续扩大覆盖，短输入结果不支持全局启用。
- 下一项优先做独立的在线confidence预算A/B：先保留原成本曲线，只换已校准深度范围内的收益预测，计入额外信号提取/同步费用，并保持greedy/logits/state oracle。不把新专家成本模型同时叠加。
- 4K以上先解决滑窗和状态语义，再测吞吐；不为了长度数字直接解除边界。
- CPU回归运行98项，96通过、2跳过。worker仍为同一PID、加载计数1；初版cache gate失败后也没有重载模型。任务结束保持READY。

原始12任务采集与对照源可在提交`ee7a725`复现；长输入使用随后增加的可配置study与adapter。各公开数据包含实际源SHA256、分析器SHA256与raw摘要，私有原始文件保留在`results/raw/e13/`，不提交权重或私有环境路径。
