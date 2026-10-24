# E36：专用 prefill 数值诊断

2026-10-23。[首轮冻结方案](E36_PREFILL_NUMERICS_PLAN.md) · [第0层后续诊断方案](E36_LAYER0_LOCALIZATION_PLAN.md) · [公开证据](../results/e36/2026-10-23-prefill-numerics.json)

**已找到并验证保留现有算术的大块 prefill 候选：七条前缀的完整数值门槛通过，prompt 本体吞吐从110.867提升至889.218 token/s（8.0206倍）。** 原生未修正的prefill仍存在分歧。候选尚未接入默认引擎，没有完整FP32整模参考、任务质量结论、4K支持、完整DFlash请求或新的decode token/s。

## 输入和对照边界

复用同一常驻完整 target、存储权重、BF16 KV 和 tokenizer。固定五条 teacher-forced 前缀：历史 `[100,101,102,103]`、英文64 token、中文64 token、英文代码512 token、中英资料512 token。后四条由冻结文本重复并截取，是受控合成前缀，不代表自然完成任务。

三条路径：逐token serial、现有 decode 算术 rows8、原生 `prefill=True` 整段。历史4 token和两条64 token上，serial/rows8 的全部48层写回残差、路由 picks、末行完整 logits 与最终有效 GDN/QSA/PLE 状态均精确一致。两条512 token仅以rows8作数值控制，未另跑逐token oracle。

观测使用实例私有 forward clone，逐层应用 pending writeback 后再取残差，不读取未完成的 `b.h`；所有行与所有层都比较，路由比较的是专家集合。未改另一工作区 K4 或原 frozen source。

## 完整轨迹的差异

| 前缀 | 长度 | 首个残差差异层 | 原生相对 rows8 的末行 logits L2 | 末行 greedy token 相同 |
| --- | ---: | ---: | ---: | --- |
| 历史合成回归 | 4 | 0 | 10.6814% | 是 |
| 英文 | 64 | 0 | 38.7185% | 否 |
| 中文 | 64 | 0 | 6.7187% | 是 |
| 英文代码 | 512 | 0 | 9.7301% | 是 |
| 中英资料 | 512 | 0 | 6.7435% | 是 |

历史10.68%差异得到复现。一个末行token改变足以说明原生路径不能直接当作现有精确同路径 oracle 的替代品，但这些数值**不能单独说明哪条路径更接近量化模型真值，也不是回答质量损失百分比**。

## 相同输入的局部 FP32 / FP64 控制

每条前缀取layer0和layer6，共10组；所有输入张量和初态都有SHA256。TF32关闭、FP32 matmul precision为highest。4项参考方程/切块/分组/输入不变测试在远端实际Torch环境的CPU上全部通过，无跳过。

**MXFP8 投影：**两种内核都读取相同BF16 attention输入，参考将存储e4m3权重和e8m0指数还原为FP32，再用PyTorch FP32矩阵乘。serial/prefill 相对参考的L2分别都落在约 **0.16096%～0.16703%**；二者之间为约 **0.00810%～0.01051%**。误差量级相近，未得到原生投影精度显著更差的证据。存储格式解码仍共享现有实现，不是完全独立格式解码器。

**GDN recurrence：**固定同一个真实投影P、零conv历史、零初始state。共同front的k/v/decay/beta与decode scratch精确相同；decode未暴露q，只能根据源码确认相同计算路径，不宣称q也经过独立逐位读回。参考在这些已有front输出上逐token计算，不重新验证front或其投影精度。

两条路径最终state相对FP64 recurrence的L2均为约 **4.39×10⁻⁸～6.72×10⁻⁷**，在不同样本上各有高低。不能把微小的局部排名扩大为整模精度结论。输出侧参考还经过BF16 readout和原有back，不能叫完整gated FP32参考；decode raw y未暴露。

这两组控制说明：单独看到投影路径不同或GDN使用不同FMA/归约，就把10%～39%的末行差异全部归因于其中一个，不符合现有证据。第0层继续分解的结果见下节。

## 第0层因子定位

此阶段是在看到上述结果后追加的诊断，复用相同前缀，不是独立测试。只执行layer0，逐项记录HC读出、投影P、GDN输出、输出投影branch、MoE输入、router/picks/weights、专家slots和写回残差。

原生prefill下完整穷举三个因子的八种组合：A把私有forward的MXFP8投影换成decode算术；B换成decode GDN chain（完整长度scratch、正确parity和conv推进）；C使用F32 MoE slots。C会同时切换shared expert路径，是联动策略，不能解释成纯粹只改slot舍入。

| 前缀 | 原生层0残差L2 | 仅A | A+C | A+B+C |
| --- | ---: | ---: | ---: | ---: |
| 历史4 token | 0.18935% | 0.12637% | 0% | 0% |
| 英文64 | 0.25968% | 0.11773% | 0.03202% | 0% |
| 中文64 | 0.25240% | 0.11719% | 0.03500% | 0% |
| 代码512 | 0.35642% | 0.11057% | 0.02529% | 0% |
| 资料512 | 0.25104% | 0.10959% | 0.03970% | 0% |

每条原生mask0均复现上阶段第0层误差。只切A，五条前缀的专家集合就全部与rows8一致；只切B或C仍未消除差异。三项组合后，五条前缀采集的全部12个组件（包括投影、router、专家slots、已写回残差、最终单层rec/conv）均精确一致。这支持“投影舍入差异可经路由放大、另有递归与MoE联动精度边界”的定位；不把它简化成某个内核算错。

这项消融本身没有执行其余47层，也不验证完整prefill→decode交接或DFlash特征。因此随后追加完整模型门槛；默认切换仍需引擎接入和完整请求测量。

## 完整模型候选与 prompt 计时

根据第0层结果追加[完整模型门槛](E36_PREFILL_CANDIDATE_PLAN.md)。候选通过私有forward保留A/B/C算术，沿用原生大行块、attention和PLE执行；未新增CUDA内核，底层仍复用TensorFold。除了上述五例，新增此前未执行的英文JSON128 token、中文技术说明384 token。

**七例全部通过**：末行完整logits、五层全部prompt特征、最终有效state精确一致；每例从各自prefill状态用原decode再执行8步，共56步，完整logits、特征、有效state逐步一致。合计核对63行完整logits、1724行DFlash特征、63次有效state。状态门槛在计时外执行，计时样本另核对末logits/全特征hash及预测token，不宣称每次计时都重新跑state oracle。

之后每例每路径预热一次，四轮AB/BA交替，共56次prompt计时；每组6672 prompt token。输入已分词，缓冲/JIT准备在外，计入State初始化、stage、全部48层、完整五层特征收集、commit、末token准备及同步。两路径均为eager prompt，已有decode Graph不在本轮切换；现有Graph runtime的prompt仍委托eager step。

| 每例四次平均 prompt wall | rows8 | 候选 | 比值 |
| --- | ---: | ---: | ---: |
| 历史4 | 62.48 ms | 61.91 ms | 1.01× |
| 英文64 | 554.64 ms | 160.54 ms | 3.45× |
| 中文64 | 558.47 ms | 155.87 ms | 3.58× |
| 代码512 | 4609.33 ms | 433.51 ms | 10.63× |
| 资料512 | 4751.73 ms | 472.08 ms | 10.07× |
| 新JSON128 | 1091.23 ms | 211.53 ms | 5.16× |
| 新中文384 | 3417.17 ms | 380.36 ms | 8.98× |

按总token/总墙钟汇总：**60.1801→7.5032秒，110.867→889.218 prompt token/s，8.0206倍，耗时减少87.5321%**。四轮汇总比值8.0901 / 8.0777 / 7.8790 / 8.0406。最长两例主导加权结果，4 token例基本不变；七条合成前缀和重复计时不足以代表所有真实请求，也不据此声称统计泛化区间。

收益来自完整执行组织：一次处理整段，减少跨8行块的重复调用/提交；现有rows8还逐块计算全部行的词表head，而专用prefill只计算最终所需行。数值控制补上了大块执行与既有decode算术之间的边界。这里没有拆分各因素的性能归因，不能叫“某个内核快了8倍”。baseline同样收集完整特征，但采用本实验harness的列表收集后拼接，不包含引擎逐块追加context等其他开销。

这是prompt-only token API。HTTP、tokenization、DFlash草稿与投影缓存建立、排队、完整请求和多请求混合未测速；capacity1024、最长prompt512+8续步，未进入大于2048位置的稀疏QSA选择或滑窗。候选仍需由引擎主线接入，再验证实际请求、调度分块和更长上下文；不把这轮收益直接乘到既有decode成绩上。

## 证据、收尾与复现

环境/输入preflight、完整轨迹与局部参考、第0层消融、完整模型候选/条件计时、收尾audit分别封存并经共享resident串行队列执行。最初排队的241在执行前审查时撤回，用新的reference_kind/boundary区分BF16投影输入与混合front输入；旧stage和取消记录保留，未执行第二次GPU诊断。

```bash
python3 scripts/e36_export.py \
  --raw results/raw/e36/final/worker \
  --out results/e36/2026-10-23-prefill-numerics.json
python3 -m unittest discover -s tests -q
```

公开证据保留逐层指标、局部参考、消融、源/内核指纹和输入hash，移除私有路径、模型对象ID与完整token输入。原始封存源、逐作业结果和外部参考源保留在Git忽略的raw归档中。

五个实际作业240/242/244/246/248全部PASS，模型对象与worker一致、load_count=1，没有重载、停止或调用empty_cache。最终独立audit验证原forward恢复、冻结依赖未变、Gloo未泄漏；GC后allocated为 **77,695,397,888 bytes**，与E36 preflight前相同。共享队列中，诊断开始前还记录过78,264,023,040 bytes；不把这一下降当作模型压缩收益。该收尾数值也不是本轮峰值显存。

原始归档 **195文件、541,884 bytes**，SHA256 `150ab4b6e628464e1c482036b03a0c821f3f5af478dc9ba156dd6a94df2110cf`。全部执行数值源与当前checkout哈希一致；导出器在audit后仅更新共享队列状态处理，以允许另一工作区立即启动新任务，公开记录其单独SHA。归档时worker已在执行其他工作区作业，公开证据如实记录RUNNING，不冒称全局空闲或干预该任务。

本地 **250项测试，234通过、16因可选依赖缺失跳过**；局部参考的4项方程测试已另外在实际Torch环境CPU上通过。导出器重新核对完整输入/源码/wrapper身份、局部参考契约、40种消融、完整模型门槛、56条计时、相同工作量/输出hash、资源回收；公开JSON通过私有路径、对象ID与常见凭据模式扫描，raw保持Git忽略。
