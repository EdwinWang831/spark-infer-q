# E33 全优化配置迁入原生动态运行时

2026-10-23，`PASS_K3E33_INTEGRATED`。**完整迁入并启用Graph后的新运行时为 47.4588 token/s；同期E33优化研究路径为 48.3779 token/s，新运行时低 1.900%。** 接入和正确性通过，性能不是新增收益，不切换默认路径。[方案](K3_E33_RUNTIME_PLAN.md)、[入口](K3_E33_RUNTIME_USAGE.md)、[脱敏证据](../results/k3-e33/2026-10-23-optimized-runtime.json)。

## 这次完整接上什么

| 部分 | 两组相同配置 |
| --- | --- |
| Target | 原48层、五处DFlash特征、最终writeback/mixer；2048物理容量 |
| 输出头 | E33融合FP64定向界＋逐通道INT8辅助筛选，候选用原BF16 BK64/SK1精算；不满足条件回退完整原头 |
| Graph | 1～8行×GDN两奇偶，共16张backbone图；筛选头另外预备8种宽度 |
| 草稿 | E28 W8A16 MLP gate/up/down，Q4草稿输出头，BF16 attention/selector |
| 预测与预算 | E17 NumPy float64线性模型＋E29 W8冻结short/long费用，当前草稿后按单请求MTPVarlen目标函数选宽度 |

研究代码来源为`2cb909b`及其冻结E33源/输入seal，迁入时只调整模块位置/运行时接口，保留筛选算术和W8打包格式。模型SHA `50294789736af0b0f527c63e68f3914ddedc52592d476d9a397fcd29f3b5c873`，cost SHA `e08a9474dcd01273779461b91a0f89e1f9065accf95095f7682f8e9b3b774971`，W8二进制SHA `1494e084fd6e901b3b850b39e0e02be93f06d47c771fa6f9c1e43b245909b6ec`；运行前强制校验。原BF16头和共享权重保留，辅助头payload仍为640,665,600 bytes，另有W8包，不称整机权重内存节省。

旧K3的44.78来自原BF16头/BF16草稿/Python预测器和三个短任务。这次的47.46来自完整E33配置及其八个任务。E33本来就包含Graph，不能把旧K3的Graph比例再次乘到48.51上。同期研究路径48.38与历史48.51是不同轮次实测值；下面的开销只按同期对照计算。

## 原生运行时接口

筛选Target显式提供`run_predictions`返回经过原头精算/回退确认的greedy IDs、DFlash特征和commit buffers；hidden或候选logits不会冒充完整logits。诊断时另算完整原头，检查全argmax、子集值、committed logits/features；计时路径不执行这些诊断。完整logit调用方仍能明确走原完整头。

完整profile将草稿生成放在本轮decode prepare，首token先输出，再产生第一轮草稿和预算。原生NEXTN在该同步路径只承担最大候选容量与轮次占位，真实草稿来自当前provider；公开ZOS hook写入最终q，再按`process_MTPInfo`→`process_outputs`完成原生账本。没有改只读metadata或手工跳过schedule/process计数。E17概率/上下文成本可用时才做本轮决策，不使用当前验证结果选宽度。

固定草稿模式仍已删除。CLI完整配置为`--profile e33 --execution graph --mode adaptive --capacity 1536`；默认仍是baseline profile。Graph workspace独占/reset/lease、EOS/长度、取消/复用和输出事件保留；没有DeepSeek GPU sampler/MTPInfoPool、随机采样或联合GPU batch。

## 同任务、同轨迹的正式计时

E33同一`e28_test_tasks.json`八个工程root、自然EOS/原output_cap，每root每模式四次、AB/BA交替，共64次计时。每组实际decode token为 **8,588**，按总token/总墙钟计算，不平均各任务TPS。

| 配置 | decode token/s | 完整请求耗时之和 | 首token准备耗时之和 |
| --- | ---: | ---: | ---: |
| legacy | 48.3779 | 265.816 s | 88.297 s |
| native | 47.4588 | 270.566 s | 89.609 s |

上述时间列各为32个请求之和。新运行时decode变化 **-1.900%**，8-root bootstrap95%区间 **-3.176%～-1.057%**；四轮整体变化依次为 **-2.846% / -1.115% / -0.901% / -2.720%**。该受控集合上新运行时存在可重复的净接入开销，不是性能晋升。不能把它泛化为任意负载的固定开销，也没有按分段timer将差额全部归给某个CPU/GPU函数。

| 工作量（每组全部正式计时） | legacy | native |
| --- | ---: | ---: |
| 验证轮次 | 2596 | 2596 |
| 验证行数 | 12748 | 12748 |
| 输出decode token | 8588 | 8588 |
| 观测接受/观测草稿 | 6012/7784 | 6012/7784 |
| 筛选头自然回退 | 0 | 0 |

两组候选并集均为1～116，按轮次中位数11。所有64次运行的完整`context/rates/width/emitted/accepted/observed`轨迹均与各自预热一致、两组一致，且与原E33历史逐步记录完全一致；输入/输出hash也相同。因此差额不是少接辅助头、换预测器、换任务、改变接受率或多验证了行。

| 类别/语言 | prompt token | completion token | legacy TPS | native TPS | native变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| long_context/zh | 1024 | 222 | 69.241 | 64.753 | -6.481% |
| multi_turn/zh | 265 | 384 | 35.653 | 35.479 | -0.487% |
| multi_turn/en | 213 | 384 | 50.608 | 49.426 | -2.337% |
| math/en | 81 | 258 | 67.456 | 66.487 | -1.438% |
| external_qa/en | 27 | 384 | 40.153 | 39.346 | -2.010% |
| external_qa/zh | 40 | 133 | 35.699 | 35.420 | -0.781% |
| code/en | 74 | 238 | 58.986 | 58.056 | -1.577% |
| long_context/en | 768 | 152 | 83.249 | 80.592 | -3.191% |

## 计时口径与准备

两组每次都重新做eager prefill，再从首token准备完成后的同一decode边界计时，**包含第一轮草稿**、F1/NumPy预测、预算、Graph验证、筛选/精算/回退、所有同步、commit和循环；native还包含原生schedule/账本/事件与结束回收。首草稿没有前移进TTFT，也没有事后加减补时。

模型加载、W8/辅助头打包、Graph捕获、每任务正确性预热、初态/参考构造和正确性核对在decode timer外。两组Graph始终16张，筛选cache固定8种宽度；正式计时无新增capture或候选子集JIT。新输入在未预热的subset count上仍可能JIT，这不是冷请求延迟或普遍服务速度承诺。

完整请求timer从已分词token的提交/参考prefill前到完成，包含准备首token和decode；不含tokenization/HTTP传输。研究对照以相同eager prefill/Graph bind提供首token准备边界，不是另一个HTTP服务。正式对照使用token API、不做逐事件文本解码；真实SSE另做正确性测试，不能把这里的TPS直接当网络并发吞吐。

辅助头保留原E33有限范围/误差界和cap2048，随机采样不使用该接口。Graph及TensorFold完成fence、CPU greedy/特征读取仍保留，未完成全异步/主机同步精简。测到的提示27～1024 token、输出133～384 token，逻辑1536/物理2048；无4K+、多请求batch、独立HF数值证明或最新TensorFold排名。

## 正确性与资源

- 本地210项：198通过、12因依赖缺失跳过。新增吞吐汇总验证总token/总时间、首token扣除和缺失/重复run/root拒绝；K1/K2/K3回归通过。
- 原生CPU四组deferred-draft/首中末拒绝及预测ID接口检查通过；prefill阶段不出草稿，真实q/提交与原生轮次一致。
- 迁入头11项功能检查：1～8行、强制cap回退、全零并列和NaN回退；全argmax/自然子集原BF16值精确一致。
- 完整native profile GPU **13组、511行完整committed logits、511行DFlash特征、116次有效state比较**精确一致。覆盖三条自然输入/旧E33预算，四组oracle-only拒绝/全接受、长度1、三阶段取消/复用、真实SSE输出/取消。oracle草稿不进入速度结果。
- 正式八root预热另检查 **4294行完整committed logits、4294行DFlash特征**，native prefill核对2492行特征；两组预热/计时共 **80次最终live state**精确一致。
- 同worker/load_count=1，未重载/停止模型或覆盖冻结研究源。收尾group/env恢复、所有lease释放，gc后allocated回到77,594,734,592 bytes，最终READY，未调用empty_cache。

首轮GPU验证脚本误读inference tensor的版本计数，在推理验证前报错；保留FAIL记录，改用显式storage/sample检查并修正分块prompt的测试驱动后，以新封存GPU v2全部通过。这个错误不是模型精度或推理失败，不能用FAIL掩盖性能结果；正式64次全部PASS，速度较慢的结论照实保留。

## 复现与证据

```bash
python3 scripts/k3e33_export.py results/raw/k3-e33 results/k3-e33/2026-10-23-optimized-runtime.json
python3 -m unittest discover -s tests -q
```

原始归档SHA256 `4daa1716d7dd137b993aa7f431c2b25077dd10c5afa7ee12530bd2345b224d36`，包含CPU/GPU失败与修正/正式64次/audit及原E33记录和seal。公开导出重新计算加权TPS/bootstrap，逐轮核对当前与历史E33轨迹及最终源码身份；保留逐run标量计时、工作量/预算SHA、内核/输入/源SHA，完整逐步记录保存在Git忽略的raw归档。不包含提示词、权重payload、私有绝对路径或对象ID。独立CLI没有为验证再加载大模型；真实HTTP复用resident profile。
