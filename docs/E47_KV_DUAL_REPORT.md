# E47：压缩KV与双256K状态

2026-10-25。**两个独立256K target-only请求的容量、Graph和状态隔离通过。**
每请求KV＋QSA索引增长项从BF16约6.94GiB降至INT8约4.13GiB，减少40.54%；
双请求研究过程峰值CUDA allocated **82.63GiB**，没有提高原88GiB分配限额。
各自三个检索键均答对，输出和终态与同格式、同前缀的独立续写一致。

**原始精度门槛未过，默认不晋升。** 原生prefill优先还造成首请求约225.8秒的输出空窗。
这轮证明两份长状态能够共存和正确执行，尚非平滑并发服务、多请求DFlash或GPU batch。

[事前方案与修订](E47_KV_DUAL_PLAN.md) · [脱敏证据](../results/e47/2026-10-25-kv-dual-256k.json) ·
[显式接口](E47_SHARED_GRAPH_USAGE.md)

## 实现与依赖

量化对象为12个target attention层的K/V；模型权重、激活、QSA原始/池化index、GDN/PLE
沿用原配置。INT8复用TensorFold的H32、32元素组、FP16 absmax缩放；FP8复用e4m3行缩放。
没有新增量化算法、修改上游算子、下载或重载权重，来源保留于[第三方说明](../THIRD_PARTY_NOTICES.md)。

`LongTarget`、`optimized_target`与`TensorFoldPrefillProvider`新增显式`kv_dtype`，已有API默认BF16。
prefill与request格式必须相同；同格式state equality和prefix哈希包含INT8的K/V缩放张量。

`SharedGraphTarget`有两个稳定地址槽：每请求独立持有KV、scale、QSA、GDN/PLE状态及Graph，
两槽共享同一个eager Buffers集合、screened head和owned prefill工作区。所有设备工作串行且
有fence，下一请求才能覆盖公共scratch。它用原生DeepInfer Scheduler驱动，不修改K4联合
forward/DFlash工作区，也不移除adaptive DFlash原有准入限制。

## 格式内精确性与精度筛查

INT8、FP8分别比较native rows8与owned chunk1024：English64、boundary2049，完整末行
logits、五层特征和有效状态（含scale）精确一致。每格式每前缀16个width/parity桶，
两格式共64个Graph验证窗口；奇数完整提交、偶数半提交，无fallback。

精度筛查固定8个英文/中文/算术/JSON/Python表达式/解释/长输入检索任务，三格式各一次
greedy输出，再在相同BF16 token前缀上比较最多8个完整词表输出位置，共54行/格式。
事前门槛：不丢失BF16已通过的客观fixture、平均KL<=0.02、raw logit相对L2<=3%、
teacher-forced top1一致率>=95%。

| 格式 | 相同前缀top1 | 平均KL | 平均raw logit相对L2 | 整段greedy与BF16完全一致 | 原门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16 | 54/54 | 0 | 0 | 8/8 | 对照 |
| INT8 | 54/54 | 0.00037965 | **6.55%** | 8/8 | 未通过L2 |
| FP8 | 54/54 | 0.00031438 | **8.42%** | 7/8 | 未通过L2 |

两格式都未丢失BF16通过的客观fixture。FP8在这组平均KL略低，INT8的raw L2较小、整段
greedy一致性更好；没有“所有指标INT8均优”的结论。
pilot-v1保留`STOP_E47_QUALITY`，没有改阈值或将其重标为通过。后续选择INT8仅做容量
研究，`quality_promoted=False`，不宣称整体BF16等效或开放随机采样质量保证。

## 两个真实长请求

两个请求同时准入，每个输入262080 tokens、输出上限64，logical262144/physical262160。
两份中英文合成ledger含不同的开头/中段/末尾registry key。A输入与E46一致并核验token SHA，
B更换三个key，输入SHA不同。生成分别为：

| 请求 | 输出tokens（含EOS） | 三个key | 检索 | 与同格式独立续写 |
| --- | ---: | --- | --- | --- |
| A | 24 | MAPLE-7319 / RIVER-4826 / CEDAR-9053 | 3/3 | tokens、状态、cache tail精确 |
| B | 27 | BIRCH-6284 / CREEK-1579 / SPRUCE-8042 | 3/3 | tokens、状态、cache tail精确 |

两份KV/scale/index/pool与GDN rec/conv存储地址不重叠；生成结果均不包含另一请求的key。
每份256K前缀各检查16个Graph桶，合计32桶、144行完整logits；features、部分提交状态及
scale精确。实际tiled QSA selector在各请求完成prefill的当刻读取，分别抽取三个尾部query，
与CPU稳定topK及低block ID平分规则一致。完整已提交前缀的KV/scale/index/pool哈希前后相同。

32个capture、81次实际replay（32个诊断桶＋49行decode），Graph/head fallback均0。
这是两个准入状态轮流执行，`gpu_batch=False`、target-only；没有草稿接受率或多请求DFlash收益。
短fixture的waiting/prepared-prompt/prepared-decode取消和同ID/槽位复用另行通过。

## 容量与调度结果

| 项目 | 观察值 |
| --- | ---: |
| INT8每请求context-growing cache/index bytes | 4,429,455,360，约4.13GiB |
| BF16同容量增长项（单独control实际布局） | 7,449,538,560，约6.94GiB |
| 增长项减少 | **40.54%** |
| 双请求峰值CUDA allocated | **88,721,504,768 bytes，82.63GiB** |
| 最低系统可用内存 | **15,157,673,984 bytes，14.12GiB** |
| A首输出 / 最后输出（active clock） | 221.16 / 449.39秒 |
| B首输出 / 最后输出（active clock） | 446.84 / 449.57秒 |
| A最大相邻输出事件间隔 | **225.80秒** |
| B最大相邻输出事件间隔 | 0.117秒 |
| 两请求active wall | 449.575秒 |

cache/index增长项包含scale及BF16 QSA index/pools，不含权重、GDN、Graph、prefill或其他
工作区；不能把40.54%当作整个进程内存下降。模型与原有resident对象仍常驻，未卸载草稿凑容量。

研究在每个首token后暂停检查Graph、state与独立续写，并在释放前校验prefix哈希。
这些诊断共15.869秒，显式从active clock排除；实际原始wall465.444秒。
此表不是未中断HTTP测试，也不是与BF16双请求或Mia原生引擎的配对性能对照。

原生Scheduler先推进A的长prefill，再优先完成B的prefill，A虽然已发布首token，后续decode
仍等待约226秒。容量问题已缓解，**prefill/decode公平调度成为当前明确的服务障碍**。
下一阶段应保留现有算术/缓存证明，控制prefill连续执行量并记录首token后最长停顿；
不能仅凭两份KV放得下就宣称达到实用并发吞吐。

## 完整256K精度对照

独立补充一对相同A输入的完整256K INT8/BF16前向，在相同24个前缀位置（含prefill末行）
比较完整词表。INT8复算输出SHA与容量阶段一致；BF16的24个输出token也完全相同，检索3/3。

| 指标 | INT8相对BF16 |
| --- | ---: |
| 相同前缀top1一致 | 24/24 |
| 平均KL（BF16→INT8，FP64计算） | 0.00038304 |
| raw logit相对L2均值 | **5.15%** |
| 中心化logit相对L2均值 | **7.94%** |

raw指标为`||b-a||/||a||`；中心化指标的分母改为`||a-mean(a)||`，两百分比不能当作同一
分母的误差能量相减。去掉公共平移后仍有剩余差异，不能简单归因于常数偏移。
最高分token稳定、KL较小，与全词表数值精度门槛未过可以同时成立。
这是一份合成检索长文的结果，尚无自然长文、多跳推理或随机采样质量结论。

## 失败记录、补充验证与审计

- job490：pilot正常完成但精度门槛未过，原结论保留。
- job492/study-v1：A的最后960个prompt tokens与B的前64个tokens同计划执行，B已覆盖共享
  scratch；复用E46固定尾tile的观测器读错请求而失败。job493独立清理审计通过。
- job494/study-v2：修复为每请求prefill完成时观察，并支持不同尾tile长度。两份完整长请求、
  Graph/scale/隔离主体均已通过并保存。随后截断聊天fixture提前EOS，取消脚本未检查active
  就prepare/cancel，触发stale handle及unfinished plan。job495独立清理审计通过。
- job496/followup-v1：保持job494为FAIL，核验其已完成容量子阶段的源码/结果链；用有效完整
  聊天修复短生命周期测试。没有重复双请求长跑。因中断时CPU logit帧未保存，另做上述单请求
  精度pair，并通过输出SHA确认与原容量样本一致。
- job498：针对成功followup独立审计。所有阶段复用worker46235/load_count1，原算子/no_grad、
  native group/env、实验模块和GPU活跃分配恢复：**77,393,408,000→77,393,408,000 bytes**。

归档不篡改两个失败job；导出器分别验证失败源码、完成容量子阶段、补充验证及审计。
E43更早未解释的32MiB差额仍单独保留，不由本轮稳定基线覆盖。

原始归档595文件、1,725,483bytes，SHA256
`a89af2d3bc0c6ea5d7a31f984d90121fc532951f809eab23f5bd2f34274529cd`，
位于忽略目录`results/raw/e47/2026-10-25-final/`，不含模型权重。

```bash
python3 scripts/e47_export.py \
  --raw results/raw/e47/2026-10-25-final/worker \
  --out results/e47/2026-10-25-kv-dual-256k.json \
  --pilot-version pilot-v1 --pilot-job 490-e47-pilot-v1.json \
  --study-version study-v2 --study-job 494-e47-study-v2.json \
  --followup-version followup-v1 --followup-job 496-e47-followup-v1.json \
  --audit-version audit-v1 --audit-job 498-e47-audit-v1.json
python3 -m unittest discover -s tests -q
```

本地368项测试：349通过、19项可选依赖跳过。范围是显式INT8容量研究和有限BF16/FP8精度
筛查；没有默认精度晋升、HTTP/实用SLO通过、GPU联合batch、多请求DFlash、分页KV或1M支持。
