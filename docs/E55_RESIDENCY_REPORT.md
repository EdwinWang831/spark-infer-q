# E55：常驻内存盘点与 INT4 KV 验证

2026-10-27。本轮推进八并发、每请求256K的容量目标，保留原生动态预算、DFlash和
E54每请求草稿投影缓存。**INT4显著减少状态分配，但当前方案仍超过88GiB限额；
短质量门槛也未通过，因此不提升为默认格式。**

[事前方案](E55_RESIDENCY_PLAN.md) · [用法](E55_INT4_KV_USAGE.md) · [可校验结果](../results/e55/2026-10-27-residency.json)

## 常驻盘点决定了优化方向

按CUDA底层storage去重，并记录各根对象对同一storage的引用；切片、别名、循环
引用不会重复计费。矩阵对象即使可调用也继续遍历，只跳过真正的函数/模块/类型。
模型权重和session对象没有删除、迁移或重新加载。

| 项目 | 实测bytes | 解释 |
| --- | ---: | --- |
| target可达独立storage | 75,002,047,104 | 69.85110GiB，与loader元数据一致 |
| 基线全部根对象去重storage | 77,249,378,284 | 1722份storage |
| 基线allocated | 77,393,408,000 | 与去重storage差144,029,716，不能视作闲置 |
| legacy target独占 | 93,385,580 | 约89MiB，无法填平八路缺口 |
| 原draft独占，优化provider创建前 | 1,756,633,600 | 多个根共享的权重不计独占 |
| 原draft独占，优化provider创建后 | 589,824,000 | 优化provider也引用了部分原始权重 |
| 优化draft新增独占storage | 347,985,920 | 新量化权重/工作区等 |

即使拿掉全部非target基线allocated，也只涉及2.22713GiB；保留target加八份FP8
环保守状态分配本身已达97.90761GiB，尚未计草稿/head/scratch。因此只清理公共
区无法完成当前八路目标。本轮没有为小收益破坏session所有权，转向显式INT4 KV。

## 实现与GPU算术验证

为现有有界索引adapter传递INT4 bit宽度，沿用上游H32/group32/FP16-scale量化和
反量化算术；支持target与prefill的显式INT4选项。BF16默认不变，INT8环路径仍不
支持。完整KV/scale、BF16 pooled QSA历史和循环状态均保留，只有原始索引用2048行环。

三格式各13例，共39例真实GPU差分验证通过，包括奇数起点、回绕、1024行准备、
近256K位置及改变绝对位置后的Graph replay。检查query、KV及所有scale、raw尾块、
pool、scores、selector IDs，以及snapshot/restore/clone。

全模型INT4 full-index与ring在英文64及2049边界输入的prefill logits、features、
完整有效state逐位一致；宽度1–8、两种parity的32个Graph/部分提交窗口也一致。
这是同格式实现验证，不能替代INT4相对BF16的质量判断。

| 格式 | 每状态allocated bytes | 去重storage bytes | KV storage bytes |
| --- | ---: | ---: | ---: |
| BF16 | 6,885,385,728 | 6,885,371,964 | 6,442,844,208 |
| FP8 | 3,765,680,640 | 3,764,619,324 | 3,322,091,568 |
| INT4 | 2,256,673,792 | 2,254,577,676 | 1,812,049,920 |

三者均含6,291,456 bytes raw ring与201,338,880 bytes BF16 pool。INT4相对FP8
每状态少1,509,006,848 allocated bytes；八份少约11.24GiB。基于storage的八份
差为11.25069GiB。两种计量分别报告，不混用。

在E54完整公共区/Graph/feature/每请求50MiB草稿缓存及reserve的外推上，仅替换
实测状态allocation差，八路约从104.06降至92.82GiB，仍比88GiB多约4.82GiB。
这是容量外推，未分配或填满八路，也没有提高限额；不能把allocator slack加到
操作系统MemAvailable来扩大预算。现有Graph包装器仍最多两个slot。

## 原有短质量门槛：不通过

沿用E47八个中英文/客观任务、每格式54个BF16 teacher位置、最多64个greedy
输出。门槛保持：无BF16已通过目标丢失、mean KL≤0.02、raw relative L2≤0.03、
teacher top1≥0.95；没有因结果不利而放宽阈值。

| 格式 | mean KL | raw relative L2 | teacher top1 | greedy整例与BF16一致 | 门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16 | 0 | 0 | 54/54 | 8/8 | 通过 |
| FP8 | 0.00031527 | 8.41832% | 54/54 | 7/8 | L2失败 |
| INT4 | 0.00205827 | 11.23022% | 54/54 | 8/8 | L2失败 |

没有丢失BF16通过的客观目标。INT4这批短任务的完整输出一致是积极信号，但样本
很小，且logit门槛失败；不能据此声称接近BF16的普适质量或比FP8更优。

## 失败记录与复用规则

616 inventory-v1漏掉可调用矩阵，低估约3GB，作为已替代记录保留；618修正后
与loader target storage一致。624 long-v1已完成BF16动态512输出及同格式greedy
512参考且输出/有效state一致，随后cleanup错误访问SharedGraphTarget的`leased`
属性而失败。626 long-v2在BF16复用检查中误判错误类型，立即失败，未运行INT4。
两次FAIL均保留，没有改写为成功。

628 long-v3改用包装器的owner释放接口。复用624时检查精确已知异常、完整BF16
输出/state验收、常驻对象、baseline、全部库源码及teacher artifact SHA，故不重跑
已完成的BF16长prefill。INT4 greedy参考复用同一已填满前缀，恢复compact seed后
逐一核验完整committed prefix，没有重复prefill。

## 单256K长请求最终验收

BF16和INT4分别完成261,120输入、512动态输出，以及复用已填满前缀的512-token
纯greedy参考。两格式各自的动态/greedy全部输出及完整有效target状态逐位一致。
这验证了INT4的speculation/rollback/部分提交实现，没有把INT4输出变化误归为
调度或草稿错误；不要求两种不同KV格式的状态逐位相同。

INT4与BF16的512输出并不完全相同。两者均包含三项预埋ledger检索键
`MAPLE-7319`、`RIVER-4826`、`CEDAR-9053`。BF16同前缀teacher抽查8个位置，
INT4 top1为8/8、mean KL为0.00806163、raw relative L2为7.00483%、centered
relative L2为8.28611%。均值平移不能完全解释偏差；8位置检索/teacher结果无法
代表一般长上下文质量，原3%门槛也没有通过。

诊断wall包括全状态哈希和teacher暂停，本轮不据此报告纯decode速度、端到端
加速或八并发吞吐。

独立作业630核对原始operator代码/globals、未初始化分布式组、native环境、私有
模块释放及628报告SHA，结果PASS。worker仍为PID46235、load_count1，最终READY，
allocated精确恢复77,393,408,000 bytes。没有重载、下载、empty_cache、并行GPU
进程、提高88GiB限额或修改独立K4工作区。

单请求峰值allocated：BF16 **80.60177GiB**、INT4 **76.29035GiB**；峰值reserved分别
80.84766/76.58594GiB。step采样host available最低21.35355/25.39056GiB。
短屏的8/8输出一致没有延伸到这条长任务：两格式从第11个输出token开始分歧。
当前8个teacher位置只覆盖较早的一段，不能据其top1全部命中掩盖后续输出变化。

## 封存与复现

忽略归档`results/raw/e55/2026-10-27-final/`包含924文件、13,439,247压缩bytes，
SHA256：`e5bf28cf77cc33e89cdc957038e6706a7ae26d8fde3c071e4a7d4a7a1ff403cb`。
其中保留616/618/620/622/624/626/628/630全部源码、manifest、作业、报告，包含
BF16/INT4 teacher logits激活及外部算子源码/运行二进制，不包含模型权重。

最终导出`PASS_E55_EVIDENCE_CANDIDATE_NOT_PROMOTED`复核源码封存、常驻对象、
worker、外部算子、kernel覆盖、quality决策、长请求全输出/state/工作量、teacher
artifact及重新计算的logit指标、BF16失败作业复用链、独立回收和E53容量参考。
公共JSON不包含激活数组或本机路径。最终本地回归414项：395通过、19项可选依赖
跳过；E55真实归档的篡改检查已执行，覆盖虚假promotion、阈值、输出/state、
teacher指标、Graph回退、假八并发和host reserve。

现有匹配常驻进程按依赖顺序运行，已存在的作业不要重复提交：

```sh
python3 scripts/e55_stage.py inventory-v2 618-e55-inventory-v2.json --phase inventory
python3 scripts/e55_stage.py kernel-v1 620-e55-kernel-v1.json --phase kernel
python3 scripts/e55_stage.py quality-v1 622-e55-quality-v1.json --phase quality --study-job 620-e55-kernel-v1.json
```

首次完整长诊断不传`--reuse-bf16`；本轮因624仅cleanup失败而采用已验证的BF16
复用。新复现实验应使用新的版本/作业号，不应人为制造该错误：

```sh
python3 scripts/e55_stage.py long-v3 628-e55-long-v3.json --phase long --study-job 622-e55-quality-v1.json --reference-job 624-e55-long-v1.json --reuse-bf16
python3 scripts/e55_stage.py audit-v1 630-e55-audit-v1.json --phase audit --study-job 628-e55-long-v3.json
python3 scripts/e55_export.py --raw results/raw/e55/2026-10-27-final/worker --out results/e55/2026-10-27-residency.json
python3 -m unittest discover -s tests -v
```

下一阶段需同时解决剩余约4.82GiB容量缺口和KV精度，之后才扩展Graph槽位及完整
多请求生命周期验证。仅将INT4接进动态路径已不足以宣布八并发256K可用；本轮
也未验证1M、随机采样、其他权重、完整模型质量或服务性能。
