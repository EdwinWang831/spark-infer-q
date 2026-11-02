# E56：INT4 精度归因、敏感层保护与量化尺度初筛

2026-10-28。**本轮没有找到同时满足精度和八路容量的方案。** 保留少数BF16层
确实降低了logits偏差，但收益不足、内存代价很大；简单MSE尺度搜索也未成为
有效候选。默认格式和E55的原质量门槛均不变。

[方案与后续诊断说明](E56_KV_PRECISION_PLAN.md) · [可校验结果](../results/e56/2026-10-28-kv-precision.json)

## 先实际测混合精度，不假设它有效

常驻模型不重载，使用未改变的每层BF16/INT4生产算子。全模型共有12个attention
层；INT4已包含H32旋转、group32和FP16 scale，这些不是本轮新增优化。

用E47六个短任务逐层消融：BF16、全INT4、仅把某一个attention层恢复BF16，
共84个teacher运行。每次强制相同BF16参考tokens，最多16位置；按六个校准任务
加权raw relative L2的降低量排名，保留负贡献，不在留出数据上选层。

最前四层为43、3、15、47（**代码的0-based layer.index**）。另两层35、27单独
恢复BF16反而略增校准误差，不能把各层贡献当作可加的固定常数。提前规定的
保护1/2/4层组合在排名结束后冻结。

留出集是未参与选层的两个E47 4096-token中英文任务，以及两个新8192-token
中英文ledger/JSON任务。新任务的目标key在尾部；它们检查长输入下的格式/数值
行为，不能证明8K远距离检索，更不能代表256K任务质量。

每个组合执行完整prefill、BF16强制teacher和最多64-token greedy输出。共20次
留出运行、每组合42个teacher位置。没有固定草稿对照，本阶段为target-only
精度诊断，不执行DFlash、动态预算或Graph验证。

| 实际缓存组合 | raw logits relative L2 | mean KL | teacher top1 | greedy整例与BF16一致 | 原质量门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16 | 0 | 0 | 42/42 | 4/4 | 通过 |
| 全INT4 | 13.81726% | 0.00143995 | 42/42 | 4/4 | L2失败 |
| 保护1层BF16 | 12.84727% | 0.00133256 | 42/42 | 4/4 | L2失败 |
| 保护2层BF16 | 10.76433% | 0.00206941 | 42/42 | 4/4 | L2失败 |
| 保护4层BF16 | 10.07388% | 0.00144490 | 42/42 | 4/4 | L2失败 |

保护4层的raw L2相对降低约27.1%，但仍高于3%门槛。所有四个留出任务的客观
目标通过，greedy输出也与BF16一致，因此“门槛失败”不是这四题答错的证据；
同样，这四题输出相同也不足以推翻E55已观察到的256K输出分歧。

逐层混合只用于实验自有state，未更改session权重或全局算子。为匹配现有provider，
state的基准dtype标签保持INT4，而每层cache实际持有BF16或INT4。均逐层记录并
核对实际格式，不用同质state的`cache_bytes()`来计费；没有把这种实验state提升
成公开混合精度API，也未验证其Graph、迁移、跨格式copy或服务生命周期。

## 容量代价排除了“堆BF16层”作为当前主要方案

以E55的八路92.81760GiB完整INT4外推为基线，增加每个受保护层的真实KV storage
差。BF16 dummy scale scalars也计费；row_bytes数值与storage含义分开检查。
新增256K allocator padding没有实测，以下仍是带原E55 reserve的容量外推，
不是八请求分配或已填满状态峰值。

保护每层在八份262160-row状态上约增加2.875GiB storage。保护1/2/4层会分别把
八路完整外推推到约95.69/98.57/104.32GiB；它们仍超过88GiB。即使质量过关，也
不能据此宣布八并发256K可用。

## K/V 拆分与尺度搜索：仅作误差归因

逐层保护失败后，另封存一个因果探针：使用BF16 cache，在每次写入新K/V行后，
通过参考INT4量化、反量化和逆H32重建，随后仍执行BF16 attention。仅改变新写行，
不重复量化历史；固定相同四个留出输入和BF16 teacher tokens。

这不是实际INT4内核：缓存仍是BF16，query/output不走真实INT4旋转路径，且多出
重建到BF16的舍入。因此**不能把模拟指标与真实INT4指标直接相减，声称各部分
贡献率**，也不能据此宣称节省内存或可部署。

| BF16缓存上的数值干预 | raw logits relative L2 | mean KL | teacher top1 |
| --- | ---: | ---: | ---: |
| 无干预 | 0 | 0 | 42/42 |
| 只做H32/BF16舍入/逆变换，不做4bit量化 | 8.19952% | 0.00103648 | 42/42 |
| 仅K做INT4重建 | 10.08274% | 0.00088827 | 42/42 |
| 仅V做INT4重建 | 12.00564% | 0.00150611 | 42/42 |
| K/V均做INT4重建 | 12.07247% | 0.00128702 | 42/42 |
| K/V逐组MSE搜索尺度后重建 | 12.02138% | 0.00268490 | 42/42 |

尺度候选固定为absmax的0.85/0.9/0.95/1/1.05/1.1倍，包含原尺度；逐group按
局部重建MSE选择，不使用答案或teacher标签调尺度。它几乎没有改善全模型raw
L2，KL还变差，不能从局部MSE更小推导最终质量更好。

H32舍入控制的较大变化值得关注，但不能称为“INT4里8.2%误差来自旋转”，因为
这里的缓存重建与真实INT4算术不同。为排除诊断插桩影响，额外运行四输入的BF16
与完全不改数值的identity hook对照，共8次运行；42个位置的完整词表logits
逐位一致。能确认的是：这条模拟中的变化来自数值干预，不能据此确定真实内核的
误差分解。MoE路由变化等原因仍是待验证假设，本轮未测router flips。

## 接下来精度方向如何收敛

不继续靠增加BF16保护层填坑，也不推进这次简单尺度搜索。下一候选应改变KV
表示本身，例如K/V非对称分组、无旋转的仿射量化，或有严格容量上限的高精度
残差/近期区；它们目前都只是候选，未实现或验收。

优先用attention组件及真实完整模型验证候选，保留同一BF16 teacher和客观任务，
将局部KV重建误差、attention误差、centered/raw logits和最终答案分开；通过后
再进入真实INT4 kernel、Graph、256K动态执行及容量验证。若要声称接近BF16，
还需更广泛的真实长任务，而不是只看本轮42个teacher位置。原KL/L2/top1/客观
目标门槛不因本次结果改变，默认格式保持原状。

## 回收、证据和复现

作业632/636/640分别为真实混合精度初筛、数值重建探针、identity控制；独立回收
审计634/638/642均PASS。worker仍为PID46235、load_count1，allocated精确回到
77,393,408,000 bytes，最终READY。全过程没有权重重载、下载、empty_cache、
提高88GiB限额或并行GPU进程。

忽略归档`results/raw/e56/2026-10-28-final/`包含858文件、2,555,880压缩bytes，
SHA256：`016547328682c68d3917893e8b68e4f6af344584d4e15fec631c48915bcf5d6b`。
保存六个作业的immutable源码/manifest/报告及E53/E55容量依据，不包含权重。
导出`PASS_E56_EVIDENCE`检查源码链、同一常驻对象、逐层格式、校准排序、冻结mask、
匹配输入/teacher、指标和质量决策算术、实际storage、容量增量、模拟scope、
identity完整logits相等记录和独立回收。它不把未保存的teacher完整数组当作可
离线重算数据；完整词表相等由封存的identity作业内实时逐位断言并记录。

最终本地回归417项：398通过、19项可选依赖跳过；E56真实归档测试覆盖改变排序/
mask、伪造格式/teacher、放宽阈值、改变决策/输出、缺失消融、资源不足，以及
将BF16模拟伪称容量节省、改尺度和干预工作量等篡改。公共JSON完整保留不利结果。

以下为本次已执行作业，复现实验应使用新的版本/作业号，且前置作业完成后再
运行依赖项；不要重复提交现有作业。原study/probe执行源码均按其实际版本封存，
后续新增的control-only标志不会回写旧stage。

```sh
python3 scripts/e56_stage.py study-v1 632-e56-study-v1.json --phase study
python3 scripts/e56_stage.py probe-v1 636-e56-probe-v1.json --phase probe --study-job 632-e56-study-v1.json
python3 scripts/e56_stage.py audit-v1 634-e56-audit-v1.json --phase audit --study-job 632-e56-study-v1.json
python3 scripts/e56_stage.py probe-audit-v1 638-e56-probe-audit-v1.json --phase audit --study-job 636-e56-probe-v1.json
python3 scripts/e56_stage.py control-v1 640-e56-control-v1.json --phase probe --study-job 632-e56-study-v1.json --control-only
python3 scripts/e56_stage.py control-audit-v1 642-e56-control-audit-v1.json --phase audit --study-job 640-e56-control-v1.json
python3 scripts/e56_export.py --raw results/raw/e56/2026-10-28-final/worker --out results/e56/2026-10-28-kv-precision.json
python3 -m unittest discover -s tests -v
```
