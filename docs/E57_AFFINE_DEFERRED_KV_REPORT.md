# E57：非对称仿射KV初筛与延后INT4压缩

2026-10-28。**六个非对称/残差候选仍未过精度门槛；BF16 prefill后转真实INT4
的路径减小了数值偏差，但也未过3%门槛。** 默认不变，八并发256K仍未实现。

[预先方案及另行封存的扩展](E57_AFFINE_KV_PLAN.md) · [可校验结果](../results/e57/2026-10-28-affine-deferred-kv.json)

## 新编码器与因果边界

新增研究用`affine_kv.py`：不做H32，实际打包4-bit nibbles为uint8，逐组保存
FP16 scale和minimum。K可按同一通道跨16/32 tokens分组，V按同一token内
16/32通道分组；也保留K/V都按通道32分组作为对照。

真实GPU验证覆盖token/channel两轴、group16/32共四种几何：解析可精确表示
网格及首packed byte、常数、CPU/GPU编码与解码一致、更换输入后的codec Graph
replay均通过。它证明编码器的这些边界，不证明压缩attention或整个引擎Graph。

全模型初筛保留BF16源缓存，在attention读时编码历史前缀、解码到BF16工作区。
每个query tile的量化切点为`floor(max(0,start+1-R)/group)*group`，仅对最早
query的近期区之前的完整token组量化，避免使用未来token拟合当前key尺度。
同tile后续query会多保留最多255行，组对齐再多至31行；这个有效残差不是
严格等于参数R。每次attention调用都检查切点，并记录干预行数、编码次数和
最大切点。首token、teacher以及greedy分支均用同一规则。

这条筛选仍保留原BF16缓存，**没有测出常驻内存下降**，也不是直接读取压缩KV
的attention内核。它是因果的完整模型数值实验，与E56的H32模拟算术也不同。

## 六个仿射/残差组合：没有合格候选

沿用E56四条4K/8K开发输入，非新的独立留出集；BF16、identity和六候选共32次
运行，每组42个BF16 teacher位置，完整最多64-token greedy。identity的完整
词表logits和输出逐位一致。

| 配置 | raw relative L2 | mean KL | teacher top1 | greedy整例与BF16相同 |
| --- | ---: | ---: | ---: | ---: |
| K/V按通道32，R0 | 10.88596% | 0.00289076 | 42/42 | 4/4 |
| K跨token32/V通道32，R128 | 9.97675% | 0.00244670 | 42/42 | 4/4 |
| K跨token32/V通道16，R128 | **9.54018%** | 0.00072172 | 42/42 | 4/4 |
| K跨token16/V通道32，R128 | 10.31469% | 0.00429231 | 42/42 | 4/4 |
| K跨token32/V通道32，R512 | 10.13511% | 0.00310664 | 42/42 | 4/4 |
| K跨token32/V通道32，R2048 | 10.13426% | 0.00128636 | 42/42 | 4/4 |

全部客观目标通过，但全部高于原raw L2≤3%门槛，`selected_candidate=null`。
没有按输出四例一致就提升质量等级，也没有更改KL≤.02/top1≥.95/不丢失BF16
已通过目标的标准。小样本输出相同不是256K无损的证据。

容量预测同时计入全部codes、FP16 scale/min、未完成token组padding以及
`R+1024 prepare+256 tile`的BF16残差，不扣掉残差影子的codes。将每层KV布局
乘12、每请求差额乘8，加到E55完整公共区/Graph/草稿/reserve外推：六组合约
**94.55–96.08GiB**，比原生INT4的92.82GiB更高，均超过88GiB。它们不是实际
八状态分配，256K allocator padding、并发prefill峰值还需另测。

## 关键扩展：BF16 prefill完成后才转真实INT4

初筛失败后另行封存native对照，测试BF16 prefill路径是否值得继续。比较：

- BF16 prefill + BF16 decode。
- 原生H32/group32 INT4贯穿prefill和decode。
- BF16 prefill完成后，逐层、每4096行转换为H32/group32 INT4，再用现有真实
  INT4 attention执行decode。

转换调用上游`quantize_ref`，实际持有uint8 packed K/V和FP16 scales，未保留
BF16 KV副本；更新state dtype、row_bytes和version。转换前后核对完整有效的
非KV状态，包括GDN/conv、PLE、pooled历史、未完成raw尾块和位置均不变。
转换期间不复用任何旧Graph；该实验为eager target-only，尚未适配动态执行。

四条开发输入之外，加两条未用于选参数的16,384-token中英文教程生成输入，
各最多128输出，teacher最多32位置。三模式共18次请求。第一输出logit来自
prefill，延后压缩模式天然保持它；因此主结论明确**剔除首logit**，比较相同
100个decode teacher位置，而不是用106位置均值夸大收益。

| 真实decode路径 | Decode raw L2 | Decode mean KL | teacher top1 | greedy整例与BF16相同 |
| --- | ---: | ---: | ---: | ---: |
| BF16全程 | 0 | 0 | 100/100 | 6/6 |
| INT4全程 | 9.58483% | 0.00458717 | 97/100 | 4/6 |
| BF16 prefill后INT4 | **7.22496%** | **0.00343760** | **98/100** | 4/6 |

剔除首logit后，延后压缩组合的raw L2降低24.62%、KL降低25.06%。英文新教程
L2从6.55597%降至5.55738%；中文从7.60228%降至6.50786%，top1从29/31增至
30/31。这支持继续研究prefill精度与压缩时点，但仍不是质量门槛通过。

两条教程的完整输出均与BF16有分歧；它们没有自动客观答案评分，不能据此断言
答案质量下降或提升。四条已有客观任务的目标和输出保持BF16一致。没有用这些
小样本宣称普适接近BF16。

转换后的每份16896容量状态storage为**370,950,156 bytes**，与原生INT4对照
完全相同；转换前为669,401,148 bytes，逐层KV差之和与总storage差精确一致。
这是真实压缩decode，不是仿射筛选的重建工作区。BF16 prefill额外峰值和并发
admission仍有代价；本轮并未解决E55八路约4.82GiB的缺口。

转换使用Torch参考量化器，贯穿INT4控制使用原生prep内的量化器。本轮没有单独
证明这两条量化实现对所有输入逐位一致。因此能确认的是**这个组合路径**改善了
已测logit指标，不能把全部差额严格分解为prefill误差累积，更不能据此断言
H32或某个router是唯一根因。

## 结论与下一步边界

停止将这六个未过关仿射组合推进为压缩attention内核。保留编码器和所有负面
结果，默认BF16不变。延后INT4压缩是实测数值更接近BF16的候选，但原3%门槛
仍失败，不提升默认，也不扩大到256K/Graph/DFlash/八并发结论。

后续优先把prefill与decode的误差来源分开：先验证转换量化器与原生prep量化的
一致性，再评估BF16 prefill与更精细压缩组合，使用新的长任务检验。若要进入
引擎，还需解决转换/Graph重捕获、部分提交和回滚、prefill内存峰值与准入，以及
最终的256K动态输出/状态验收。当前没有吞吐或decode加速结论。

## 回收与复现

作业644/648分别为仿射初筛和真实延后压缩实验，独立审计646/650均PASS。
worker保持PID46235、load_count1，allocated精确恢复77,393,408,000 bytes，
队列为空。没有重载、下载、empty_cache、调整88GiB限额或并行GPU任务。

忽略归档`results/raw/e57/2026-10-28-final/`包含686文件、2,202,065压缩bytes，
SHA256为`0a38f9af1be7aaf555499fcd65e574545f160ee13f2934daa0da9f5634b45a18`。
保存四次作业的源码、manifest、报告及E53/E55容量依据；没有权重。完整teacher
logits没有保存在归档中，identity的逐位比较及转换前后hash由封存作业内执行；
导出复核记录、算术、干预工作量和源码链，不冒充离线重算完整logits。

最终导出`PASS_E57_EVIDENCE`，复核同一常驻对象、外部算子源码/二进制、codec
检查、每query tile因果切点断言记录、teacher/输出工作量、原门槛、候选选择、
布局计费、真实native dtype、转换storage差、非KV不变、剔除首logit和独立回收。
本地回归423项：404通过、19项可选依赖跳过。新增纯布局测试及真实归档篡改测试
已执行；不会把BF16重建筛选标记为压缩常驻或把未测试Graph说成已接入。

现有作业不要重复提交；复现实验使用新版本/作业号，按依赖顺序排队：

```sh
python3 scripts/e57_stage.py screen-v1 644-e57-screen-v1.json --phase screen
python3 scripts/e57_stage.py audit-v1 646-e57-audit-v1.json --phase audit --study-job 644-e57-screen-v1.json
python3 scripts/e57_stage.py deferred-v1 648-e57-deferred-v1.json --phase deferred --study-job 644-e57-screen-v1.json
python3 scripts/e57_stage.py deferred-audit-v1 650-e57-deferred-audit-v1.json --phase audit --study-job 648-e57-deferred-v1.json
python3 scripts/e57_export.py --raw results/raw/e57/2026-10-28-final/worker --out results/e57/2026-10-28-affine-deferred-kv.json
python3 -m unittest discover -s tests -v
```

后续验证：[E58](E58_NATIVE_CONVERSION_REPORT.md)已在相同六条真实前缀上核对
新原生转换与Torch转换，codes/scales、完整teacher logits、greedy输出和有效
状态均逐位一致，并复现本报告控制；此结果补充上述转换实现混杂项的已测范围。
