# E58：原生KV转换一致性与INT4/FP8混合对照

2026-10-28。**转换一致性已在本轮覆盖的真实前缀上确认；少数层FP8改善L2，
但没有同时满足原精度门槛与八路容量。** 默认保持原状，不宣称吞吐提升。

[事前方案](E58_NATIVE_CONVERSION_PLAN.md) · [可校验结果](../results/e58/2026-10-28-native-conversion.json)

## 新转换内核与原生prep一致

新增`kv_conversion_kernels.py`，对已准备好的BF16 K/V行直接调用TensorFold
原生`quant_groups_4`或`store_kv` JIT体。调用者拥有目标cache；内核不改变
session权重或源cache，也不自动替换state指针。变更指针后必须作废/重新捕获
Graph，本轮未复用旧target Graph。

对同一批projection先做BF16和INT4/FP8原生prep，再把BF16结果转换，与原生
压缩prep比较。两个格式各覆盖位置/行数(0,1)、(1,3)、(2047,8)、(261120,1024)，
共8例，K/V编码及所有有效scale逐位一致，紧随写入区之后的guard行未改变。
两个格式在固定目标地址、替换源值后的codec Graph replay也一致。这里只验证
转换codec的Graph，不是验证整个target/引擎Graph。

652 gate-v1在偏移为1的Triton标量特化上失败：Python int不能直接`.to(int64)`；
改为先与row tensor求和后转换类型。失败及原源码保留，654 gate-v2全部通过。
模型未重载，失败后allocated也恢复基线；未把失败记录改为PASS。

## 同一BF16前缀上的36次解码

沿用E57六条4K/8K/16K中英文输入，每条**仅一次BF16 prefill**，合计6次、
57,344输入tokens。保留完整源前缀，恢复compact seed后逐模式克隆私有state，
核对完整有效前缀一致、KV/循环状态不共享地址。每次解码后源前缀指纹不变。
这里的短状态clone峰值可容纳，不代表256K或八路准入方案。

各模式均使用相同prefill logits、BF16 teacher tokens及输出上限64/128：

- BF16 KV decode。
- Torch参考转换后INT4 decode（E57复现控制）。
- 新原生转换后INT4 decode。
- 2层FP8＋其余INT4：zero-based层43、3。
- 4层FP8＋其余INT4：层43、3、15、47。
- 全FP8 KV decode。

FP8保护层来自E56原先冻结的六短任务排序，未根据这批解码结果重新选层。
这里“全FP8”只指转换后的KV；所有模式prefill都为BF16，target权重本身及
已量化的专家/矩阵均未改变。

混合state的基准dtype标签为INT4，但真实cache逐层为INT4或FP8；逐层核对并
使用实际unique storage计费，不依赖同质`cache_bytes()`估计。它仍是研究
state，没有提升为公开混合精度接口、Graph/跨格式迁移或服务准入配置。

## E57的改善可复现，转换器不是这批差异的来源

对六个完整真实BF16前缀的12层K/V逐4096行比较原生与参考编码；native4模式
全部code/scale不一致计数为0。两个转换路径的完整teacher logits、greedy输出、
最终有效KV/scale/pool/GDN/PLE状态在六条输入上全部逐位一致。

BF16控制与Torch4控制的输入hash、teacher tokens、decode逐位置指标和完整
输出，也分别与E57对应记录严格一致。新转换实现没有制造上一轮的收益。
这补上了E57报告中的转换实现混杂项，但结论只覆盖已测的实际数据和几何，
不宣称数学上对所有输入/设备逐位等价。

## 混合精度没有通过原门槛

共36次解码。106个teacher位置中有6个是未经压缩的prefill输出；主指标剔除
它们，使用**相同100个decode位置**。四个已有客观任务和两个教程生成任务均
完整运行，不用首token或相同前几个词代替整段输出。

| BF16 prefill后的KV | Decode raw L2 | Decode mean KL | teacher top1 | 完整greedy与BF16一致 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 0 | 0 | 100/100 | 6/6 |
| Torch INT4 | 7.22496% | 0.00343760 | 98/100 | 4/6 |
| 原生INT4 | 7.22496% | 0.00343760 | 98/100 | 4/6 |
| 2层FP8＋INT4 | 5.96713% | 0.00352362 | 96/100 | 4/6 |
| 4层FP8＋INT4 | 5.19764% | 0.00316983 | 97/100 | 4/6 |
| 全FP8 | **4.06414%** | **0.00115019** | **100/100** | **5/6** |

原门槛KL≤.02、raw L2≤.03、top1≥.95、无BF16已通过目标丢失不变。所有压缩
模式均因L2失败，`selected_candidate=null`。四条客观任务未丢失，教程任务没有
外部正确性评分；token分歧不能直接解释为答案正确率变化。

尤其是2层FP8：L2下降，KL略升，top1从98降到96，说明增大精度位宽或降低L2
都不保证实际决策更接近BF16。全FP8在100个teacher位置均命中，仍只有5/6段
完整输出相同，说明短teacher窗口不能替代完整生成验收。

## 实际storage与容量外推

每个16896容量私有state的去重storage在六输入上固定；转换前均为669,401,148
bytes。逐层KV减少量之和与state总storage减少量精确相等，非KV有效state未变。
源BF16前缀副本仍保留作实验控制，因此不能把这些私有state数值当整个进程峰值。

| 解码KV | 私有state storage bytes | 八路262160容量完整预算外推 |
| --- | ---: | ---: |
| 原生INT4 | 370,950,156 | 92.82GiB |
| 2层FP8＋INT4 | 387,170,324 | 94.69GiB |
| 4层FP8＋INT4 | 403,390,492 | 96.57GiB |
| 全FP8 | 468,271,164 | 104.07GiB |

外推沿用E55公共区/Graph/feature/每请求草稿缓存/reserve，只替换实测格式的
KV storage（包括FP8 dummy scales），不是8路执行或allocator峰值。BF16 prefill、
转换临时区、源前缀保留、未知新服务工作区的并发准入还需另测；当前88GiB限额
未改变，所有组合仍超限。

## 本轮结论

原生转换器可在已测路径复现参考转换，数值改善应继续从prefill/解码轨迹本身
研究，而不是归因到转换实现差异。少数层FP8有L2收益，但top1和容量代价使它
不能直接替代INT4；全FP8最接近BF16，却仍未通过原门槛，也不满足八路预算。

本轮没有合格的新默认候选。后续应针对attention误差及其到MoE路由/状态的
传播做定位，并用更广泛的有客观评分任务检验实际质量；不继续把少量案例的
L2下降当作质量结论。Graph、DFlash/动态预算、256K及八并发均未在本轮适配。

## 封存、回收与复现

六条BF16完整词表teacher logits及forced tokens保存为本地`.npz`，逐文件SHA与
每模式记录绑定。导出验证形状、finite、BF16 argmax及token一致；后续可复用
参考，避免为了拿同一baseline重复prefill。候选完整teacher数组未保存，原生/
Torch完整logits相等由封存作业内实时逐位断言记录，不冒充离线重算证明。

654/656为通过的gate/study，658/660独立回收审计通过；652失败保留。worker仍
为PID46235、load_count1，allocated精确恢复77,393,408,000 bytes，队列为空。
没有重载、下载、empty_cache、调整88GiB限额、修改目标权重或并行GPU任务。

忽略归档`results/raw/e58/2026-10-28-final/`含875文件、49,974,245压缩bytes，
SHA256：`81a6fe1616c59cdfcb5d059098a1687b423d8d51a583270c3551df872e11e480`。
其中包含完整BF16参考激活、源码/manifest/作业和E57/E55/E53依据，没有权重。
公共JSON不包含logit数组，只保存哈希、指标、输出和状态指纹。

最终导出`PASS_E58_EVIDENCE`复核原生prep/Graph gate、6次prefill与私有副本、
匹配teacher、原门槛、实际格式/storage、所有首logit剔除、逐模式结果、候选
选择、E57控制复现、失败来源、参考artifact和回收链。426项本地测试：407通过、
19项可选依赖跳过；真实归档篡改测试已执行。codec Graph通过不等于新引擎路径
Graph化，低raw L2也不等于通过答案质量验收。

以下为已执行依赖链，复现实验使用新版本/作业号，不重复提交已存在项：

```sh
python3 scripts/e58_stage.py gate-v2 654-e58-gate-v2.json --phase gate
python3 scripts/e58_stage.py study-v1 656-e58-study-v1.json --phase study --study-job 654-e58-gate-v2.json
python3 scripts/e58_stage.py gate-audit-v1 658-e58-gate-audit-v1.json --phase audit --study-job 654-e58-gate-v2.json
python3 scripts/e58_stage.py audit-v1 660-e58-audit-v1.json --phase audit --study-job 656-e58-study-v1.json
python3 scripts/e58_export.py --raw results/raw/e58/2026-10-28-final/worker --out results/e58/2026-10-28-native-conversion.json
python3 -m unittest discover -s tests -v
```

执行源码在每个stage内不可变封存；当前gate脚本去掉了一个无行为作用的空切片
断言，未改转换内核或已执行数值逻辑。回溯实际GPU验证以封存manifest为准。
