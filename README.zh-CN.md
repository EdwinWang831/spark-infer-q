# SparkInfer-Q

[English](README.md) | **简体中文**

在单台 NVIDIA DGX Spark（GB10 / SM121）上加速 Qwen 模型推理的研究项目，借鉴 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)。

## 现状一览

截至 2026-10-28，本分支包含 E0–E58 研究记录（E26 依用户要求中止）、已验证的 K1–K3 原生调度运行时、单请求开销优化、双请求 target-only API 及显式双长请求动态适配。实验候选保留各自的验证范围，默认路径不变。

已汇总的 main 与独立的 K4/服务、自演化设计路线，见[仓库导航](docs/REPOSITORY_STATUS.md)。

E33 用 INT8 辅助筛选、原 BF16 头最终裁决，在相同 W8 草稿/线性预算/Graph 基线上，实际 decode 46.91→48.51 token/s（+3.41%，root95% +2.39%～+4.12%）；8 条旧工程 root、64 次计时，输出、提交 logits/特征/状态及预算轨迹一致。当前仍为研究路径；E30–E32 及 E34–E35 候选均未晋升。

- **做什么**：Qwen3.8-Flash-Next（NVFP4，TensorFold 内核）用 DFlash2 草稿模型做推测解码，由 DeepInfer 的 MTPVarlen 决定每一步验证多少个草稿。性能实验及 K2 动态预算运行时采用同步、单请求、greedy 执行。
- **最大的单项 decode 改进**：E15/E16 训练的轻量置信度预测器。在另行冻结的新任务上，它比默认的滞后预算快约 10%（E16：44.08 vs 40.21 tok/s），目前是可选路径。
- **prompt 处理**：8 行分块让完整请求快 1.67 倍，首 token 时间之和从 19.69 秒降到 4.23 秒；decode 速度基本没变（E8）。
- **双请求 GPU batch**：DeepInfer 原生调度器已能让两个请求共享一次 target 前向。受控 target-only 对照中，decode 总吞吐从 23.64 升至 38.63 tok/s（+63.4%）；目前是显式同步 API，尚未验证联合 DFlash 和 HTTP，见[报告](docs/JOINT_TARGET_EXECUTION_REPORT.md)。
- **大块 prefill（E36）**：保留现有 decode 算术的候选，在七条前缀上通过 logits、全部草稿特征、有效状态和 8 步交接的精确对照；prompt 本体从 110.87 提升到 889.22 tok/s（8.02倍）。目前是研究适配器，尚无完整请求或 HTTP 收益，见[报告](docs/E36_PREFILL_NUMERICS_REPORT.md)。
- **Prefill 剖析（E37）**：两个 512-token 样本中，路由专家占 GPU 内核累计耗时 52.5%–54.9%。GDN scratch 复用使配对 prompt wall 增加 1.02%，未采用，继续保留 E36，见[报告](docs/E37_PREFILL_PROFILE_REPORT.md)。
- **自主 Prefill 模块（E40）**：由引擎管理分块、工作区与特征捕获，显式接入 target-only/DFlash，仍复用 TensorFold 算子。七条前缀和56步 decode 交接精确通过；eager token API配对实测中，受控512-token输入/24-token输出的DFlash请求耗时4.839→0.684秒，对照为旧rows8，不是最新原生引擎或Graph/E33。[报告](docs/E40_PREFILL_EXECUTOR_REPORT.md) · [用法](docs/E40_PREFILL_EXECUTOR_USAGE.md)。
- **完整优化链路验证（E41）**：自主 prefill 已通过 Graph/E33/W8 交接、部分提交、取消复用及本机 HTTP 功能检查，输入覆盖至1,536 tokens。相同配置的 token API 请求中，code513 总耗时5.464→1.202秒，mixed1536 为15.347→2.131秒，对照为rows8；残差 decode 为54.63→54.41 tok/s，没有新增 decode 收益。[报告](docs/E41_PREFILL_GRAPH_REPORT.md)。
- **更大自主 prefill 分块（E42）**：在已优化的512行对照上，1024行使长输入prefill合计耗时减少5.80%，code1024/mixed1536完整请求耗时减少4.58%/1.79%。logits、特征、状态、Graph交接与生命周期通过；开放1024行显式选项，默认保留512。残差decode仅+0.15%，没有证实新增decode收益。[报告](docs/E42_WIDER_PREFILL_REPORT.md)。
- **Mia专家tile接入（E44）**：将保留MIT署名的共享输入tile接入自主prefill后端。相对已优化chunk1024，长输入prefill合计耗时减少6.97%；code1024/mixed1536首token时间减少10.60%/3.76%，完整请求耗时减少4.51%/1.47%。完整模型及最终公开接口精确通过；显式启用，没有新增decode收益或最新Mia排名结论。[报告](docs/E44_MIA_PREFILL_TILE_REPORT.md)。
- **有界草稿历史与4K（E45）**：每请求2040行镜像环通过绝对位置缓存、完整4K Graph/QSA/oracle和生命周期检查，原始草稿特征固定99.61MiB。配对汇总decode60.001→59.972 tok/s，没有新增提速；双长请求仅验证target-only隔离。这是每请求1M的基础，target分页/量化KV及位置扩展质量仍待验证。[报告](docs/E45_LONG_CONTEXT_REPORT.md) · [用法](docs/E45_LONG_CONTEXT_USAGE.md)。
- **直接256K实跑（E46）**：单个原生请求处理262,080-token输入，用时230.4秒（1,137.5 input tok/s），生成24 tokens；三处合成检索键、16种长前缀Graph桶和target-only续写/状态检查通过。峰值CUDA allocated81.41GiB；尚未验证双请求256K或1M。[报告](docs/E46_256K_REPORT.md)。
- **双256K状态与KV压缩（E47）**：两个独立262,080-token target-only请求共享计算缓冲区、输出头和prefill工作区；INT8 KV＋QSA索引每请求4.13GiB，峰值allocated82.63GiB，各自检索3/3，Graph/状态/缩放因子隔离通过。原始logit精度门槛未过，原生prefill优先导致225.8秒输出空窗；这是容量研究结果，默认不变。[报告](docs/E47_KV_DUAL_REPORT.md) · [用法](docs/E47_SHARED_GRAPH_USAGE.md)。
- **双256K公平调度（E48）**：显式使用原生轮询阶段，在1024行prefill计划之间提供decode机会。一组同期配对中，A最长输出停顿225.42→0.842秒，完成时间453.03→241.60秒；双请求总耗时仍约453秒。输出与完整已提交状态一致，B首token时间增加0.076%。仅target-only token API；INT8精度门槛仍未过，默认不晋升。[报告](docs/E48_PREFILL_FAIRNESS_REPORT.md) · [用法](docs/E48_PREFILL_FAIRNESS_USAGE.md)。
- **长输出动态DFlash与四路容量（E49）**：BF16单请求261,120-token输入/512-token输出配对，target-only→动态DFlash的decode16.36→30.81 tok/s（+88.37%），输出和完整已提交状态一致，完整wall减少5.27%。INT8配对为13.29→29.18 tok/s（+119.58%），与BF16输出在326 tokens后分歧。四份完整INT8状态＋公共区外推90.14GiB，尚未计Graph/head/草稿，超过当前88GiB限额；未执行四路长请求。[报告](docs/E49_LONG_DYNAMIC_REPORT.md)。
- **自主长请求执行优化（E50）**：显式草稿query Graph和原生输出发布缓存保持完整输出/状态/预算轨迹一致；同期BF16单256K/512-token配对decode32.14→34.49 tok/s（+7.34%），完整wall减少1.58%。八状态＋公共区外推FP8为106.77GiB、INT4为95.52GiB，尚非八请求执行或INT4质量结果。当时有界QSA原始索引仅完成CPU模型；GPU实现及验证见下方E51。[报告](docs/E50_OWN_OPTIMIZATION_REPORT.md)。
- **有界QSA原始索引（E51）**：显式2048行环保留完整FP8/BF16 K/V和池化历史，每请求tensor storage实省0.744GiB。GPU回绕/Graph/生命周期及同格式256K动态输出/有效状态验证通过。八份FP8状态＋公共区外推降到100.81GiB，仍未计Graph/head/草稿且超过当前88GiB限额。BF16 decode34.44→34.45 tok/s基本持平；共享地址AB/BA复测38.494→38.490 tok/s持平，未确认首次FP8单组的大幅增益。[报告](docs/E51_BOUNDED_INDEX_REPORT.md) · [用法](docs/E51_BOUNDED_INDEX_USAGE.md)。
- **Decode专家流水筛查（E52）**：复用E43流水核，96组decode边界和24份真实完整MoE均精确，但热复用耗时+0.787%、缓存扰动−0.007%，区间均跨零，未过5%门槛；不做整模替换，已有decode速度和默认路径不变。[报告](docs/E52_DECODE_PIPELINE_REPORT.md)。
- **长请求并发预算（E53）**：实际运行两个独立261120-token输入、各512-token输出，输出及完整有效状态与独立执行相同，峰值allocated81.14GiB；但A有7轮验证宽度和144轮预测概率变化，严格门槛失败，适配保留研究态、不晋升。计入完整开销的八路估算103.67GiB，比当前88GiB上限高15.67GiB；未运行八路。[报告](docs/E53_LONG_CONCURRENCY_REPORT.md) · [研究用法](docs/E53_LONG_CONCURRENCY_USAGE.md)。
- **每请求草稿投影缓存（E54）**：切换时保留上下文投影/K/V和计数，增加50MiB/请求；重建诊断复现E53分歧。两个填满的261120-token输入、各512输出的动态请求，其输出、完整有效状态、特征、概率、宽度和接受轨迹均与独立执行及原E53单请求对照一致，严格门槛通过。峰值allocated81.24GiB，默认关闭；八路预算仍约104.06GiB，高于88GiB，本轮不声明吞吐提升。[报告](docs/E54_REQUEST_DRAFT_CACHE_REPORT.md) · [用法](docs/E54_REQUEST_DRAFT_CACHE_USAGE.md)。
- **常驻盘点与INT4 KV（E55）**：去重storage确认target权重69.85GiB，只清理公共区无法让八份FP8状态进入88GiB。显式INT4＋索引环使每状态allocation从3.51降至2.10GiB；39例GPU差分、32个全模型Graph窗口通过。沿用八任务BF16相对质量门槛，INT4因raw logits L2偏差11.23%未通过（top1 54/54），不提升默认或质量等级。单261120输入/512输出动态请求均与同格式greedy全输出/有效状态一致，但INT4与BF16输出有分歧。八路allocation外推104.06→92.82GiB，仍超88GiB。[报告](docs/E55_RESIDENCY_REPORT.md) · [用法](docs/E55_INT4_KV_USAGE.md)。
- **INT4精度归因（E56）**：84次校准teacher运行排序12个attention层，再冻结1/2/4层BF16保护组合，在四条留出4K/8K中英文输入上验证。保护4层使raw logits L2从13.82%降至10.07%，仍未过3%门槛，且增加八路容量预算；这批四例greedy输出均与BF16一致。K/V重建与尺度搜索仅为BF16存储诊断，尚无新内核或质量提升结论；identity插桩完整logits一致，独立回收通过。[报告](docs/E56_KV_PRECISION_REPORT.md)。
- **非对称仿射KV与延后压缩（E57）**：无旋转4bit打包编码通过解析、CPU/GPU及变输入codec Graph检查；六个因果重建候选仍未过3% raw logits门槛，八路布局外推也超过88GiB。另测真实原生INT4 decode：BF16 prefill后转换KV，在4K/8K/16K共100个匹配decode teacher位置上，L2从9.58%降至7.22%、KL减少25.1%，转换后storage与原生INT4一致。两条128-token教程输出仍与BF16有分歧，不提升质量/默认，不宣称Graph或动态路径已适配。[报告](docs/E57_AFFINE_DEFERRED_KV_REPORT.md)。
- **原生KV转换与混合解码（E58）**：INT4/FP8转换在8组偏移/形状及codec Graph回放中与原生prep一致；六条真实前缀的新旧INT4转换，完整logits/输出/有效状态逐位一致。6次BF16 prefill供36次私有副本解码，并保存可复用teacher。INT4、4层FP8＋INT4、全FP8的decode L2为7.22%/5.20%/4.06%，top1为98/97/100（共100），完整输出一致4/4/5例（共6）；仍全部未过3%，八路外推92.82/96.57/104.07GiB均超88GiB，不提升默认或引擎Graph路径。[报告](docs/E58_NATIVE_CONVERSION_REPORT.md)。
- **还没做到**：整体速度没有超过原生 TensorFold（E7 时 45.62 vs 49.18 tok/s，E8 之后没有重测）；多请求 DFlash、长上下文HTTP服务、物理KV分页、独立压缩KV质量校准、超出256K的上下文和随机采样仍待推进。

## 主要结果

| 实验 | 改了什么 | 结果 | 现状 |
| --- | --- | --- | --- |
| [联合 Target](docs/JOINT_TARGET_EXECUTION_REPORT.md) | 原生调度下两个请求共享一次 target 前向 | 两组中英文请求对 × 四轮，decode 总吞吐 23.64 → 38.63 tok/s（+63.4%） | 可选、同步 target-only；尚未接双请求 DFlash |
| [E36](docs/E36_PREFILL_NUMERICS_REPORT.md) | 大块 prefill，统一投影/递归/MoE 算术 | 七条4–512 token前缀、四轮，prompt本体 110.87 → 889.22 tok/s（8.02倍） | 研究适配器；精确门槛通过，默认引擎待接入 |
| [E37](docs/E37_PREFILL_PROFILE_REPORT.md) | Prefill 剖析与单变量 GDN scratch 复用 | 48 次配对 prompt 中复用 wall +1.02%；独立 GPU 剖析显示专家是最大成本 | 不采用复用；没有新增速度收益或默认变更 |
| [E38](docs/E38_PREFILL_EXPERT_REPORT.md) | 两组输入共享专家权重加载与解包 | 路由估算少23.02%权重阶段，但完整MoE设备区间成本+3.93% | 组件精确通过，成本不通过；未进入完整prompt A/B |
| [E39](docs/E39_PREFILL_CTA_REPORT.md) | CTA共享压缩权重，每warp保持16行累加器 | 寄存器占用降低，但完整MoE设备区间成本+3.11% | 不采用；没有新增prompt/decode收益或默认变更 |
| [E41](docs/E41_PREFILL_GRAPH_REPORT.md) | 自主512行 prefill 接入 Graph/E33/W8 完整路径 | code513总耗时5.464→1.202秒；mixed1536为15.347→2.131秒；残差decode −0.41% | 精确性与生命周期通过；显式启用，输入≤1,536 tokens；无4K/HTTP测速结论 |
| [E42](docs/E42_WIDER_PREFILL_REPORT.md) | 自主prefill分块512→1024，decode配置与调度预算相同 | 长输入prefill耗时−5.80%；code1024/mixed1536完整请求−4.58%/−1.79% | 显式选项；短输入不提速，不宣称新增decode收益 |
| [E43](docs/E43_PREFILL_PIPELINE_REPORT.md) | 单tile专家MMA/FMA指令流水 | 完整MoE扰动成本+0.143%，组件精确通过 | 不采用；更早总allocated审计保留32MiB未解释增量 |
| [E44](docs/E44_MIA_PREFILL_TILE_REPORT.md) | 将Mia共享输入plan64/RT4S4 gate-up/RT2S3 down接入自主prefill | 适用MoE成本−26.62%；长输入prefill耗时−6.97%；code1024/mixed1536首token耗时−10.60%/−3.76% | 显式后端；完整模型/最终接口及本轮稳定基线审计通过，默认不变 |
| [E45](docs/E45_LONG_CONTEXT_REPORT.md) | 有界草稿历史/绝对位置投影，修复原生调度水位计数 | 完整4K QSA/Graph及请求精确通过，原始特征环99.61MiB/请求；decode60.001→59.972 tok/s | 显式启用；无新增提速，尚无分页/量化KV或1M支持 |
| [E46](docs/E46_256K_REPORT.md) | 直接256K原生请求，Graph预热改用小状态seed | 262,080-token输入/230.4秒；生成24 tokens、检索3/3、Graph/状态精确通过；峰值allocated81.41GiB | 单请求容量/正确性，BF16 KV；无256K并发或通用质量结论 |
| [E47](docs/E47_KV_DUAL_REPORT.md) | INT8/FP8 KV入口，双私有Graph槽共享串行工作区 | 两个262,080-token请求，KV/index各4.13GiB，峰值82.63GiB，各检索3/3 | 容量/隔离通过；原始精度门槛未过，225.8秒调度空窗；target-only |
| [E48](docs/E48_PREFILL_FAIRNESS_REPORT.md) | 双target-only槽交替获得原生prefill/decode机会 | 一组双256K配对：A最长停顿225.42→0.842秒，完成453.03→241.60秒；总耗时453.24→452.27秒；完整状态一致 | 显式延迟改善；INT8精度门槛仍未过，无HTTP/batch或通用TPS结论 |
| [E49](docs/E49_LONG_DYNAMIC_REPORT.md) | 长输出target-only与动态DFlash对照，无固定草稿组；四状态容量检查 | BF16单请求decode16.36→30.81 tok/s，输出512 tokens，完整状态一致；总wall减少5.27% | 单组配对；四256K容量90.14GiB超过88GiB，INT8精度不晋升 |
| [E50](docs/E50_OWN_OPTIMIZATION_REPORT.md) | 长后缀草稿query Graph与原生输出增量读取 | BF16单256K decode32.14→34.49 tok/s（+7.34%）；完整输出/状态/预算一致，总wall减少1.58% | 显式启用、完整请求仅一组；八FP8状态外推106.77GiB，有界原始索引尚为CPU模型 |
| [E6](docs/E6_DRAFT_HEAD_REPORT.md) | 草稿输出头换成完整词表 Q4 | decode 41.70 → 45.54 tok/s（+9.23%） | 默认 |
| [E8](docs/E8_INTEGRATION_REPORT.md) | prompt 按 8 行分块，沿用 decode 算术 | 完整请求快 1.67 倍；decode −1.14% | 默认 |
| [E13](docs/E13_DRAFT_CACHE_CONFIDENCE_REPORT.md) | 增量缓存 DFlash 的历史投影 | 1K–2K 输入 +6.27%；短输入 −0.31% | 可选 |
| [E15](docs/E15_CONFIDENCE_TRAINING_REPORT.md) | 训练七深度置信度预测器 | 6 个新任务上比滞后预算 +10.00%，比固定七草稿 +10.88% | 可选 |
| [E16](docs/E16_BILINGUAL_EXPANSION_REPORT.md) | 用 600 条中英文提示重新训练 | 20 个新任务上比滞后预算 +9.61%，比 E15 的预测器 +3.18%（其中 2 个任务变慢）；英文 +7.60%，中文 +10.75% | 可选 |
| [E17](docs/E17_COMPLETED_TRAINING_REPORT.md) | 中英文预测器定向补训 | 20个root上36.83 tok/s；比固定预算+11.02%，比E16仅+0.09%，区间跨零 | 未确认超过E16 |
| [E19](docs/E19_MLP_CAPACITY_REPORT.md) | 相同数据的11,157参数MLP | 离线效率比线性低0.55%，未过晋升门槛 | E19候选未晋升 |
| [E21](docs/E21_ADAPTIVE_COLLECTOR_REPORT.md) | 自适应训练标签采集器 | 六个root上采集计算耗时减少21.13%；已知中文例仍慢2.15% | 采集入口已验证，不是服务TPS |
| [E22](docs/E22_PREDICTOR_DIAGNOSIS_REPORT.md) | 可选NumPy标量MLP运行时 | 小头0.3353→0.0118毫秒（28.38×）；诊断回放仍比线性低0.06% | CPU原型，没有新增decode TPS |
| [E24](docs/E24_COVERAGE_EXPANSION_REPORT.md) | 504条新root覆盖问答/代码/数学/多轮/长输入后补训 | 离线比冻结MLP高0.252%、比线性高0.212%，区间均为正 | 离线门槛通过 |
| [E25](docs/E25_ONLINE_COMPARISON_REPORT.md) | 12个root、108次实际decode配对计时 | 新MLP 47.15、冻结MLP 47.28、线性47.49 token/s；两个相对区间均跨零 | 未确认实际提速 |
| [E27](docs/E27_LINEAR_COST_REPORT.md) | 相同线性权重的NumPy执行＋因果实时费用对照 | 原版43.39、NumPy43.65、实时修正43.42 token/s；NumPy +0.60%，重复敏感性区间跨零 | 不替换默认；MLP/混合暂停 |
| [E28](docs/DRAFT_BACKBONE_W8_REPORT.md) | 草稿五层MLP，INT8权重＋BF16激活 | 草稿阶段耗时−6.89%；固定预算decode 42.04→42.29 token/s（+0.61%，区间−0.52%～+1.66%） | 正确性通过，未确认decode收益 |
| [E29](docs/E29_TARGET_VERIFICATION_REPORT.md) | target CUDA Graph，保留DFlash特征捕获 | 相同线性预算47.05→47.65 token/s（+1.28%）；Graph＋W8为48.01（+2.04%）；两项区间均为正 | 8条旧root受控正收益，默认不变 |
| [E30](docs/E30_TARGET_PROFILE_REPORT.md) | 冻结W8/线性/Graph剖析，保持精度的BF16头布局 | target内核时间：专家41.3%、量化投影26.9%、BF16为23.4%；布局仅头部+0.8%，未过门槛 | 剖析完成、候选拒绝，无新增decode TPS |
| [E31](docs/E31_EXPERT_SCALE_REPORT.md) | FP8专家scale成对转换，保持精度 | 转换静态指令减半；专家pair热复用+0.23%、缓存扰动−0.18%，区间均跨零 | 数值检查通过，候选拒绝，不进入整模长测 |
| [E32](docs/E32_COMPACT_STAGING_REPORT.md) | MXFP8投影有效输入行紧凑staging | 每CTA共享内存少16%～31%；真实回放热复用−1.06%、缓存扰动−0.92% | 224组筛查＋96份真实输入通过，性能拒绝 |
| [E33](docs/E33_EXACT_HEAD_SCREEN_REPORT.md) | INT8辅助头筛选＋定向FP64误差界＋原BF16子集精算 | 完整头耗时−36.24%；实际decode 46.91→48.51 token/s（+3.41%，区间为正） | 8条root/64次对照、精确检查通过；研究路径，默认不变 |
| [E34](docs/E34_DEVICE_SCREEN_REPORT.md) | GPU有序候选压缩＋固定容量精算Graph | 修复捕获输入生命周期后，32个真实窗口精确检查通过；相对E33完整头耗时−3.78% | 未过事前5%组件门槛，不做完整decode长测或晋升 |
| [E35](docs/E35_MOE_OVERLAP_REPORT.md) | 共享/路由专家双stream fork/join | 完整MoE耗时热复用−3.04%、缓存扰动−0.50%（区间跨零）；时间线显示重核几乎不重叠 | 96项功能/24份真实输入检查通过，未进入完整decode |
| [E11](docs/E11_PROJECTION_REPORT.md) | 投影改用 MXFP8 split-K | decode +0.62%，完整请求 −0.64% | 可选，默认不变 |
| [E10](docs/E10_TARGET_KERNEL_REPORT.md) | 调专家核的 warp 数和预取深度 | holdout 上 +0.08% | 未采用 |
| [E12](docs/E12_EXPERT_VERIFICATION_REPORT.md) | 按路由专家并集估计验证成本 | 离线成本误差降 26.17%，但专家并集要验证后才知道；能在线用的特征只降 2.42% | 未采用 |
| [E14](docs/E14_CONFIDENCE_BUDGET_REPORT.md) | 按草稿 margin 分桶调整预算 | +0.065%，区间跨零 | 未采用 |

**怎么读这些数字**

- 每个实验用自己的任务集，不同行的 tok/s 不能直接比较，收益也不能相乘。
- 都是单台 Spark、greedy 解码。联合 Target 测量双请求总吞吐，E47/E48研究双target-only 256K请求；较早实验多使用合成提示，上下文最多约 2K token。
- 生成实验核对输出 token、committed logits 和 GDN/QSA 状态；CPU 原型和采集研究的具体核验范围见各自报告。
- 特征提取、CPU 传输、预测等额外开销都算在耗时里。

其余实验是打基础和做诊断：

| 实验 | 内容 | 结论 |
| --- | --- | --- |
| E0–E2 | 环境审计、DeepInfer 构建、SM121 内核探测、模型加载与前向 | 完整 target 权重可以加载并运行；原生 DeepGEMM BF16 不支持 SM121 |
| [E3](docs/E3_WIDE_BLOCK_REPORT.md) | 接入 DFlash2 和 DeepInfer 原生动态预算，8 行 / 最多 7 个草稿 | 状态正确；按深度先验的动态预算与合适的固定长度打平（42.21 vs 42.20 tok/s） |
| [E4](docs/E4_MATCHED_REPORT.md) | 短上下文同机对照 | 动态比固定六草稿 +2.12%；没有领先原生 TensorFold MTP |
| [E5](docs/E5_ENGINE_BRIDGE_REPORT.md) | 让 DeepInfer C++ Scheduler 驱动 Qwen target-only 请求 | 七组 GPU 场景通过；GPU 仍逐个请求执行 |
| [K1](docs/K1_RUNTIME_REPORT.md) | 在 E5 上实现可调用的 Target-only 运行时、流式事件与取消 | 七组真实 GPU 场景通过：233 行 logits、91 次有效状态检查；推测解码尚未接入此运行时 |
| [K2](docs/K2_DFLASH_SCHEDULER_REPORT.md) | 原生 Scheduler 接入 Q4 DFlash2 与动态预算 | 现有 E17 本轮动态预算，保留原生滞后/E16 动态对照；同步执行 |
| [K3](docs/K3_GRAPH_RUNTIME_REPORT.md) | 动态解码接入 Target CUDA Graph | 显式执行选项、静态工作区及 eager 回退；保留主机完成同步 |
| [K3 + E33](docs/K3_E33_RUNTIME_REPORT.md) | 辅助头/W8/NumPy/Graph 完整迁入原生运行时 | 47.459 vs 同期 E33 48.378 tok/s；约1.90%接入开销，显式profile |
| [K4](docs/K4_RUNTIME_OVERHEAD.md) | 减少单请求轮询、token 历史复制和 CPU 元数据分配 | 47.282 → 47.994 tok/s（+1.51%）；同期 E33 差距 2.36% → 0.89% |
| [E7](docs/E7_TENSORFOLD_MATCH_REPORT.md) | 充分预热后对照原生 TensorFold | 整体 45.62 vs 49.18 tok/s；8 个自然完成的任务 45.57 vs 43.58 |
| [E9](docs/E9_DECODE_PROFILE_REPORT.md) | 用 Nsight 剖析 decode | target 验证占 83.81%，草稿占 14.39% |

[E20固定宽度采集](docs/E20_FAST_COLLECTOR_REPORT.md)保留为研究对照，推荐已验证的自适应[采集入口](docs/FAST_COLLECTOR_USAGE.md)。[E18短草稿](docs/E18_PREDRAFT_COMPUTE_REPORT.md)只让草稿计算减少0.74%，没有建立在线提速。

**E23–E25（2026-10-23 UTC）：** 固定数据的新机会损失目标[未通过独立门槛](docs/E23_OBJECTIVE_COMPARISON_REPORT.md)。随后[504条新root的覆盖补训](docs/E24_COVERAGE_EXPANSION_REPORT.md)已完成，并通过72个root的离线测试；但[实际decode对照](docs/E25_ONLINE_COMPARISON_REPORT.md)比冻结MLP低0.263%、比线性低0.717%，区间均跨零。没有确认实际提速，不替换当前策略；离线估算和实测decode仍分开报告。

**E26–E27（2026-10-23 UTC）：** [混合路线](docs/E26_RESIDUAL_HYBRID_PLAN.md)依用户要求在18/108次计时后取消，不作性能结论。[线性实际对照](docs/E27_LINEAR_COST_REPORT.md)完成12条新root、108次计时：实时费用修正比原版+0.07%、比冻结费用NumPy−0.53%；纯NumPy比原版+0.60%，纳入重复波动的区间仍跨零。全部正确性通过，常驻模型未重载，默认不变。

**E28（2026-10-23 UTC）：** [草稿MLP W8A16](docs/DRAFT_BACKBONE_W8_REPORT.md)完成8条旧工程回归root、64次计时，target输出token、已提交logits与live state检查全部通过。观测草稿接受率73.74%→73.58%，验证多16轮。BF16控制仍常驻，量化包大小不是实测整机净省内存。本轮仅固定预算，量化后的动态策略尚未验证。

**E29（2026-10-23 UTC）：** [target验证对照](docs/E29_TARGET_VERIFICATION_REPORT.md)完成32个校准前缀、1,536次宽度计时、96次三组decode和32次同期预算计时。已有线性预算比固定7草稿少计算36.72%的验证行、验证耗时少12.70%，decode 42.32→47.31 token/s（+11.78%）；这是复核已有策略，不是新预测器收益。纯Graph在决策/提交轨迹精确相同下+1.28%，是本轮新增执行结果。W8及费用变化比纯Graph再高0.75%，但区间跨零。输出token、logits、特征与live state检查通过，权重保持常驻。

**E30（2026-10-23 UTC）：** [历史核对与当前路径剖析](docs/E30_TARGET_PROFILE_REPORT.md)确认E10/E11已试过专家发射/预取和投影调参。本轮4条旧校准root、32个早期步骤统计真实target GPU内核，profiler扰动4.85%～8.67%。新NK→KN输出头布局在32份真实激活上保持logits精确一致，但只让头部省约0.8%时间，还多1.27GB副本，因此不进入整模长测。target/draft未重载，没有新训练或精度修改。

**E31（2026-10-23 UTC）：** [专家scale成对转换](docs/E31_EXPERT_SCALE_REPORT.md)通过65,536对scale穷举、24组真实路由专家输出和8次诊断前向的Graph/logits/特征/状态检查。CUDA已提供该成对操作，这是实现优化尝试，不是新算法。转换静态指令减半，但两种计时条件均无可靠收益，停止候选、不做整模长测。权重常驻，默认路径不变。

**E32（2026-10-23 UTC）：** [紧凑staging](docs/E32_COMPACT_STAGING_REPORT.md)去掉无效输入行填零，保持有效行MMA/累加顺序。224组算术/边界筛查、96份真实输入及8次诊断logits/特征/状态检查通过。首版采集器漏掉类方法调用、完整性断言失败，原记录保留并用私有代理补齐。两种计时区间均为负，不进入整模长测；权重常驻、默认不变。

**E33（2026-10-23 UTC）：** [头筛选报告](docs/E33_EXACT_HEAD_SCREEN_REPORT.md)保留首轮成本门槛失败记录，随后仅融合相同保守区间计算。32份校准窗口均保留原argmax，完整头耗时少36.24%；另8条旧工程root、64次完整生成实测decode +3.41%（root95% +2.39%～+4.12%），四轮均为正。两组均验证12,748行，接受/提交轨迹相同；筛选路径2,596个计时轮次无自然回退。原BF16权重最终裁决，启用时辅助payload多约641MB；未切默认服务，也未新增原生TensorFold排名结论。

## 工作原理

每个 decode 步分三步：

1. **出草稿**：DFlash2 读取 target 已提交的特征，一次给出最多 7 个草稿 token。输出头用 Q4，默认草稿主干和 selector 保持 BF16。E28另行测试INT8 MLP权重＋BF16激活。
2. **定预算**：决定这一步让 target 验证几行（1–8 行）。K2 服务默认使用现有 E17 预测器加本轮 post-draft MTPVarlen 动态预算；保留原生滞后/E16 动态预算对照。
3. **验证并提交**：target 一次验证这些行，接受与 greedy 结果一致的最长前缀，只把这部分写入 GDN/QSA 状态。

| 部分 | 来源 | 本仓库代码 |
| --- | --- | --- |
| Target 执行、CUDA 内核、GDN/QSA 提交 | TensorFold；可选Mia/TensorFold prefill专家tile由本项目接入 | `scripts/e3_target.py`、`spark_infer_q/csf_loader.py`、`spark_infer_q/prefill_experts.py` |
| 草稿模型 | DFlash2 | `spark_infer_q/dflash2_block.py`、`draft_head.py`（Q4 头）、`draft_context.py`（上下文缓存） |
| 验证预算 | DeepInfer MTPVarlen | `spark_infer_q/verification_budget.py`、`confidence_head.py`、`confidence_features.py` |
| greedy 验证规则 | 本仓库 | `spark_infer_q/speculative.py` |
| prompt 分块 | 本仓库 | `spark_infer_q/prompt.py` |
| 调度桥接（E5） | DeepInfer C++ Scheduler | `spark_infer_q/deepinfer_bridge.py`、`executor.py` |
| 联合 Target 运行时 | 本仓库的执行与所有权适配，复用 DeepInfer 调度和 TensorFold 算子 | `spark_infer_q/kaifra_joint_target.py`、`kaifra_joint_runtime.py` |

完整执行器是 [`scripts/e8_ours.py`](scripts/e8_ours.py)，提供 SSE 接口。它需要准备好的研究环境、外部依赖和检查点，见[用法说明](docs/PROMPT_USAGE.md)。实验之间由[常驻 worker](docs/RESIDENT_RESEARCH_WORKER.md) 保持权重加载。

[K1 运行时](docs/K1_RUNTIME_USAGE.md)提供原生 Scheduler 驱动的 Target-only API 和 loopback SSE 入口；[K2](docs/K2_DFLASH_SCHEDULER_USAGE.md)进一步接入 DFlash2 和原生动态预算。K2 动态预算限定单请求，HTTP 一次一请求；独立的[联合 Target API](docs/JOINT_TARGET_EXECUTION_REPORT.md#api-与复现)已通过完整模型双请求 GPU 执行、状态、取消复用与自然 EOS 检查，尚未切换默认服务路径。

## 下一步

- 保留E29 Graph为可选研究路径，优先target数据搬运/融合及真正共享GPU计算；W8相对纯Graph的独立增益仍未确认。
- E8 之后重新对照原生 TensorFold。
- 暂缓MLP/混合路线。先检查线性预算改变能否减少实际验证工作，再决定是否继续训练；保留独立测试与实际decode门槛。
- 更长上下文：E50已优化单256K动态链路并核算八份完整状态；接下来优先验证保留FP8 KV的有界原始索引暂存、完善内存池预算及多请求草稿/提交隔离。INT4另需质量门槛；每请求1M仍需超出当前262K配置的位置/质量证据。
- 将已验证的联合 Target 接入多请求动态 DFlash，再推进异步服务和更广任务覆盖。多请求调度的整体设计见[全局调度 RFC](docs/GLOBAL_COST_AWARE_SCHEDULER_RFC.md)（仅为设计提案）。
- [提案状态表](docs/EXPERIMENTS.md#proposals-without-results-yet)区分尚未执行、部分验证的想法与后续已完成实验。

## 快速开始（不需要 GPU，不下载权重）

后续 DeepInfer 构建以 Python 3.12 为目标。以下脚本只用 Python 标准库，**不会安装依赖或修改 GPU 设置**。

```bash
git clone https://github.com/EdwinWang831/spark-infer-q.git
cd spark-infer-q

# 输出 JSON 环境报告，不导入 PyTorch，也不探测 torch.cuda。
python3 scripts/probe_environment.py --static-only

# 指向已有的本地 Hugging Face config.json，不下载检查点。
python3 scripts/inspect_hf_config.py /path/to/model/config.json

# 运行单元测试，无需 GPU；没有 PyTorch 时会跳过少数测试。
python3 -m unittest discover -s tests -v
```

**隐私：** 环境报告默认不提交。分享前先检查内容；不要提交 API key、token、模型权重、私有路径或个人标识信息。

## 文档导航

| 想看 | 去哪里 |
| --- | --- |
| 所有实验的计划、通过标准和结论 | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| 单个实验 | `docs/E*_PLAN.md`（事先冻结的方案）、`docs/E*_REPORT.md`（报告）、`results/eN/`（脱敏数据） |
| 用法 | [prompt 分块执行器](docs/PROMPT_USAGE.md)、[Q4 草稿头](docs/DRAFT_HEAD_USAGE.md)、[草稿上下文缓存](docs/DRAFT_CONTEXT_CACHE_USAGE.md)、[投影路由](docs/E11_PROJECTION_USAGE.md)、[常驻 worker](docs/RESIDENT_RESEARCH_WORKER.md) |
| 架构与移植 | [候选架构](docs/ARCHITECTURE.md)、[适配接口](docs/ADAPTER_INTERFACES.md)、[初始源码审计](docs/PORTING_MATRIX.md) |
| 设计提案 | [全局调度](docs/GLOBAL_COST_AWARE_SCHEDULER_RFC.md)、[Kaifra愿景](docs/KAIFRA_VISION_RFC.md)、[实施路线](docs/KAIFRA_IMPLEMENTATION_ROADMAP.md)、[Decode前沿](docs/KAIFRA_DECODE_FRONTIER_RFC.md)及[验证矩阵](docs/KAIFRA_DECODE_VALIDATION_MATRIX.md)。均为设计提案，不代表服务能力已实现 |

## 研究问题与范围

能否把 DeepInfer 在引擎、调度、推测验证和内存管理上的长处用到**单台 DGX Spark** 上，同时把模型执行器和 GPU 内核适配到 Qwen3.8-Flash-Next？还是基于 vLLM / SGLang 的实现，能以更少的工程量得到经过验证的速度？**推理底座由实验结果决定**，不预先选定。

- **范围内：** 验证 SM121 兼容性、Qwen 模型适配、GPU 投机验证、准确的 GDN/QSA 状态回滚与提交、延迟和吞吐分析。
- **初期范围外：** 重新训练 Qwen、搭建生产 Web 服务、假设 DeepInfer 已支持 GB10，以及未经实测就宣称加速。
- **基线：** 在同一台 Spark、相同输入、采样设置和检查点格式下对照 TensorFold、vLLM 与 SGLang。公开结果只作参考；同条件性能和运行可用性需要实际验证。

## 上游项目与归属

- [DeepSeek DeepInfer](https://github.com/deepseek-ai/DeepInfer)（Apache-2.0）
- [TensorFold](https://github.com/ashhart/TensorFold)（模型执行与 CUDA 内核；须保留适用的上游许可证和声明）
- [Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)（外部提供的草稿检查点与参考实现）
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

**引入上游源码或权重前，须检查其许可证和声明。** 本仓库包含研究计划、诊断/构建/测试工具、有界的实验请求执行器、对外部 TensorFold 模块和完整 target 的适配、有界的 eager DFlash2 桥接，以及脱敏实验记录。现包含保留MIT署名的Mia/TensorFold prefill专家tile适配，见[第三方说明](THIRD_PARTY_NOTICES.md)；不包含完整Qwen模型实现或模型权重，加载与其余target算子仍依赖外部TensorFold源码和检查点。

许可证：[Apache 2.0](LICENSE)。
