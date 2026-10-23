# E30：历史 target 优化复核与当前 Graph 路径剖析

2026-10-23 UTC，`PASS_E30_PROFILE_LAYOUT_REJECTED`。当前W8草稿＋冻结线性预算＋完整target Graph的剖析完成；一个保持精度的BF16输出头存储布局候选通过数值检查，但未过性能门槛，停止，不进行完整decode长测。没有新的整模吞吐提升或默认路径变更。[方案](E30_TARGET_PROFILE_PLAN.md) · [脱敏证据](../results/e30/2026-10-23-target-profile-layout.json)。

## 之前做过哪些 target 优化

| 实验 | 实际改动 | 结果与范围 | 本轮处理 |
| --- | --- | --- | --- |
| [E2](E2_NATIVE_REPORT.md) | SM121原生NVFP4专家执行，保留BF16激活 | 建立可用与数值验证，不是整模提速证明 | 沿用 |
| [E10](E10_TARGET_KERNEL_REPORT.md) | 专家warp数和寄存器预取深度 | 独立复核缓存扰动约+0.08%；未晋升 | 不重复扫描 |
| [E11](E11_PROJECTION_REPORT.md) | BF16块/warp/stage，MXFP8显式partial归并 | 整段decode约+0.62%，部分任务退化；未晋升 | 不重复同类参数扫描 |
| [E29](E29_TARGET_VERIFICATION_REPORT.md) | 完整target前向Graph，保留DFlash特征 | 相同线性预算及轨迹下decode+1.28% | 作为研究基线 |

报告提交分别为E10 `e9e3fcd`、E11 `ebbd016`、E29 `1cabb37`。因此当前工作不是第一次优化target，也不能把旧候选重新命名后当成新方向。上述收益分别来自不同实验，不能相加或相乘。

## 当前路径的 GPU 热点

复用E29校准清单中的英文问答、英文代码、英文长输入、中文长输入四条，分别观察前8个动态decode轮次，共32轮。不是新的独立测试集，也没有测完整请求速度。冻结E28 W8量化算法/扩展、E17 NumPy线性模型、E29 W8费用和原target Graph。

使用PyTorch CUPTI profiler记录实际CPU/CUDA活动。导出时仅统计Chrome trace的`cat=kernel`，且位于CPU `user_annotation=e30.verify`范围内的真实内核；排除`gpu_user_annotation`父级范围，避免把父级与子内核重复相加。原worker报告中的`gpu_kernels`字段也含CUDA父级annotation，不能直接求和；本地导出器从原始trace重新分类并复算。

| target验证内核家族 | 占target GPU内核累计时间 |
| --- | ---: |
| NVFP4专家 | 41.271% |
| 量化投影 qmmf | 26.862% |
| BF16矩阵乘 | 23.365% |
| 其他 | 8.502% |

共57,504个target内核事件，累计1.884561s；GPU kernel区间并集1.883849s，包含拷贝/清零的活动并集1.884553s；对应主机verify范围1.956934s。累计时长、区间并集和主机耗时分别报告，不混作同一口径。这里是采样的target范围，不是完整decode阶段占比。

完整词表BF16输出头通过唯一grid `[1,3880,1]`定位，32轮各1次，约占target kernel累计时间的10.544%。其余BF16时间主要为既有混合层投影等，不能把23.4%全部归到输出头。

每条normal/profile/normal；profiler让这些短段变慢约4.85%～8.67%，所以采样耗时不能作为正常吞吐。原始trace没有提供新的DRAM/SM硬件计数器；静态寄存器、grid或估算occupancy也不能证明带宽饱和。本轮没有重试更改系统profiler权限、重新加载模型或重启worker配合Nsight。

## 为什么量化后还有 BF16 矩阵

当前模型是混合精度。草稿的W8只替换五层MLP；草稿Q4输出头是其自己的压缩执行版本，不会同步替换target输出头。target的routed experts保持NVFP4压缩，量化投影沿用相应格式，但现有加载路径仍把完整词表头及混合层投影作为BF16矩阵执行。

BF16激活与BF16权重也需要区分：NVFP4专家保留BF16输入/中间表示，并不等于把整份专家权重展开成BF16。当前测得的target头shape为`[248320,2560]`，BF16体积1,271,398,400 bytes。若进一步量化这个最终验证头，需要另行验证target数值和生成质量；本轮只改物理存储读取布局，不改变精度或量化策略。

## 单一候选：输出头 NK→KN 预排布

旧E11测试了块形状/warp/stage，但没有测试这个权重转置存储候选。新核保留原BF16数值、BK64、SK1、BM16、BN64、4warp、3stage和FP32累加，改变权重在内存中的布局及Triton读取方向。原head/全局dispatch未替换；候选只在隔离微基准执行。

使用四条root前8轮的32份真实head输入，不是随机合成激活。候选全部logits与原BF16头及实际Graph logits位级一致。每份输入6轮配对交替，每区间10次调用，共384个CUDA-event区间；packing、clone、检查和profiler不在微计时中。矩阵远大于L2，但没有把这视作全缓存驱逐证明。

| 汇总32份输入的各组中位数 | 原NK路径 | 转置KN候选 |
| --- | ---: | ---: |
| GPU时间之和 | 214.955685 ms | 213.243374 ms |

微基准速度因子+0.803%，耗时减少0.797%，按4个root重采样的速度增益95%区间+0.304%～+1.311%。全部数值检查通过，但不满足事前规定的至少10%头部节时门槛，因此拒绝性能晋升。

若这个小幅head节时在相同工作量下完整兑现，按本轮采样target kernel口径一阶折算也只有约0.084%的target kernel累计时间；这不是实际decode提升。候选还额外保留1.27GB完整权重副本，准备/验证耗时419.152ms。没有上线、完整decode新成绩或实际省内存结论；不为这个微小信号继续扩大实验。

## 正确性与常驻证据

每条采样前单独建立所需串行target前缀，对照完整token、所有已提交logit/特征和最终live state；三个normal/profile运行的预算/提交轨迹也完全相同。布局候选另外核对32份真实矩阵输出，全部精确一致。

同一PID、load_count=1、同一target/draft对象；completed_jobs100→102、failed_jobs保持5，两个作业执行正常结束，布局候选的REJECT是性能决策而非运行失败。结束READY，GPU allocated回到原常驻快照77,561,180,160 bytes，转置副本未留在默认模型中。

归档SHA256 `6c9a9026c2fb3e437d856e7718ef51baac75dcd644f3953f88ae30457222ecef`，166文件逐一核对，含101个固定上游源码文件与4份原始trace。原始数据在Git忽略的`results/raw/e30/final/worker`。导出器复查所有运行源/输入封存、原始trace指纹、真实kernel分类、计时顺序、微基准汇总、拒绝门槛与resident身份；公开JSON不包含私有路径、对象ID或原始trace。历史`draft_context.py`与本地仅相差一个文件末尾空行，哈希复核使用归档版本。

```bash
python3 scripts/e30_report.py results/raw/e30/final/worker results/e30
python3 -m unittest discover -s tests -v
```

本机210项测试，195通过，15项缺少可选PyTorch跳过。新增trace计数测试检查父级annotation不重复计入、范围外工作不混入以及区间并集不重复计算重叠。

后续仍应围绕占当前target内核时间约68%的专家与量化投影，先识别可减少的实际工作/数据搬运，再选择一个结构候选。当前数据没有证明具体内核的带宽/算力瓶颈，也没有给出可信的剩余提速百分比。既有量化、线性预算及Graph保持冻结，不开新训练或量化分支。
