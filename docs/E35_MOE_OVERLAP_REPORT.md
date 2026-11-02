# E35：共享/路由专家双 stream，未形成可靠的 MoE 提速

2026-10-23 UTC，`REJECT_E35_MOE_COST`。完整MoE的热复用耗时少3.042%，128MiB缓存扰动下仅少0.500%，后者速度增益95%区间跨零，未过[事前方案](E35_MOE_OVERLAP_PLAN.md)的5%组件节时门槛。全部正确性检查通过，但不进入完整decode长测、不修改默认路径。没有新增decode token/s。[完整脱敏计时与时间线汇总](../results/e35/2026-10-23-moe-overlap.json)。

## 测的结构改动

源码确认当前NVFP4已按expert plan聚合同专家的行，不能把“加grouping”重新算作优化。本轮也没有找到并宣称省掉大块权重读取；测试的是把每层已有的独立shared expert分支提前，与router/select及routed expert分支并发执行。

辅助stream持有shared gate/up、原SwiGLU和shared down；主stream保留router、select、routed gate/up、routed down。输入就绪后fork，原slot写回前join，临时张量通过record_stream保留到主stream复制完成。routed down跳过shared slot，故不读取尚未提交的shared activation。使用原扩展/权重，保留每个核的BF16/FP32算术、路由ID/权重、skip规则及最后逐slot合并次序。

全部操作在实例私有runner和克隆的forward模块中完成，未替换全局dispatch，也未修改另一个工作区K4。原target为E33已Graph化/已筛选输出头，草稿沿用E28 W8/Q4，预算是E17 NumPy与E29冻结费用；没有重新训练或校准费用。

## 正确性与真实输入

- 原checkpoint第0/23/47层，1～8行、连续/带stride输入、BF16/FP32 slot输出：共96项，对照完整activation、slot输出、router logits、pick和wts，候选eager/Graph重放均精确一致，尾行哨兵不变。
- 辅助stream从另一个独立caller stream调用，再次核对全部上述输出，通过。
- 四条E33旧校准root、各前2轮、三层，共24份真实MoE输入；所有模式的Graph输出与原前向逐位一致。8次诊断遍历的完整logits、五处DFlash特征及提交live state与冻结target一致。

诊断遍历是在所选层插入隔离回放后继续原前向，验证采集/回放及逐组件替代一致性；不是48层全部替换为并发Graph后的完整生成验证。该验证只有组件成本过门槛才会执行，本轮没有执行。

## 完整 MoE Graph 配对成本

每个真实输入分别捕获串行/双stream Graph，包含router/select、两个expert分支及全部slot写回。两种缓存条件各6轮AB/BA，热复用每区间10次、缓存扰动每区间1次，共576个CUDA-event区间、3168次计时MoE调用。计时包含Graph依赖等待，排除capture、正确性检查和扰动本身；缓存扰动不等于证明DRAM缓存全部驱逐。

| 24份输入各组中位数之和 | 串行 | 双 stream | 耗时减少 | 速度增益root95% |
| --- | ---: | ---: | ---: | ---: |
| 热复用 | 14.416435 ms | 13.977856 ms | 3.042% | +2.555%～+4.362% |
| 128MiB缓存扰动 | 15.480352 ms | 15.402928 ms | 0.500% | −0.596%～+1.566% |

等价组件速度增益分别为+3.138%和+0.503%；扰动条件英文代码root慢1.006%。不使用热复用结果替代原定扰动条件的门槛，也不将组件变化外推为decode收益。这是四条已用校准root的诊断，不是独立测试集或新TensorFold对照。

## 时间线说明了什么

四条root首轮第0层，各模式额外运行一次CUPTI诊断，共8份trace。导出只读取真正`cat=kernel`事件，排除CUDA user annotation父级与scope外事件；GPU累计时长、活动区间并集和两家族交集分开计算，剖析不进入上表计时。

每份trace均19个真实kernel事件。串行的expert/projection位于同一stream；候选确实使用不同stream，但shared量化投影与routed专家重核的交集分别为 **0、0、0.128、0微秒**，没有形成可观的重核重叠。

串行profile中shared两个矩阵核合计约12.6～14.6微秒，routed两个专家核约293.8～1012.4微秒。shared分支通常在router/select完成、routed重核启动前就结束，主要隐藏的是较短的前段工作。这些值有profiler扰动，只用于解释该候选的时间线，不能当正常decode分段或无条件理论上限。

因此，“有两条stream”不能等同于“两个主要重核的耗时被合并”。本候选不改变实际权重体积和有效矩阵计算，现有数据也没有测DRAM/SM硬件计数器，不能进一步宣布带宽饱和或专家内核已无空间。大块GDN等量化投影不属于本次shared-expert分支，不能据此否定全部投影优化。

## 常驻、归档与复现

`189-e35-moe-overlap.json`和`190-e35-audit.json`作业均PASS；前者的REJECT是成本决策。worker/target/draft对象不变，load_count=1；completed_jobs141→143，failed_jobs保持9。最终READY，allocated回到77,594,734,592 bytes，无重载/重启/empty_cache。

原始归档211文件，SHA256 `fff555b54ecae003a3153954720c63d6db0d0fd14e464063c3657424163944cf`，在Git忽略的 `results/raw/e35/final/worker`。导出器逐文件验证source/input seal、外部固定源码、实际runner与本地源、计时完整性/门槛、8份trace指纹和时间线、对象与最终内存。公开JSON不含原始提示词、权重、私有绝对路径或对象ID。

```bash
python3 scripts/e35_export.py --raw results/raw/e35/final/worker --out results/e35/2026-10-23-moe-overlap.json
python3 -m unittest discover -s tests -q
```

本机220项测试，205通过、15缺少可选PyTorch跳过。新增验证确保热缓存收益不能覆盖扰动条件失败、timing/层缺失或重复被拒绝、错误调用次数/数值失败/NaN计时被拒绝，以及时间线不重计父级annotation或家族内部重叠。

本轮关闭这条双stream候选。后续若追求更大收益，需针对占时更大的routed专家/GDN投影改变实际工作或权重复用，或以新的独立协议测试更少的验证工作量；不能沿用本轮结果承诺具体收益。
