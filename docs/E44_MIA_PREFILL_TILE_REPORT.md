# E44：Mia共享输入专家tile接入自主prefill

2026-10-24启动，2026-10-25收尾。**完整模型、请求和最终公开接口均通过，显式启用。**
在我们现有chunk1024/Graph/E33/W8路径上，两个长输入的完整prefill耗时减少9.36%/5.45%，
实际TTFT减少10.60%/3.76%。这是借鉴并适配Mia机制的增量，不是原创该CUDA算法或
超过最新Mia原生引擎的排名结论；decode没有新增已证实收益，默认路径保持不变。

[事前方案](E44_MIA_PREFILL_TILE_PLAN.md) · [公开证据](../results/e44/2026-10-25-mia-prefill-tile.json) ·
[使用接口](E40_PREFILL_EXECUTOR_USAGE.md)

## 机制与来源

采用Mia仓库commit `656879fd752749bbd11a6e1a065a326c00940911` 应用patch后的
`zig/kernels/cuda/fn_experts_tile.cu`，源SHA256
`d09b7a01489679580d1083d2b7ff3a9d4d893d0ef8621ed8b84514689dbb0393`。
保留三个原device body：gate/up RT4/S4，F32/BF16 down RT2/S3，plan64、4-warps，
rows>=1024启用。每warp处理不同32列输出，CTA共享输入片段并异步预取，同时用同一
权重片段服务多组行。host launch、私有dispatch及与我们prefill模块的连接由本项目适配。
来源和MIT版权保留于[第三方说明](../THIRD_PARTY_NOTICES.md)。

E38/E39共享的是同一输出列块的权重；这里共享输入并让warps处理不同输出列，执行结构
不同。E43证明仅搬指令交错没有收益，本轮不将两个结果混用。没有tile/warp/stage参数扫描，
没有修改原权重、全局dispatch、激活/KV精度、scale-FMA顺序或F32 down。

## 组件筛选

56项人工路由检查：rows1/15/16/17/31/32/33/48/63/64/65/128/512/1024，连续/stride，
F32/BF16 down、guard、两次Graph replay及独立caller stream，完整act/y全部精确。
English64/code512/code1024/mixed1024 × 层0/23/47，完整router logits、pick/wts、act/y精确。

每份两缓存条件、六轮AB/BA；完整MoE timer含router/select/plan、路由/共享专家和slots。
主要门槛预先限定在六份1024行输入，其他六份短输入保留相同旧路径作控制：

| 1024行输入、128MiB扰动 | 原版中位数之和 | 候选 | 耗时减少 |
| --- | ---: | ---: | ---: |
| code1024/mixed1024 × 三层 | 37.014 ms | 27.159 ms | **26.62%** |

六轮汇总均更快，各例减少20.60%–34.16%，达到至少5%、至少5/6轮胜出及每例不慢超过2%
的门槛。全12份数据保留在JSON，短控制不用于选择候选。扰动不等于所有缓存完全驱逐。

资源也有代价：gate/up255寄存器/线程、64-byte stack；down128寄存器、stack0；
cuobjdump报告static shared均1024bytes、local0。host另设置dynamic shared：gate/up34,816、
down19,200bytes/CTA。不把local0解释成无任何栈访问，也未隔离各因素的硬件stall成本。

## 完整模型与请求

两臂均chunk1024、physical2048/logical1792、scheduler token_budget1024，Graph/E33原BF16
最终裁决、W8草稿MLP/Q4草稿头、BF16 attention/selector、E17 NumPy及冻结E29成本。
仅prefill专家后端不同；greedy输出cap64，首轮草稿延迟到首token后计入decode。

数值检查覆盖English64/code512/code513/mixed1024/Chinese1536：完整末行logits、全部3649行
五层草稿特征和有效GDN/QSA/PLE状态逐位一致。每条width1–8，奇数完整/偶数半提交，
40窗口、185行完整logits、16种width/parity桶实际Graph replay，无fallback。
这些检查包含候选替换全部48层，区别于仅选定三层的组件筛选。

三条prompt每臂预热一次，四轮AB/BA，共24次正式prompt；state创建、stage、48层、特征、
末行head、commit与完成同步在内，workspace/profile初始准备与正确性比较在外：

| 输入 | 原版 | 候选 | 耗时减少 |
| --- | ---: | ---: | ---: |
| English64 | 156.53 ms | 153.34 ms | 2.04%（同路径控制波动） |
| Code1024 | 729.08 ms | 660.84 ms | **9.36%** |
| Mixed1536 | 1,149.74 ms | 1,087.04 ms | **5.45%** |

两条长输入相同10,240 tokens/臂，总wall7,515.29→6,991.50ms，减少6.97%。1536分成
[1024,512]，尾块保留旧专家路径，因此不能把26.62%的组件收益直接当作整体收益。

完整优化请求为每例每臂一次预热，再四轮AB/BA：8预热+32正式请求。所有输出tokens、
事件位置、全部prompt/decode接受/宽度/概率/head轨迹、已提交特征上下文及最终有效状态
一致；正式请求没有新capture、head JIT形状或Graph/head fallback。

四次均值，单位毫秒：

| 输入 | 输出tokens | TTFT原版→候选 | 完整请求原版→候选 | 请求耗时减少 |
| --- | ---: | ---: | ---: | ---: |
| English23 | 64 | 108.42→107.74 | 1,121.06→1,130.00 | −0.80% |
| Chinese23 | 51（EOS） | 111.50→109.53 | 1,784.34→1,799.37 | −0.84% |
| Code1024 | 64 | **749.10→669.66** | 1,999.10→1,908.86 | **4.51%** |
| Mixed1536 | 64 | **1,182.54→1,138.08** | 1,833.82→1,806.77 | **1.47%** |

TTFT分别减少10.60%/3.76%。短输入不启用新tile，没有系统性收益。相同972输出tokens/臂
汇总完整wall26,953.28→26,580.00ms；这组人工工作负载不代表通用服务流量。
长文本为受控重复/截断内容，和E42任务不完全相同，不直接横比历史tok/s。

token API计时从submit前到drain后同步；TTFT为native首输出事件，不含HTTP/客户端分词。
残差decode是完整wall减native TTFT，含少量submit/事件边界开销，不是纯GPU时间。
相同956个首token后tokens耗时18,347.04→18,479.94ms，即 **52.11→51.73 tok/s（−0.72%）**。
没有新增decode提速声明。收益集中于prefill，完整请求收益随输出耗时被摊薄。

## 最终公开接口和收尾

测量版用私有provider注入候选。随后新增 `PrefillExpertTile` 与provider可选
`expert_backend` 参数，替代实验proxy接线，原backend=None路径保留。
最终生产接口另做GPU检查：code1024/mixed1536完整logits/features/state、16个Graph窗口，
一对code1024完整请求，以及长输入的waiting/prepared-prompt/prepared-decode取消复用均精确。
本机中文SSE24token/length与取消1token/cancelled通过。该轮为功能验证，不冒充又一次性能测量；
HTTP没有测速或公网部署结果。所有执行模块/源码/binary链分别封存。

本轮一直复用已存在的worker46235/load_count1，区别于E42历史进程，无本轮模型重载或cache清理。
450独立审计针对最终接口通过：原算子/no_grad包装、native group/environment和stage模块释放；
**allocated 77,393,408,000→77,393,408,000 bytes**，与正式study/接口基线相同。

E43更早的77,359,853,568-byte基线仍有32MiB未解释差额，详见[E43失败记录](E43_PREFILL_PIPELINE_REPORT.md)。
本轮不清除共享cache或其他工作区对象，不将后来的稳定基线审计覆盖早期失败，也不宣称内存压缩收益。

原始归档481文件、2,363,823bytes，SHA256
`650f5192e3fac0d4c06f6bad528ed90513c91aecfd5b6a18bd60145df343e9b1`，
在忽略目录 `results/raw/e44/2026-10-25-final/`，含原Mia源/许可证、生成源、header、
两binary及资源、五阶段job/manifest、优化profile冻结数据；不含模型权重。

```bash
python3 scripts/e44_export.py \
  --raw results/raw/e44/2026-10-25-final/worker \
  --out results/e44/2026-10-25-mia-prefill-tile.json \
  --build-version build-v1 --build-job 436-e44-build-v1.json \
  --component-version component-v1 --component-job 438-e44-component-v1.json \
  --study-version study-v1 --study-job 442-e44-study-v1.json \
  --integration-version integration-v1 --integration-job 446-e44-integration-v1.json \
  --audit-version audit-v1 --audit-job 450-e44-audit-v1.json
python3 -m unittest discover -s tests -q
```

本地341项测试：322通过、19项可选依赖跳过；源/工作量/状态/最终内存证据的篡改被导出器拒绝。
范围仍为单GPU/单请求、输入≤1536、physical≤2048、greedy，prefill eager、decode Graph。
没有4K、vision、随机采样、多请求tile服务、最新Mia同负载排名或普遍质量结论。
