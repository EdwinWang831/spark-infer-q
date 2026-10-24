# E40：自主 Prefill 模块及原生请求接入

2026-10-24。**首个自主管理 prompt 执行组织的模块已落地，显式接入 target-only 和动态
DFlash 请求路径。** 不再需要 E36 的私有前向源码克隆，投影/GDN/MoE 仍保留 E36 A/B/C
算术；没有新增 CUDA 内核或替换默认服务。外部 TensorFold 算子和加载器仍是依赖。

[事前方案](E40_PREFILL_EXECUTOR_PLAN.md) · [接口与用法](E40_PREFILL_EXECUTOR_USAGE.md) ·
[公开证据](../results/e40/2026-10-24-prefill-executor.json)

## 执行边界

- `spark_infer_q/prefill.py`：拥有分块、单流串行工作区、结果生命周期和失败隔离。
- `spark_infer_q/prefill_tensorfold.py`：拥有层顺序和 GDN prompt 状态推进，适配外部算子。
  浅拷贝权重对象并包装 MXFP8 矩阵的 prompt 入口，选择同一矩阵的 decode 算术；共享张量
  不复制、不改写。不读取执行上游源码、不修改上游全局 dispatch。
- GDN 保留新建整块 scratch（E37 未证明池化收益）；F32 MoE slots 同时保留原 shared
  expert 算术。BF16 激活和 BF16 KV 不变。
- `KaifraEngine.from_native` / `speculative_engine` 新增可选 `prefill_executor`。
  调度器发来的 prompt plan 由其处理，decode/验证继续沿用原 target；默认仍为 rows8。
  旧逐行 logits observer 与此末行 logits 接口显式互斥。

目前仍借用 TensorFold State/Buffers、stage、attention/PLE/HC/MoE/commit 和 CUDA 内核。
层顺序/GDN 状态推进的上游来源及 MIT 许可见 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)。
这是执行层自主化的第一步，不宣称整个引擎独立或原创所有算法。

## 数值、状态和生命周期

同一常驻完整模型，capacity1024。七条 E36 冻结前缀（4/64/64/512/512/128/384 token）
全部通过：末行完整 logits、五层全部 DFlash 特征、有效 GDN/QSA/PLE 状态精确一致；各自
续接八步原始 decode，56 步 logits、特征和有效状态精确一致。输入 hash 与原 E36 作业核对。

额外用 129 行工作区将 512-token 输入拆为 `[129,128]`、`[129,126]` 两次调用，验证
非零位置续填、短尾拼接及最终状态。独立新 state 关闭 capture 后，logits/状态仍一致。
同一工作区在请求间复用；原生请求取消后重新使用同一 ID 通过。

初始原生 API 接线检查为英文/中文 × target-only/DFlash × 两模式 × 两次，共16次请求，
输出 token、位置和接受/宽度轨迹一致。随后独立预热请求测试再比较最终完整有效状态。
这些是 token API 检查，没有新增 HTTP、随机采样或任务质量结果。

## Prefill 本体配对计时

英文64、代码512、资料512，三条输入各四轮 AB/BA。每模式4,352 prompt token；每例两臂
各预热一次后计时。权重/缓冲准备在外，state 创建、stage、48层、五层完整特征、commit、
末 token 准备与同步在内；输出和有效状态对照在计时外。

| 输入 | rows8 平均 wall | 自主模块平均 wall |
| --- | ---: | ---: |
| 英文64 | 577.761 ms | 169.843 ms |
| 代码512 | 4,770.190 ms | 462.483 ms |
| 资料512 | 4,821.838 ms | 478.249 ms |

总 wall **40,679.160 → 4,442.299 ms**，总 token / 总 wall 为 **106.984 → 979.673 tok/s**，
约 **9.157×**。这是 E36 大块执行收益的模块化复现，不是新内核再叠加9倍收益；任务集合
与 E36/E37不同，不能用979.673和历史汇总直接比较增量。

## 完整请求配对计时

使用真实 DeepInfer C++ Scheduler，target-only 为 NEXTN0，动态 DFlash 为 NEXTN7，分别
创建上下文。两模式均 logical capacity768、physical capacity1024、token budget512、
greedy、输出上限24。Target 是 **eager WideTarget**，草稿为现有 K2 adaptive provider；
没有切到 Graph/E33/W8 优化配置，也没有与 TensorFold 原生 MTP 比较。

英文/中文输入经过模板后均23 token；code512复用受控 E36 重复文本并截断的512 token，
不是自然任务质量样本。每例每臂先预热一个完整请求，之后四轮 AB/BA；共12次预热、
48次正式请求，所有请求均输出24 token。权重、provider/engine构建在计时外；从 submit
到 drain 完成及最终 GPU 同步在内，包含 prompt、首轮草稿、动态预算、验证和提交。

下表为每例四次均值；TTFT 来自原生 API 首输出事件，没有 HTTP、客户端分词或网络时间。

| 路径 / 输入 | TTFT rows8 → 自主模块 | 完整请求 rows8 → 自主模块 | 请求耗时减少 |
| --- | ---: | ---: | ---: |
| target / 英文23 | 202.23 → 112.64 ms | 1,151.67 → 1,058.22 ms | 8.11% |
| target / 中文23 | 214.14 → 114.64 ms | 1,187.28 → 1,075.49 ms | 9.42% |
| target / code512 | 4,620.25 → 440.44 ms | 5,590.36 → 1,413.67 ms | 74.71% |
| DFlash / 英文23 | 227.07 → 128.28 ms | 702.25 → 600.36 ms | 14.51% |
| DFlash / 中文23 | 229.03 → 127.68 ms | 1,321.31 → 1,219.70 ms | 7.69% |
| DFlash / code512 | 4,592.79 → 439.39 ms | 4,838.86 → 683.77 ms | 85.87% |

每例所有预热/计时请求的输出 hash、最终有效状态和输出位置一致；DFlash 的接受量、观察量
与验证宽度轨迹也一致。收益主要在 prompt/TTFT；不宣称新增 decode 吞吐提升。三条工程
输入及四次重复不能给出普遍性能/质量结论，更不能证明领先最新 TensorFold Zig 引擎。

## 失败记录、审计与复现

- 330首轮接线测试混用了 target-only / NEXTN7 上下文，被配置检查拒绝；334按两种上下文
  分开后通过。失败发生前的检查不当作完整请求成功证据。
- 提交时漏写 phase 使336/340重复执行验证；不计入正式请求统计。342在执行前取消，
  提交器已改为必须显式指定阶段，并增加拒绝默认阶段的测试。原始记录均保留。
- 344完成预热请求对照。346审计把原有 `commit @torch.no_grad()` 包装误认成非原始函数。
  348只读定位；350验证内部函数代码与冻结源码编译结果一致，同时保留原 no_grad 包装，
  正确通过。没有为通过审计而恢复/替换任何上游函数，也未重跑 GPU 性能来追逐结果。

最终GPU审计确认原环境和Gloo释放，审计时库文件与334/344实际执行源码一致，仍为同一worker319、
load_count1、同一 target/draft 对象。请求作业前 allocated **78,402,499,584 bytes**，最终
**77,829,615,616 bytes**，不高于该基线。验证作业更早的基线为77,762,506,752 bytes；期间
有其他工作区串行任务，故不把基线变化或分配下降解释成压缩收益，也不把 reserved 当峰值。

最终原始封存 **641文件、1,194,886 bytes**，SHA256
`3727fbf6e1641a9aa4ceab4bcaea8ca969ff8cb7f35b518a5edff11d665b7715`。
此前包含失败审计的封存也保留，没有覆盖。

```bash
python3 scripts/e40_export.py \
  --raw results/raw/e40/2026-10-24-final-v2/worker \
  --out results/e40/2026-10-24-prefill-executor.json
python3 -m unittest discover -s tests -q
```

GPU审计后，代码复核补上关闭阶段同步失败的隔离：此前成功执行过的executor若在close
遇到fence异常，也必须禁止继续advance，并保留工作区以便安全处理。新增CPU故障注入
测试通过。只修改close的异常分支；成功路径、advance和全部数值provider未改，未重复GPU
计时。导出器要求当前源码恰好等于封存版加这一处明确变更，其他库文件逐字节相同，并在
公开JSON记录测量版/当前版hash，避免把后加的防护说成已经过GPU故障测试。

本地317项测试：298通过、19项可选依赖跳过。默认服务、4K+、Graph请求组合、更大的
prefill tile和新专家内核仍待后续独立验证；本轮不推送或合并main。
