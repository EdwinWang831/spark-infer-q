# E37：prefill 的真实耗时与 scratch 复用结论

2026-10-24。结论：**不采用 GDN scratch 复用；下一项优先研究 prefill 路由专家计算。** 本轮完成事前方案、真实模型配对计时、独立剖析与收尾审计，没有新增已证实的速度收益。

事前[方案](E37_PREFILL_PROFILE_PLAN.md)、[公开结果](../results/e37/2026-10-24-prefill-profile.json)。沿用 E36 已执行的投影、GDN、F32 MoE 算术和完整五层草稿特征；没有修改权重、CUDA 内核或默认服务。

## 单变量结果

同一常驻模型，四条已封存输入按 code512 → english64 → mixed512 → JSON128 执行，覆盖大→小→大→中。每条六轮 AB/BA 交替，共 48 次未开启 profiler 的 prompt 计时。每模式 7,296 prompt tokens，吞吐使用总 token / 总 wall。

| 模式 | 总 wall | prompt tok/s | 决策 |
| --- | ---: | ---: | --- |
| E36 fresh scratch | 7,889.764 ms | 924.743 | 保留 |
| 单 stream、单 owner 复用 scratch | 7,970.096 ms | 915.422 | 不采用 |

复用模式总 wall **增加 1.018%**，六轮只有一轮略快（0.015%）；四条逐例结果如下。它没有达到事前的 wall 至少降低 2%、至少 5/6 轮更快、每例不能慢超过 2% 的门槛。这个结果不支持“复用一定变慢”的普遍结论，但足以拒绝此次候选，不为追逐微小波动追加长测。

| 输入 | fresh 平均 wall | pool 平均 wall | pool wall 变化 |
| --- | ---: | ---: | ---: |
| code512 | 453.995 ms | 459.398 ms | +1.190% |
| english64 | 161.266 ms | 170.370 ms | +5.645% |
| mixed512 | 483.835 ms | 484.381 ms | +0.113% |
| JSON128 | 215.865 ms | 214.201 ms | −0.771% |

pool 预备一个 512 行、10,682,368-byte scratch 和 64/128/512 行视图，供 36 个串行 GDN 层复用，替代每层构造。预备和 JIT/预热在计时外；State 初始化、输入 stage、48 层、全部特征捕获/拼接、commit、末 token 与同步在计时内，输出 hash/状态比较在计时外。复用路径共调用 1,152 次，符合正确性、预热和正式计时的预定次数。

这里的 924.743 tok/s 不能与 E36 七条输入的 889.218 tok/s 直接比较，工作负载不同；它也不是完整请求或 decode 吞吐。

## GPU 内核时间分布

另取 code512、mixed512，各运行一次 CUDA-event 阶段诊断及一次 CUPTI trace。下表仅统计 prompt 范围内真实 `cat=kernel` 的累计时间，排除父级 annotation；不包含正式 A/B 计时。

| 内核家族 | code512 时间 / 内核累计占比 | mixed512 时间 / 内核累计占比 |
| --- | ---: | ---: |
| 路由专家 | 210.584 ms / **52.49%** | 248.140 ms / **54.86%** |
| 量化矩阵投影 | 58.298 ms / 14.53% | 63.450 ms / 14.03% |
| GDN 递推 | 47.386 ms / 11.81% | 51.018 ms / 11.28% |
| BF16 矩阵投影 | 30.082 ms / 7.50% | 31.936 ms / 7.06% |
| attention 内核 | 1.506 ms / 0.38% | 1.622 ms / 0.36% |
| 其他内核 | 53.332 ms / 13.29% | 56.130 ms / 12.41% |

每条 trace 有 2,052 次 kernel 调用。其中路由专家各 96 次，包含 48 次 gate/up 和 48 次 F32 down：code 分别 122.518 / 88.066 ms，mixed 分别 147.528 / 100.612 ms。**专家计算是这两条输入最大的已测 GPU 成本。** 这不是对所有语言、长度或 decode 阶段的结论。

| 剖析范围 | code512 | mixed512 |
| --- | ---: | ---: |
| profiler prompt wall | 439.757 ms | 489.256 ms |
| kernel 累计时间 | 401.187 ms | 452.296 ms |
| kernel 活动区间并集 | 401.187 ms | 452.296 ms |
| kernel + memcpy/memset 活动区间并集 | 401.867 ms | 452.990 ms |
| prompt wall 中无上述设备活动的区间 | 37.890 ms | 36.266 ms |

这些 trace 中 kernel 重叠可忽略，但分析器仍用区间并集计算活动时间。最后一行包含主机提交、同步、剖析扰动等，不等于可消除的 Python 时间，更不是 SM 利用率或 DRAM 带宽。

## 主机与阶段时间的边界

CUDA-event 阶段诊断把嵌套子区间从父段拆出，避免把投影重复计入 GDN/attention。主要 event 区间：

| 阶段 | code512 GPU interval | mixed512 GPU interval |
| --- | ---: | ---: |
| MoE 整段 | 228.565 ms | 259.979 ms |
| forward 矩阵投影 | 68.865 ms | 69.012 ms |
| hyper-connection | 57.000 ms | 57.736 ms |
| GDN 其余工作（包含递推） | 49.089 ms | 48.682 ms |
| 输入 stage | 21.880 ms | 26.780 ms |
| State 初始化 | 9.116 ms | 9.545 ms |

event 区间仍包含提交间隙，不能替代纯 kernel 时间；两种诊断也是分别执行，数值不必逐项相等。MoE 的主机区间为 147.116 / 166.909 ms，末 token 准备的主机区间为 123.739 / 134.778 ms：其中有等待设备完成的时间，**不能相加后声称都是 Python 开销**。CUDA API 时间也会与设备执行重叠。

## 正确性、恢复与证据

- 四条输入的完整末行 logits、全部 1,216 行五层拼接特征，其 hash 与封存 E36 一致；fresh/pool 所有有效 state 逐位一致。
- 每条再运行八步原始 decode，32 步全部 logits、特征与有效 state 精确一致。每次共用 decode scratch 前已即时 commit，避免别名覆盖。
- 48 次正式 prompt 都在计时外校验输出 hash 和完整最终 state；四次诊断也通过相同门槛。
- 270 测量作业、271 独立收尾审计均 PASS；同一 worker / target / draft 对象，`load_count=1`，未重载或清空缓存。收尾 allocated 回到实验前 **77,728,952,320 bytes**，源和原始 forward 已恢复。
- 本地导出器重验两个 stage manifest、wrapper、作业 hash、E36 输入/源 hash、外部源 seal、trace SHA256 与重算统计；拒绝缺失计时、状态失败、输出漂移和不一致汇总。原始 trace、token IDs、对象地址及私有路径不进入公开结果。

原始证据归档 134 文件、1,059,289 bytes，SHA256 `e7446dfc85d3912b5143120706cfe6168d5e760b206cc0b03868980aceda3cbc`，保留在 Git 忽略的本地 raw 目录。

本地测试共 258 项：242 通过、16 项可选依赖跳过。其中新增八项覆盖 scratch 所有权/视图、trace 区间与去重、加权统计、决策门槛及数值证据拒绝条件。另在临时归档副本中分别篡改 wrapper、执行源、trace、worker 身份，导出器全部拒绝；提交中的执行源 SHA256 与两个远端 stage manifest 一致。

## 下一项实验

优先检查 **prefill 专家内核如何复用同一专家的多行输入与权重**，先获取真实路由的每专家行数、tile 尾部浪费与 gate/up、down 的分别成本，再选择一个同算术候选，进行完整 prompt A/B。现有源已经使用 `PREFILL_TILE=16`，其注释记录 64 更慢；不能把盲目加大 tile 或重复 E10 warps 调参当作新收益。

继续保留 BF16 激活、F32 down 输出和现有 reduction 顺序。任何改变算术的实现须先过完整 logits、草稿特征、state 与后续 decode 检查；组件提速仍需完整 prompt 计时支持。GDN 递推、hyper-connection 和输入准备保留为后续候选。本轮没有执行上述专家候选，也不预报提速倍数；另一工作区继续负责 K4 动态 DFlash / 服务路线。

复核公开结果：

```sh
python3 scripts/e37_export.py --raw results/raw/e37/2026-10-24-final/worker --out results/e37/2026-10-24-prefill-profile.json
python3 -m unittest tests.test_prefill_scratch tests.test_e37_accounting tests.test_e37_export -v
```

全部证据仅覆盖四条 64–512-token 受控前缀、容量 1024、greedy token API；没有新增随机采样、4K+ 长上下文、任务质量、HTTP、完整 DFlash 请求或默认引擎集成结果。
