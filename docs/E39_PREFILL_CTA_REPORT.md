# E39：CTA 协作加载正确，但完整 MoE 成本仍未通过

2026-10-24。**不采用本候选，保留 E36 prefill。** CTA 共享权重加载把 gate/up、down 每线程寄存器降低到96、65，但完整 MoE 的扰动条件成本仍增加 **3.112%**，没有新增 prompt/decode 提速。

[事前方案](E39_PREFILL_CTA_PLAN.md)、[公开证据](../results/e39/2026-10-24-prefill-cta.json)。本轮只有一个固定设计，没有扫描 tile、warp 或流水级数；沿用 E38 封存的四条输入和层0/23/47。

## 实现与正确性

32行plan中的>16行item，由一个64线程CTA的两个warp各处理16行；压缩NVFP4权重通过cp.async进入双缓冲共享内存，再由两个warp读取。每warp独立保留原解码、scale FMA、K-block累加顺序、SwiGLU/F32 down。<=16行item仍走原单warp实现，短于256行的输入保留原MoE。两个互斥分支分别启动。

所有CTA线程使用同一个item/column索引，skip/cnt分支一致；在两个消费者均读取共享槽之后才覆盖，copy wait与CTA同步保证可见。未更改权重、BF16激活或最终slot顺序，也未修改原全局算子。

- 40项人工路由：十种行数1/15/16/17/31/32/33/64/128/512，连续/stride、F32/BF16 down；完整act/y、guard和两次Graph重放全部精确，独立caller stream通过。
- 四条真实输入×三层，共12份完整MoE的router logits、pick、wts、完整act/y全部精确。
- 观测过程在继续下一层前恢复原MoE，四条完整控制前向的末行logits/五层拼接特征hash与E36一致；这不等于候选替换全部48层后的状态或质量验证。

## 配对成本

每份输入两种缓存条件、六轮AB/BA、两模式，共288个CUDA-event区间、576次完整MoE调用。计时包含router/select/plan、路由/共享专家和写回；Graph捕获、预热、检查和128MiB扰动在外。warm每区间3次、扰动每区间1次，以12份输入各组中位数之和汇总。

| 条件 | 原版 | CTA候选 | 成本变化 |
| --- | ---: | ---: | ---: |
| 热复用 | 42.977 ms | 43.811 ms | **+1.941%** |
| 128MiB缓存扰动 | 42.713 ms | 44.043 ms | **+3.112%** |

扰动条件六轮全部更慢；code三层合计13.004→13.406 ms（+3.09%），mixed三层15.316→16.241 ms（+6.04%）。短输入控制使用相同原实现，波动约+0.006%/+0.027%，不算候选收益。缓存扰动不等于证明全部缓存被驱逐。

未达到事前“至少减少5%、至少5/6轮更快、每例不能慢超过2%”的组件门槛，因此停止，**没有运行完整prompt A/B、候选decode交接或HTTP测量**。本轮数据不能与E38的另一轮计时作直接候选排名。

## 资源信息与解释边界

| 本轮binary内核 | gate/up寄存器/线程 | down寄存器/线程 | 编译报告共享内存/CTA（gate/up / down） |
| --- | ---: | ---: | ---: |
| 原单组16行分支 | 126 | 95 | 0 / 0 bytes |
| CTA双warp分支 | 96 | 65 | 3328 / 2176 bytes |

均无local-memory spill。降低单warp累加器资源的目标实现了，但没有得到净收益。共享内存读写、CTA同步、异步复制等待以及额外分支启动都增加工作；本轮没有隔离其各自成本，也没有测硬件stall/DRAM/SM计数器，不能归因为其中某一项或宣称专家内核已经无空间。

E38、E39均未证实相应的权重复用设计有收益。下一步优先验证 E36 大块prefill接入实际请求后的首token时间及完整状态交接；若继续专家内核，应先取得同步/内存成本证据，再决定是否值得更复杂设计。

## 收尾与复核

320构建、321组件、323独立审计全部PASS，性能决策为拒绝。观测回调在finally中恢复，没有复现E38的128MiB滞留。同worker/target/draft、load_count=1，无重载或empty_cache，allocated回到实验前 **77,762,506,752 bytes**，原函数与冻结源恢复、native group关闭。未修改K4工作区。

归档174文件、973,421 bytes，SHA256 `59e0c4b704fdecf33c8e992155f9cec5d5c9dc72c0e2b30d53e3fdef67d03251`。本地导出重新验证每个manifest/wrapper、作业链、输入hash、外部源、生成源/binary、数值门槛、全部计时、资源指纹和显存收尾；原始输入IDs、私有路径、对象地址和binary留在Git忽略的raw目录。

```sh
python3 scripts/e39_export.py --raw results/raw/e39/2026-10-24-final/worker --out results/e39/2026-10-24-prefill-cta.json
python3 -m unittest discover -s tests -q
```

305项本地测试：286通过、19项可选依赖跳过。范围为受控64–512-token输入的选定组件；没有新增独立任务质量、4K+、随机采样或服务性能结论。
