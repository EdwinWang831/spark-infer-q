# E2：最小请求执行器、独立 GPU 数值参考与局部计时

**已把真实 GDN/QSA block 接到自有的最小 target-only 请求执行器，并完成 GPU 验证。** 最终 20 项记录全部完成：16 项请求/状态检查、2 项独立数值诊断、2 项局部计时及输出检查。状态与原生执行的严格对照逐 bit 一致；独立 PyTorch GPU 参考存在数值差异。本轮没有完整模型质量或速度结论，也没有接入 DeepInfer C++ scheduler。

最终运行：2026-10-20 09:15:43–09:16:15 America/Los_Angeles。完整结果和源码/扩展摘要见 [executor JSON](../results/e2/2026-10-20-executor.json)。本地标准库测试共 23 项通过，其中新增执行器测试 16 项。

## 实现与边界

- [Executor](../spark_infer_q/executor.py)：请求身份及 slot 代次、显式行位置、prepare/dispatch/finish、取消与排空、完整 checkpoint/restore、事务回滚。逻辑位置仅在设备完成后发布；取消后禁止旧结果或旧 checkpoint 作用于新请求。
- [真实 block backend](../spark_infer_q/tensorfold_block.py)：复用外部 TensorFold 的原生 forward/commit 与压缩权重，按请求压紧交错输入，计算后散回原行；padding 不进入模型。独立 stream 与输入生命周期、输出消费依赖显式关联。
- Decode CUDA Graph 的 key 包含请求身份/代次、窗口长度、state 版本、capacity 和 GDN bank parity；commit 在图外执行。扩容、恢复和释放前使相关 graph 失效，并保留图引用的状态及 tensor。图缓存上限 8。
- 默认最多 2 个请求、4 个显式 checkpoint、context 上限 64、每请求窗口最多 32 行、联合有效输入最多 64 行。每次 dispatch 额外备份完整已提交前缀；这有复制和内存成本，不能直接推广到大 context 或生产调度。
- 真实 layer 0/GDN 与 layer 3/QSA **独立运行**，没有串成相邻层。输入是固定种子的合成 BF16 hidden states，宽度 10240；每层自己的真实 NVFP4 routed experts、MXFP8 投影和共享专家均参与执行。KV 使用 BF16，MTP 关闭。

沿用上一里程碑的 87 个固定外部源码文件和 6 个已构建 SM121 扩展，逐一检查源码摘要并记录二进制摘要；没有重建、修改外部 forward 或扩展源码。checkpoint revision 仍为 `b9ce18a6887e18c089fdea8509a33e63429a3300`。

## 每层 8 项状态/生命周期检查

| 检查 | 真实 GPU 观察 |
| --- | --- |
| 两请求交错位置、不同前缀及 NaN padding | prefill 后联合 decode，散回行输出和各自状态与独立原生执行逐 bit 一致；padding 为零 |
| in-flight cancel→drain→slot 复用 | 排空前槽位不可用，取消行输出被抑制；复用代次增加，旧完成/快照被拒绝，新请求输出正确 |
| 覆盖已保存 KV prefix 后 restore | 先保存空状态和 5 行前缀，再恢复空状态并覆盖 prefix；完整快照恢复后重放输出/状态逐 bit 一致 |
| capacity 8→16 与图失效 | state layout version 更新，旧图失效；输出/状态与足够容量基线逐 bit 一致 |
| 连续 6 次单 token 图解码 | 图外提交、设备位置持续更新；GDN 2 次捕获/4 次命中，QSA 1 次捕获/5 次命中；输出/状态与 eager 逐 bit 一致 |
| prepared cancel 与全 padding batch | 未启动的取消计划被拒绝；全 padding 不调用模型 |
| producer/backend/consumer 三条 stream | dispatch 后释放输入，finish 后由 consumer 直接复制输出，仅等待 consumer；结果逐 bit 一致 |
| 已发出 CUDA 工作后的 host 异常 | 同步后恢复整个已提交前缀，逻辑位置不变；再次执行正确；没有注入 CUDA device fault |

Host 测试另外覆盖部分请求取消、同名请求清理、metadata 原子校验、错误推进量、checkpoint 预算、disjoint prepared work 串行 dispatch、失败 fence 隔离等。它们与真实 GPU 用例分开计数，不能替代多线程压力、设备故障恢复或 DeepInfer 调度器验收。

## 数值参考与质量分开判断

[PyTorch reference](../scripts/e2_block_reference.py) 新增 CUDA 模式，默认 CPU 模式保留。它使用独立的 FP32 matmul、recurrence、attention 和 routing，匹配存储权重及 BF16 舍入边界；专家按需逐个解码，并与前轮共用已检查的 CPU NVFP4 解码器。不是完全独立的权重格式实现。

实验显式把容器原有的 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` 覆盖为 **0**，同时 `allow_tf32=False`、matmul precision `highest`，结果中记录并断言这些设置。FP32 累加顺序仍可与 CUDA 内核不同。

| 17 行 prefill 对照 | GDN layer 0 | QSA layer 3 |
| --- | --- | --- |
| 最终 block 输出相对 L2 差 | 0.09776% | 0.13366% |
| attention 分支相对 L2 差 | 0.17731% | 0.22341% |
| MoE 输入相对 L2 差 | 0.20047% | 0.31352% |
| Top-10 专家集合不同的行数 | 0/17 | 0/17 |
| 专家排序不同的 slot 数 | 2/170 | 8/170 |

GDN recurrent/conv 相对 L2 差分别约 0.06784%/0.02506%；QSA K/V 约 0.06496%/0.05536%。给 PyTorch router 同一份原生 MoE 输入，两层专家集合一致；给 stable Top-10 同一份原生 logits，选择逐项一致。执行器输出与直接调用同一原生 block 的输出逐 bit 一致。

前轮 CPU 全路径的 GDN 第 6 行专家集合分歧继续保留在历史报告中。本轮在相同输入的独立 GPU 参考上没有再观察到集合分歧，**不能据此认定所有路由边界差异已解决**。

独立数值诊断中的 `PASS` 表示完成有限值比较并输出差异，1% L2 仅触发 review flag，不是质量或等价性的验收线。没有为强求一致而改变原生算法；也不能凭更大或更小的误差声称模型更聪明。是否更优秀，需要完整 logits、真实任务、采样及质量评测。

## 局部性能

同一层、同一权重、相同输入序列，比较本执行器的 eager 与 graph 路径。各先 prefill 4 行、warmup decode 4 步，再交替顺序采样 24 步。计时包含请求 metadata、输入检查、完整状态备份、forward、commit、输出 masking 和完成等待；每步两路径输出逐 bit 一致。

| 单层执行器 wall time | GDN eager | GDN graph | QSA eager | QSA graph |
| --- | --- | --- | --- | --- |
| 中位数 ms | 1.391 | 1.048 | 1.181 | 0.851 |
| p95 ms | 1.555 | 1.140 | 1.371 | 0.926 |
| eager / graph 中位耗时 | 1.327× | — | 1.388× | — |

这是小 context、单请求、已加载真实单层的局部 wall time，分别减少约 24.7%/27.9%。不含权重加载和参考解码；没有整模、多轮统计置信区间、长上下文或服务吞吐比较，不能把上述倍率外推为完整模型加速。

## 内存、隔离与复现

分别加载一层时 Torch allocated 峰值约 **1.8388 GiB / 1.8674 GiB**，reserved 约 **1.9688 GiB / 1.9316 GiB**，包含本轮独立参考和备份的开销，不是纯推理或整模峰值。独立容器仍限 2 CPU/6 GiB RAM，Torch allocator 限 3 GiB；本轮容器生命周期含预跑和最终运行，cgroup peak 为 2,753,249,280 bytes，max/OOM/OOM kill 都为 0。cgroup 和 CUDA allocator 记账范围不同。

结束后实验容器均停止。现有服务 health 正常，PID、启动时间、重启次数以及配置、镜像、参数、环境和挂载定义不变；没有测量实验期间服务的瞬时延迟。

首轮 layer 0 共 9 项、随后两层共 18 项均完成；代码复核后增加输入 `record_stream` 与输出 consumer 等待关系，并新增跨 stream 用例，再跑最终 20 项。预跑没有发现失败，也不把这些重复结果累计为最终用例数。原始报告、runner 和日志留在私有实验区；公开 JSON 的 harness SHA 与本次提交逐文件核对。

复现入口：[runner](../scripts/e2_executor_smoke.py)、[环境入口](../scripts/e2_run_executor.sh)、[权重/配置映射](../scripts/e2_block_support.py)。在前轮隔离实验目录旁建立 `e2-executor`，放置本项目 `spark_infer_q/` 包及上述 runner、shell、support、reference 文件；沿用 `e2-block` 和 `e2-moe` 的已验证源码、权重和扩展，运行 shell 即可。无需额外安装包或重建内核。

## 下一阶段

接 DeepInfer 调度/metadata 边界，同时推进完整权重加载、ngram 放置和运行峰值核对，再形成 target-only 的真实 token→logits 路径。完整 logits 与任务质量分别评估，允许经证据证明的优化优于基线。随后接原生 Qwen MTP 和 accepted-prefix 状态契约；整体工作继续覆盖算子、模型、调度和内存。
