# 文档索引 / Documentation index

首次阅读建议先看[中文研究导览](RESEARCH_OVERVIEW.zh-CN.md)或 [English overview](RESEARCH_OVERVIEW.md)：从 DFlash 预测器、验证预算和请求调度，直接定位实现、实验和 RSI 方案。A [CPU-only example](../examples/adaptive_budget.py) runs the released predictor and budget rule without model weights.

这里按研究主题列出全部实验（E0–E58）和运行时里程碑（K1–K4），每项一句结论。细节见各自的报告，脱敏数据在 [`results/`](../results)，按时间顺序的完整记录见 [EXPERIMENTS.md](EXPERIMENTS.md)。分支和 PR 的位置见 [REPOSITORY_STATUS.md](REPOSITORY_STATUS.md)。

每个实验通常有两份文档：`*_PLAN.md` 是运行前冻结的方案（任务、变量、通过门槛），`*_REPORT.md` 是结果，失败和被否决的结果同样保留。

**现状一栏的含义**：**默认**＝已进入默认执行路径；**可选**＝通过正确性检查，需显式开启；**未采用**＝没过事前门槛；**基础**＝功能和正确性里程碑，不是性能结论；**诊断**＝测量或剖析；**进行中**＝还在分支上。

所有数字只在各自报告的任务、硬件和计时口径内成立；不同行的 tok/s 不能直接比较，收益也不能相乘。

## 1. 基础与基线（E0–E7）

| 实验 | 内容 | 结论 | 现状 | 文档 |
| --- | --- | --- | --- | --- |
| E0 | 环境指纹与上游兼容性审计 | 静态检查通过，未执行 CUDA | 诊断 | [报告](E0_REPORT.md) |
| E1 | 构建 DeepInfer 核心，探测 SM121 内核 | 最小 GPU 检查通过；原生 DeepGEMM BF16 不支持 SM121 | 诊断 | [构建](E1_BUILD_REPORT.md) · [GPU](E1_GPU_REPORT.md) |
| E2 | 模型适配：专家、MoE、GDN/QSA block、执行器、整模加载 | 完整 target 可加载（CUDA 权重 69.85 GiB），block 与状态检查通过 | 基础 | [加载](E2_LOAD_REPORT.md) · [block](E2_BLOCK_REPORT.md) · [MoE](E2_MOE_REPORT.md) · [专家](E2_EXPERT_REPORT.md) · [原生专家](E2_NATIVE_REPORT.md) · [执行器](E2_EXECUTOR_REPORT.md) |
| E3 | DFlash2 推测解码 + DeepInfer MTPVarlen 预算 | 8 行 / 最多 7 个草稿，token 与状态和 serial 一致；按深度先验的动态预算与合适的固定长度打平（42.21 vs 42.20 tok/s） | 基础 | [宽草稿](E3_WIDE_BLOCK_REPORT.md) · [DFlash2](E3_DFLASH2_REPORT.md) · [完整生成](E3_FOLLOWUP_REPORT.md) · [预算审计](E3_BUDGET_AUDIT_REPORT.md) · [局部策略](E3_LOCAL_POLICY_REPORT.md) · [快路径](E3_DRAFT_FASTPATH_REPORT.md) |
| E4 | 短上下文同机对照 | 动态比固定六草稿 +2.12%；没有领先原生 TensorFold MTP | 诊断 | [报告](E4_MATCHED_REPORT.md) · [方案](E4_MATCHED_PLAN.md) · [TensorFold 公开数据审计](E4_TENSORFOLD_COMPARISON.md) |
| E5 | DeepInfer C++ Scheduler 驱动 Qwen target-only 请求 | 七组 GPU 场景通过；GPU 仍逐个请求执行 | 基础 | [报告](E5_ENGINE_BRIDGE_REPORT.md) · [方案](E5_ENGINE_BRIDGE_PLAN.md) |
| E7 | 充分预热后对照原生 TensorFold | 整体 45.62 vs 49.18 tok/s；8 个自然完成的任务 45.57 vs 43.58。此后没有重测 | 诊断 | [报告](E7_TENSORFOLD_MATCH_REPORT.md) · [方案](E7_TENSORFOLD_MATCH_PLAN.md) |

## 2. Decode：草稿和验证预算（决定算哪些 token）

| 实验 | 内容 | 结论 | 现状 | 文档 |
| --- | --- | --- | --- | --- |
| E6 | 草稿输出头换成完整词表 Q4 | decode 41.70 → 45.54 tok/s（+9.23%） | 默认 | [报告](E6_DRAFT_HEAD_REPORT.md) · [方案](E6_DRAFT_HEAD_PLAN.md) · [用法](DRAFT_HEAD_USAGE.md) |
| E12 | 按路由专家并集估计验证成本 | 离线误差降 26.17%，但并集要验证后才知道；在线可用的特征只降 2.42% | 未采用 | [报告](E12_EXPERT_VERIFICATION_REPORT.md) · [方案](E12_EXPERT_VERIFICATION_PLAN.md) |
| E13 | 增量缓存 DFlash 历史投影 | 1K–2K 输入 +6.27%，短输入 −0.31% | 可选 | [报告](E13_DRAFT_CACHE_CONFIDENCE_REPORT.md) · [方案](E13_DRAFT_CACHE_CONFIDENCE_PLAN.md) · [用法](DRAFT_CONTEXT_CACHE_USAGE.md) |
| E14 | 按草稿 margin 分桶调预算 | +0.065%，区间跨零 | 未采用 | [报告](E14_CONFIDENCE_BUDGET_REPORT.md) · [方案](E14_CONFIDENCE_BUDGET_PLAN.md) |
| E15 | 训练七深度置信度预测器 | 6 个新任务上比滞后预算 +10.00%，比固定七草稿 +10.88% | 可选 | [报告](E15_CONFIDENCE_TRAINING_REPORT.md) · [方案](E15_CONFIDENCE_TRAINING_PLAN.md) |
| E16 | 600 条中英文提示重新训练 | 20 个新任务上比滞后预算 +9.61%，比 E15 +3.18% | 可选 | [报告](E16_BILINGUAL_EXPANSION_REPORT.md) · [方案](E16_BILINGUAL_EXPANSION_PLAN.md) |
| E17 | 定向补训 | 比固定预算 +11.02%，比 E16 仅 +0.09%，区间跨零 | K2 运行时默认；未确认超过 E16 | [报告](E17_COMPLETED_TRAINING_REPORT.md) · [方案](E17_TARGETED_RETRAINING_PLAN.md) |
| E18 | 真正缩短草稿 / 跳过草稿 | 草稿计算只少 0.74%，没有在线提速 | 未采用 | [报告](E18_PREDRAFT_COMPUTE_REPORT.md) · [方案](E18_PREDRAFT_COMPUTE_PLAN.md) |
| E19 | 同数据的 11,157 参数 MLP | 离线效率比线性低 0.55% | 未采用 | [报告](E19_MLP_CAPACITY_REPORT.md) · [方案](E19_MLP_CAPACITY_PLAN.md) |
| E20/E21 | 精确的训练标签加速采集 | 自适应采集器在 6 个 root 上少用 21.13% 采集计算 | 可选（采集工具） | [E20](E20_FAST_COLLECTOR_REPORT.md) · [E21](E21_ADAPTIVE_COLLECTOR_REPORT.md) · [用法](FAST_COLLECTOR_USAGE.md) |
| E22 | NumPy 标量 MLP 运行时 | 小头 0.3353 → 0.0118 ms（28.38 倍）；回放仍比线性低 0.06% | 诊断 | [报告](E22_PREDICTOR_DIAGNOSIS_REPORT.md) |
| E23/E24 | 新训练目标；504 条新 root 覆盖补训 | 新目标没过独立门槛；补训后离线比线性 +0.212% | 离线通过 | [E23](E23_OBJECTIVE_COMPARISON_REPORT.md) · [E24](E24_COVERAGE_EXPANSION_REPORT.md) |
| E25 | 补训后的实际 decode | 新 MLP 47.15、冻结 MLP 47.28、线性 47.49 tok/s，区间跨零 | 未采用 | [报告](E25_ONLINE_COMPARISON_REPORT.md) · [方案](E25_ONLINE_COMPARISON_PLAN.md) |
| E26 | 残差混合路线 | 按用户要求在 18/108 次计时后中止 | 中止 | [方案](E26_RESIDUAL_HYBRID_PLAN.md) |
| E27 | NumPy 线性运行时 + 实时成本修正 | NumPy +0.60%，区间跨零 | 未采用 | [报告](E27_LINEAR_COST_REPORT.md) · [方案](E27_LINEAR_COST_PLAN.md) |
| E28 | 草稿 MLP 的 INT8 权重 + BF16 激活 | 草稿阶段 −6.89%；decode +0.61%，区间跨零 | 未确认 | [报告](DRAFT_BACKBONE_W8_REPORT.md) · [方案](DRAFT_BACKBONE_W8_PLAN.md) |

## 3. Decode：target 执行与内核

| 实验 | 内容 | 结论 | 现状 | 文档 |
| --- | --- | --- | --- | --- |
| E9 | Nsight 剖析 decode | target 验证占 83.81%，草稿占 14.39% | 诊断 | [报告](E9_DECODE_PROFILE_REPORT.md) · [方案](E9_PROFILE_PLAN.md) |
| E10 | 调专家核的 warp 数和预取深度 | holdout 上 +0.08% | 未采用 | [报告](E10_TARGET_KERNEL_REPORT.md) · [方案](E10_TARGET_KERNEL_PLAN.md) |
| E11 | 投影改用 MXFP8 split-K | decode +0.62%，完整请求 −0.64% | 可选 | [报告](E11_PROJECTION_REPORT.md) · [方案](E11_PROJECTION_PLAN.md) · [用法](E11_PROJECTION_USAGE.md) |
| E29 | target CUDA Graph | 相同预算下 47.05 → 47.65 tok/s（+1.28%），区间为正 | 可选 | [报告](E29_TARGET_VERIFICATION_REPORT.md) · [方案](E29_TARGET_VERIFICATION_PLAN.md) |
| E30 | 冻结路径剖析；输出头布局 | 专家 41.3%、量化投影 26.9%、BF16 23.4%；新布局头部只省 0.8% | 未采用 | [报告](E30_TARGET_PROFILE_REPORT.md) · [方案](E30_TARGET_PROFILE_PLAN.md) |
| E31 | FP8 专家 scale 成对转换 | 两种计时区间都跨零 | 未采用 | [报告](E31_EXPERT_SCALE_REPORT.md) · [方案](E31_EXPERT_SCALE_PLAN.md) |
| E32 | MXFP8 投影的紧凑 staging | 两种计时都变慢（−1.06% / −0.92%） | 未采用 | [报告](E32_COMPACT_STAGING_REPORT.md) · [方案](E32_COMPACT_STAGING_PLAN.md) |
| E33 | INT8 辅助筛选 + 原 BF16 头精算 | 完整头耗时 −36.24%；decode 46.91 → 48.51 tok/s（+3.41%），区间为正 | 可选 | [报告](E33_EXACT_HEAD_SCREEN_REPORT.md) · [方案](E33_EXACT_HEAD_SCREEN_PLAN.md) · [原始提案](EXACT_HEAD_SCREEN_PLAN.md) |
| E34 | GPU 端候选压缩 | 比 E33 头耗时再少 3.78%，没到 5% 门槛 | 未采用 | [报告](E34_DEVICE_SCREEN_REPORT.md) · [方案](E34_DEVICE_SCREEN_PLAN.md) |
| E35 | 共享 / 路由专家双 stream 并行 | MoE 耗时热复用 −3.04%、缓存扰动 −0.50%，没到成本门槛；重核几乎不重叠 | 未采用 | [报告](E35_MOE_OVERLAP_REPORT.md) · [方案](E35_MOE_OVERLAP_PLAN.md) |
| E52 | decode 专家流水 | 两种计时区间都跨零 | 未采用 | [报告](E52_DECODE_PIPELINE_REPORT.md) · [方案](E52_DECODE_PIPELINE_PLAN.md) |

## 4. Prefill（prompt 处理）

| 实验 | 内容 | 结论 | 现状 | 文档 |
| --- | --- | --- | --- | --- |
| E8 | prompt 按 8 行分块，沿用 decode 算术 | 首 token 时间之和 19.69 → 4.23 秒，完整请求 1.67 倍；decode −1.14% | 默认 | [集成](E8_INTEGRATION_REPORT.md) · [分块](E8_PREFILL_REPORT.md) · [用法](PROMPT_USAGE.md) |
| E36 | 大块 prefill，保持 decode 等价的算术 | 七条前缀精确对照通过；prompt 本体 110.87 → 889.22 tok/s（8.02 倍） | 基础 | [报告](E36_PREFILL_NUMERICS_REPORT.md) · [方案](E36_PREFILL_NUMERICS_PLAN.md) |
| E37 | prefill 剖析；GDN scratch 复用 | 路由专家占 52.5%–54.9%；复用让 wall +1.02% | 未采用 | [报告](E37_PREFILL_PROFILE_REPORT.md) · [方案](E37_PREFILL_PROFILE_PLAN.md) |
| E38 | prefill 专家权重复用 | 可少 23.02% 加载阶段，但候选让 MoE 成本 +3.93% | 未采用 | [报告](E38_PREFILL_EXPERT_REPORT.md) · [方案](E38_PREFILL_EXPERT_PLAN.md) |
| E39 | CTA 协作加载 | 寄存器更少，但 MoE 成本 +3.11% | 未采用 | [报告](E39_PREFILL_CTA_REPORT.md) · [方案](E39_PREFILL_CTA_PLAN.md) |
| E40 | 自主 prefill 执行模块 | 接入 target-only 和动态 DFlash 请求路径，算子仍来自 TensorFold | 可选 | [报告](E40_PREFILL_EXECUTOR_REPORT.md) · [方案](E40_PREFILL_EXECUTOR_PLAN.md) · [用法](E40_PREFILL_EXECUTOR_USAGE.md) |
| E41 | E40 接入 Graph/E33/W8 完整链路 | 相对 8 行分块：mixed1536 请求 15.35 → 2.13 秒，code513 5.46 → 1.20 秒；decode 不变 | 可选 | [报告](E41_PREFILL_GRAPH_REPORT.md) · [方案](E41_PREFILL_GRAPH_PLAN.md) |
| E42 | prefill 分块加大到 1024 行 | 长输入 prefill wall −5.80% | 可选，默认 512 | [报告](E42_WIDER_PREFILL_REPORT.md) · [方案](E42_WIDER_PREFILL_PLAN.md) |
| E43 | 只靠指令调度的专家流水 | 无收益 | 未采用 | [报告](E43_PREFILL_PIPELINE_REPORT.md) · [方案](E43_PREFILL_PIPELINE_PLAN.md) |
| E44 | Mia 共享输入专家 tile | MoE 成本 −26.62%，长输入 prefill wall −6.97% | 可选 | [报告](E44_MIA_PREFILL_TILE_REPORT.md) · [方案](E44_MIA_PREFILL_TILE_PLAN.md) |

## 5. 运行时与多请求（K1–K4）

| 里程碑 | 内容 | 结论 | 现状 | 文档 |
| --- | --- | --- | --- | --- |
| K1 | 在 E5 上实现 target-only 运行时：请求 API、流式事件、取消 | 七组真实 GPU 场景通过（233 行 logits、91 次状态检查） | 基础 | [报告](K1_RUNTIME_REPORT.md) · [用法](K1_RUNTIME_USAGE.md) |
| K2 | 原生 Scheduler 接入 Q4 DFlash2 与动态预算 | 单请求同步执行，预算与原循环逐宽度一致 | 基础 | [报告](K2_DFLASH_SCHEDULER_REPORT.md) · [用法](K2_DFLASH_SCHEDULER_USAGE.md) |
| K3 | 运行时接入 target CUDA Graph | 显式选项，带 eager 回退 | 可选 | [报告](K3_GRAPH_RUNTIME_REPORT.md) · [用法](K3_GRAPH_RUNTIME_USAGE.md) |
| K3 + E33 | E33/W8/Graph 整体迁入运行时 | 47.459 vs 同期 E33 48.378 tok/s，约 1.90% 接入开销 | 可选 | [报告](K3_E33_RUNTIME_REPORT.md) · [用法](K3_E33_RUNTIME_USAGE.md) |
| K4（单请求开销） | 减少轮询、历史复制和 CPU 元数据分配 | 47.282 → 47.994 tok/s（+1.51%） | 可选 | [报告](K4_RUNTIME_OVERHEAD.md) |
| 联合 Target | 两个请求共享一次 target forward | 总 decode 吞吐 23.64 → 38.63 tok/s（+63.4%），target-only | 可选 | [报告](JOINT_TARGET_EXECUTION_REPORT.md) · [方案](JOINT_TARGET_EXECUTION_PLAN.md) |
| K4（多请求服务） | 双请求动态 DFlash 联合验证、异步 SSE | 功能通过；相对优化后的单请求顺序处理，三组负载 −1.09% 到 +13.51%，尚未稳定领先 | 进行中 | [PR #19](https://github.com/EdwinWang831/spark-infer-q/pull/19) |

## 6. 长上下文与内存（E45–E58）

目标是 8 个并发、每个 256K 的请求，放进 88 GiB 的 CUDA 分配上限。

| 实验 | 内容 | 结论 | 现状 | 文档 |
| --- | --- | --- | --- | --- |
| E45 | 有界草稿历史；4K QSA 与 Graph | 精确检查通过；每请求特征环 99.61 MiB；无提速 | 可选 | [报告](E45_LONG_CONTEXT_REPORT.md) · [方案](E45_LONG_CONTEXT_PLAN.md) · [用法](E45_LONG_CONTEXT_USAGE.md) |
| E46 | 单请求直接跑 256K | 262,080 个输入 token，prefill 230.4 秒，检索 3/3，峰值 81.41 GiB | 基础 | [报告](E46_256K_REPORT.md) · [方案](E46_256K_PLAN.md) |
| E47 | 压缩 KV（INT8/FP8），两个 256K 状态 | 两个 262,080-token 请求同时驻留，每个 4.13 GiB，峰值 82.63 GiB；精度门槛未过 | 未采用为默认 | [报告](E47_KV_DUAL_REPORT.md) · [方案](E47_KV_DUAL_PLAN.md) · [用法](E47_SHARED_GRAPH_USAGE.md) |
| E48 | 两个 256K 请求的 prefill/decode 公平调度 | A 的最大输出间隔 225.42 → 0.84 秒，完成时间 453.03 → 241.60 秒 | 可选 | [报告](E48_PREFILL_FAIRNESS_REPORT.md) · [方案](E48_PREFILL_FAIRNESS_PLAN.md) · [用法](E48_PREFILL_FAIRNESS_USAGE.md) |
| E49 | 256K 输入、512 输出的动态 DFlash | decode 16.36 → 30.81 tok/s（+88.37%，相对 target-only）；四个 256K 状态预计 90.14 GiB，超上限 | 基础 | [报告](E49_LONG_DYNAMIC_REPORT.md) · [方案](E49_LONG_DYNAMIC_PLAN.md) |
| E50 | 自主长上下文链路优化；八路容量核算 | decode 32.14 → 34.49 tok/s（+7.34%）；八路预计 FP8 106.77 GiB、INT4 95.52 GiB | 可选 | [报告](E50_OWN_OPTIMIZATION_REPORT.md) · [方案](E50_OWN_OPTIMIZATION_PLAN.md) |
| E51 | 有界原始 QSA 索引环 | 每请求省 0.744 GiB；decode 不变 | 可选 | [报告](E51_BOUNDED_INDEX_REPORT.md) · [方案](E51_BOUNDED_INDEX_PLAN.md) · [用法](E51_BOUNDED_INDEX_USAGE.md) |
| E53 | 两个 256K 长请求并发 | 两个 261,120-token / 512 输出请求同时运行，输出与状态和单独运行一致，峰值 81.14 GiB；动态轨迹的严格门槛未过 | 未采用 | [报告](E53_LONG_CONCURRENCY_REPORT.md) · [方案](E53_LONG_CONCURRENCY_PLAN.md) · [用法](E53_LONG_CONCURRENCY_USAGE.md) |
| E54 | 每请求私有的草稿投影缓存（50 MiB） | 两个 256K 动态请求恢复严格一致；八路预计 104.06 GiB | 可选 | [报告](E54_REQUEST_DRAFT_CACHE_REPORT.md) · [方案](E54_REQUEST_DRAFT_CACHE_PLAN.md) · [用法](E54_REQUEST_DRAFT_CACHE_USAGE.md) |
| E55 | 常驻内存盘点；INT4 KV | 每状态 BF16 6.41 / FP8 3.51 / INT4 2.10 GiB；八路 INT4 预计 92.82 GiB，仍超 4.82 GiB；logit 误差门槛未过 | 未采用 | [报告](E55_RESIDENCY_REPORT.md) · [方案](E55_RESIDENCY_PLAN.md) · [用法](E55_INT4_KV_USAGE.md) |
| E56 | INT4 误差归因、少数层保留 BF16 | 误差有所下降但不够，内存代价大 | 未采用 | [报告](E56_KV_PRECISION_REPORT.md) · [方案](E56_KV_PRECISION_PLAN.md) |
| E57 | 非对称仿射 KV；先 BF16 prefill 再转 INT4 | 都没过 3% 误差门槛 | 未采用 | [报告](E57_AFFINE_DEFERRED_KV_REPORT.md) · [方案](E57_AFFINE_KV_PLAN.md) |
| E58 | 原生 KV 转换；INT4/FP8 混合 | 转换一致；没有配置同时满足精度和八路容量 | 未采用 | [报告](E58_NATIVE_CONVERSION_REPORT.md) · [方案](E58_NATIVE_CONVERSION_PLAN.md) |

## 7. 设计提案与未执行的方案

| 文档 | 内容 | 现状 |
| --- | --- | --- |
| [全局调度 RFC](GLOBAL_COST_AWARE_SCHEDULER_RFC.md) | 跨请求验证预算、Prefill/Decode 协调、统一内存准入 | 设计 |
| [Kaifra 愿景](KAIFRA_VISION_RFC.md) · [实施路线](KAIFRA_IMPLEMENTATION_ROADMAP.md) | 基于 DeepInfer 的自适应推理引擎的定位与 K0–K7 阶段 | 设计；K1–K4 已部分实现 |
| [Decode 前沿 RFC](KAIFRA_DECODE_FRONTIER_RFC.md) · [验证矩阵](KAIFRA_DECODE_VALIDATION_MATRIX.md) | 八个 decode 研究方向及止损规则 | 设计 |
| [未执行方案表](EXPERIMENTS.md#proposals-without-results-yet) | prefill 数值参考、专家占用速率、专家感知预算在线阶段、草稿 fidelity | 未执行或部分执行 |
| [推理栈的轻量递归自我改进](USE_TIME_SELF_EVOLUTION_PLAN.md) | 改进器在使用中改进推理栈，也改进自己的搜索方式；裁判（精确回放、逐位 oracle、晋升规则）冻结，不改变任何输出。v2 按 E17–E58 重排了可进化部件，v3 把递归层改为主线；11-06 补充裁判强度曲线和白天 CPU 同址运行 | 设计；EV0–EV8 未执行 |

## 8. 其他参考

- 架构与移植：[候选架构](ARCHITECTURE.md)、[适配接口](ADAPTER_INTERFACES.md)、[初始源码审计](PORTING_MATRIX.md)、[prefill 下一阶段（E8 前）](PREFILL_NEXT.md)
- 运行环境：[常驻研究 worker](RESIDENT_RESEARCH_WORKER.md)
- 历史汇总：[E0–E9 结果表](RESULTS.md)；执行记录 [E16](E16_BILINGUAL_EXPANSION_STATUS.md)、[E17](E17_TARGETED_TRAINING_STATUS.md)、[E17 执行](E17_TARGETED_TRAINING_EXECUTION.md)、[E19](E19_MLP_CAPACITY_STATUS.md)、[E23/E24](E23_E24_EXECUTION_STATUS.md)
- 数据许可：[OASST1 署名](data/OASST1_ATTRIBUTION.md)
