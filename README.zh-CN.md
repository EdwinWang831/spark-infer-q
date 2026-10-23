# SparkInfer-Q

[English](README.md) | **简体中文**

在单台 NVIDIA DGX Spark（GB10 / SM121）上加速 Qwen 模型推理的研究项目，借鉴 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)。

## 现状一览

截至 2026-10-23，仓库包含 E0–E22 的实现、测试和脱敏证据，以及源码封存的 E23/E24 研究流程。进行中的流程不代表已经取得性能成果。

- **做什么**：Qwen3.8-Flash-Next（NVFP4，TensorFold 内核）用 DFlash2 草稿模型做推测解码，由 DeepInfer 的 MTPVarlen 决定每一步验证多少个草稿。性能实验及 K2 动态预算运行时采用同步、单请求、greedy 执行。
- **最大的单项 decode 改进**：E15/E16 训练的轻量置信度预测器。在另行冻结的新任务上，它比默认的滞后预算快约 10%（E16：44.08 vs 40.21 tok/s），目前是可选路径。
- **prompt 处理**：8 行分块让完整请求快 1.67 倍，首 token 时间之和从 19.69 秒降到 4.23 秒；decode 速度基本没变（E8）。
- **还没做到**：整体速度没有超过原生 TensorFold（E7 时 45.62 vs 49.18 tok/s，E8 之后没有重测）；也还没有多请求 GPU batch、4K 以上上下文和随机采样。

## 主要结果

| 实验 | 改了什么 | 结果 | 现状 |
| --- | --- | --- | --- |
| [E6](docs/E6_DRAFT_HEAD_REPORT.md) | 草稿输出头换成完整词表 Q4 | decode 41.70 → 45.54 tok/s（+9.23%） | 默认 |
| [E8](docs/E8_INTEGRATION_REPORT.md) | prompt 按 8 行分块，沿用 decode 算术 | 完整请求快 1.67 倍；decode −1.14% | 默认 |
| [E13](docs/E13_DRAFT_CACHE_CONFIDENCE_REPORT.md) | 增量缓存 DFlash 的历史投影 | 1K–2K 输入 +6.27%；短输入 −0.31% | 可选 |
| [E15](docs/E15_CONFIDENCE_TRAINING_REPORT.md) | 训练七深度置信度预测器 | 6 个新任务上比滞后预算 +10.00%，比固定七草稿 +10.88% | 可选 |
| [E16](docs/E16_BILINGUAL_EXPANSION_REPORT.md) | 用 600 条中英文提示重新训练 | 20 个新任务上比滞后预算 +9.61%，比 E15 的预测器 +3.18%（其中 2 个任务变慢）；英文 +7.60%，中文 +10.75% | 可选 |
| [E17](docs/E17_COMPLETED_TRAINING_REPORT.md) | 中英文预测器定向补训 | 20个root上36.83 tok/s；比固定预算+11.02%，比E16仅+0.09%，区间跨零 | 未确认超过E16 |
| [E19](docs/E19_MLP_CAPACITY_REPORT.md) | 相同数据的11,157参数MLP | 离线效率比线性低0.55%，未过晋升门槛 | 没有MLP在线测速 |
| [E21](docs/E21_ADAPTIVE_COLLECTOR_REPORT.md) | 自适应训练标签采集器 | 六个root上采集计算耗时减少21.13%；已知中文例仍慢2.15% | 采集入口已验证，不是服务TPS |
| [E22](docs/E22_PREDICTOR_DIAGNOSIS_REPORT.md) | 可选NumPy标量MLP运行时 | 小头0.3353→0.0118毫秒（28.38×）；诊断回放仍比线性低0.06% | CPU原型，没有新增decode TPS |
| [E11](docs/E11_PROJECTION_REPORT.md) | 投影改用 MXFP8 split-K | decode +0.62%，完整请求 −0.64% | 可选，默认不变 |
| [E10](docs/E10_TARGET_KERNEL_REPORT.md) | 调专家核的 warp 数和预取深度 | holdout 上 +0.08% | 未采用 |
| [E12](docs/E12_EXPERT_VERIFICATION_REPORT.md) | 按路由专家并集估计验证成本 | 离线成本误差降 26.17%，但专家并集要验证后才知道；能在线用的特征只降 2.42% | 未采用 |
| [E14](docs/E14_CONFIDENCE_BUDGET_REPORT.md) | 按草稿 margin 分桶调整预算 | +0.065%，区间跨零 | 未采用 |

**怎么读这些数字**

- 每个实验用自己的任务集，不同行的 tok/s 不能直接比较，收益也不能相乘。
- 都是单台 Spark、单请求、greedy 解码。大多数任务是合成提示，上下文最多约 2K token。
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
| [E7](docs/E7_TENSORFOLD_MATCH_REPORT.md) | 充分预热后对照原生 TensorFold | 整体 45.62 vs 49.18 tok/s；8 个自然完成的任务 45.57 vs 43.58 |
| [E9](docs/E9_DECODE_PROFILE_REPORT.md) | 用 Nsight 剖析 decode | target 验证占 83.81%，草稿占 14.39% |

[E20固定宽度采集](docs/E20_FAST_COLLECTOR_REPORT.md)保留为研究对照，推荐已验证的自适应[采集入口](docs/FAST_COLLECTOR_USAGE.md)。[E18短草稿](docs/E18_PREDRAFT_COMPUTE_REPORT.md)只让草稿计算减少0.74%，没有建立在线提速。

**E23/E24执行（2026-10-23 UTC）：** 新机会损失目标已完成[60个新root的独立测试](docs/E23_OBJECTIVE_COMPARISON_REPORT.md)，比同结构BCE +0.013%、比线性+0.019%，区间均跨零，未进入在线晋升。另504条[覆盖扩展流程](docs/E24_COVERAGE_EXPANSION_PLAN.md)已通过分词容量和六场景GPU检查，正在采集学习数据，扩大数据后的拟合/测试仍待完成；见[带时间的状态](docs/E23_E24_EXECUTION_STATUS.md)。

## 工作原理

每个 decode 步分三步：

1. **出草稿**：DFlash2 读取 target 已提交的特征，一次给出最多 7 个草稿 token。输出头用 Q4，草稿主干和 selector 保持 BF16。
2. **定预算**：决定这一步让 target 验证几行（1–8 行）。K2 服务默认使用现有 E17 预测器加本轮 post-draft MTPVarlen 动态预算；保留原生滞后/E16 动态预算对照。
3. **验证并提交**：target 一次验证这些行，接受与 greedy 结果一致的最长前缀，只把这部分写入 GDN/QSA 状态。

| 部分 | 来源 | 本仓库代码 |
| --- | --- | --- |
| Target 执行、CUDA 内核、GDN/QSA 提交 | TensorFold（外部提供源码和检查点） | `scripts/e3_target.py`、`spark_infer_q/csf_loader.py` |
| 草稿模型 | DFlash2 | `spark_infer_q/dflash2_block.py`、`draft_head.py`（Q4 头）、`draft_context.py`（上下文缓存） |
| 验证预算 | DeepInfer MTPVarlen | `spark_infer_q/verification_budget.py`、`confidence_head.py`、`confidence_features.py` |
| greedy 验证规则 | 本仓库 | `spark_infer_q/speculative.py` |
| prompt 分块 | 本仓库 | `spark_infer_q/prompt.py` |
| 调度桥接（E5） | DeepInfer C++ Scheduler | `spark_infer_q/deepinfer_bridge.py`、`executor.py` |

完整执行器是 [`scripts/e8_ours.py`](scripts/e8_ours.py)，提供 SSE 接口。它需要准备好的研究环境、外部依赖和检查点，见[用法说明](docs/PROMPT_USAGE.md)。实验之间由[常驻 worker](docs/RESIDENT_RESEARCH_WORKER.md) 保持权重加载。

[K1 运行时](docs/K1_RUNTIME_USAGE.md)提供原生 Scheduler 驱动的 Target-only API 和 loopback SSE 入口；[K2](docs/K2_DFLASH_SCHEDULER_USAGE.md)进一步接入 DFlash2 和原生动态预算。K2 动态预算限定单请求，HTTP 一次一请求；联合 GPU batch 待后续接入。

## 下一步

- E8 之后重新对照原生 TensorFold。
- 完成已封存的E23/E24目标与覆盖实验；独立测试和完整在线A/B通过后，才考虑替换当前策略。
- 4K 以上上下文：先处理 DFlash 的 2048 滑窗和绝对位置。
- 让两个请求共享一次 target forward，从 E5 的调度桥接出发。多请求调度的整体设计见[全局调度 RFC](docs/GLOBAL_COST_AWARE_SCHEDULER_RFC.md)（仅为设计提案）。
- 五份已写好、还没执行的方案（prefill 数值参考、专家占用速率、专家感知预算的在线阶段、草稿 fidelity、精确输出头筛选），见[未执行方案列表](docs/EXPERIMENTS.md#proposals-without-results-yet)。

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

**引入上游源码或权重前，须检查其许可证和声明。** 本仓库包含研究计划、诊断/构建/测试工具、有界的实验请求执行器、对外部 TensorFold 模块和完整 target 的适配、有界的 eager DFlash2 桥接，以及脱敏实验记录。不包含完整的 Qwen 模型实现、第三方内核或模型权重；完整加载与冒烟测试依赖外部提供的 TensorFold 源码和检查点文件。

许可证：[Apache 2.0](LICENSE)。
