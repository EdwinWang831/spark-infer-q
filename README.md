# SparkInfer-Q

**简体中文** | [English](README.en.md)

在单台 NVIDIA DGX Spark（GB10 / SM121）上研究本地大模型的投机解码、计算预算分配和运行时执行。目前以 Qwen3.8-Flash-Next 和 DFlash2 草稿模型为实验对象，接入 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 的调度与推测验证机制。

**从这里开始：** [中文研究导览](docs/RESEARCH_OVERVIEW.zh-CN.md) · [English overview](docs/RESEARCH_OVERVIEW.md) · [CPU 预测器示例](examples/adaptive_budget.py) · [实验索引](docs/README.md)。

研究导览按“问题、实现、证据、局限”介绍项目。文档于 2026-11-02 核对；硬件报告覆盖截至 2026-10-28 的 E0–E58，以及 K1–K4 运行时里程碑。

## 研究问题

一台 CPU 与 GPU 共享 128 GB 内存的桌面设备，能不能让一个大型 MoE 模型跑得更快、上下文更长、同时服务多个请求，并且**输出和已提交状态与同一个量化 target 的串行执行一致**？压缩 target KV 的候选还需要通过独立的 BF16 KV 相对精度门槛。

- 模型：Qwen3.8-Flash-Next，NVFP4 量化，48 层（36 层 GDN 线性注意力 + 12 层 QSA 稀疏注意力），每层 512 个路由专家取 top-10，target 权重约 70 GiB。
- 范围：单台 Spark、greedy 解码；训练预测器时冻结 target 和草稿主干。当前执行层复用外部 TensorFold 算子和 DeepInfer 调度/预算接口。vLLM、SGLang 仍是未实测的对照候选，尚未建立生产服务或跨模型泛化结论。

## 本项目的具体工作

| 主线 | 本仓库开展的工作 | 实现与证据 |
| --- | --- | --- |
| 接受率预测 | 使用 target 验证前可获得的信号，训练、校准并执行轻量草稿前缀接受率预测器 | [预测器](spark_infer_q/confidence_head.py)、[特征](spark_infer_q/confidence_features.py)、[E16](docs/E16_BILINGUAL_EXPANSION_REPORT.md)、[E17](docs/E17_COMPLETED_TRAINING_REPORT.md) |
| 动态验证预算 | 结合预测推进量与验证成本，选择本轮验证宽度，并接入上游 MTPVarlen | [预算规则](spark_infer_q/confidence_head.py)、[运行时接入](spark_infer_q/kaifra_speculative.py)、[K2](docs/K2_DFLASH_SCHEDULER_USAGE.md) |
| 执行与评估 | 请求/状态所有权、prefill 执行、共享 target 计算、生命周期检查、配对实验与证据导出 | [联合 target](spark_infer_q/kaifra_joint_target.py)、[prefill](spark_infer_q/prefill.py)、[E54](docs/E54_REQUEST_DRAFT_CACHE_REPORT.md)、[测试](tests) |

本仓库的工作集中在预测、预算接入、执行适配与评估；模型架构、DFlash2 方法、DeepInfer Scheduler/MTPVarlen 和 TensorFold 算子来自上游。依赖关系与归属见下文说明。

## 方法

每个 decode 步分三部分：

1. **出草稿**：DFlash2 草稿模型读取 target 已提交的特征，一次给出最多 7 个候选 token。
2. **定预算**：根据接受率预测和实测成本，决定这一步让 target 验证几行（DeepInfer 的 MTPVarlen）。
3. **验证并提交**：target 一次前向验证这些行，只接受和 greedy 结果一致的最长前缀，写入 GDN/QSA 状态。

预测器估计推进概率，不负责生成答案。当前 post-draft 策略在完成草稿后选择**验证长度**，已付出的草稿成本仍要计入。跨请求调度和 GPU 合批是另外的机制，见[研究导览](docs/RESEARCH_OVERVIEW.zh-CN.md)。

所有实验遵守同一套规范：

- **正确性契约**：每个候选都和逐 token 的 serial 执行对照输出 token、committed logits 和完整状态，不一致就不算通过。
- **事先冻结方案**：任务、变量和通过门槛在运行前写进 `*_PLAN.md`，结果写进 `*_REPORT.md`。
- **负结果同样保留**：没过门槛的候选记为"未采用"，不事后放宽阈值。
- **开销全计入**：特征提取、CPU 传输、预测等额外耗时都算进计时。

## 主要结果

| 方向 | 结果 | 实验 |
| --- | --- | --- |
| Decode：验证预算 | 训练的轻量接受率预测器，比滞后预算快约 10%（20 个新任务上 44.08 vs 40.21 tok/s） | [E15](docs/E15_CONFIDENCE_TRAINING_REPORT.md)、[E16](docs/E16_BILINGUAL_EXPANSION_REPORT.md) |
| Decode：草稿 | 草稿输出头换成完整词表 Q4，+9.23% | [E6](docs/E6_DRAFT_HEAD_REPORT.md) |
| Decode：target 执行 | 精确筛选输出头 +3.41%（46.91 → 48.51 tok/s）；CUDA Graph +1.28% | [E33](docs/E33_EXACT_HEAD_SCREEN_REPORT.md)、[E29](docs/E29_TARGET_VERIFICATION_REPORT.md) |
| Prefill | 8 行分块让完整请求快 1.67 倍；再改用 512 行分块的自主 prefill，1,536-token 请求从 15.35 秒降到 2.13 秒 | [E8](docs/E8_INTEGRATION_REPORT.md)、[E41](docs/E41_PREFILL_GRAPH_REPORT.md) |
| 长上下文 | 单请求 256K（262,080 token）跑通，检索 3/3；256K 输入下动态 DFlash 让 decode 比 target-only 快 88%（16.36 → 30.81 tok/s） | [E46](docs/E46_256K_REPORT.md)、[E49](docs/E49_LONG_DYNAMIC_REPORT.md) |
| 多请求 | 两个请求共享一次 target 前向，总 decode 吞吐 +63.4%；两个 256K 请求同时运行，输出和状态与各自单独运行一致 | [联合 Target](docs/JOINT_TARGET_EXECUTION_REPORT.md)、[E53](docs/E53_LONG_CONCURRENCY_REPORT.md)、[E54](docs/E54_REQUEST_DRAFT_CACHE_REPORT.md) |

每个数字只在对应报告的任务和计时口径内成立，不同行之间不能比较，收益也不能相乘。大多数任务是合成提示。**全部实验的一句话结论见 [docs/README.md](docs/README.md)。**

## 已测试但未采用的候选

- **已测内核候选没有通过晋升门槛。** E9 中 target 验证占 decode 时间的 84%，E30 中专家核是最大的一块。调 warp 和预取、改 staging、权重复用、双 stream、指令流水等 11 项候选（E10、E30–E32、E34、E35、E37–E39、E43、E52）未被采用；这一结论不排除其他内核设计或工作负载。
- **已测更大预测器没有确认 decode 收益。** MLP 补训后离线评估略好于线性（+0.21%），但实际 decode 中没有超过线性（E19、E24、E25）。
- **验证后才获得的特征不能直接支持因果决策。** 专家并集能解释验证成本，但要验证完才知道（E12）。

有效的改进主要来自"决定算什么"（验证预算、草稿、输出头筛选）和重新组织 prefill。

## 当前卡点

- **内存**：目标是 8 个并发、每个 256K 的请求，放进 88 GiB 的 CUDA 分配上限。即使把 KV 压到 INT4，预计仍需 92.82 GiB（E55）。
- **KV 压缩精度**：所有压缩候选仍未通过事前定的 3% logit 相对 L2 门槛。E58 中，BF16 prefill 后使用 FP8 / INT4 解码，在六条 4K–16K 前缀的相同 100 个 teacher decode 位置上得到 4.06% / 7.22%；这不是 256K 质量或吞吐结果。
- **多请求服务**：双请求动态 DFlash 和异步 SSE 已能运行，但还没有稳定超过优化后的单请求顺序处理（[PR #19](https://github.com/EdwinWang831/spark-infer-q/pull/19)，进行中）。
- **对照原生 TensorFold**：E7 时整体 45.62 vs 原生 49.18 tok/s，之后没有重测。

## 反馈驱动的自改进（方案）

[PR #15](https://github.com/EdwinWang831/spark-infer-q/pull/15) 提议让推理栈在日常使用中轻量地递归自我改进：改进器利用使用反馈改进推理栈，也改进自己的搜索方式；候选在时间留出的新数据上评估，按净收益决定晋升或回滚，裁判保持冻结。v3 方案的 **EV0–EV8 均未执行**。冻结预测器、手工工程提速和动态宽度决策，都不是递归自我改进的实验证据。与科研智能体的联系及拟议评估闭环见[研究导览](docs/RESEARCH_OVERVIEW.zh-CN.md#运行反馈与-rsi)。

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| [`spark_infer_q/`](spark_infer_q) | 可复用的库：greedy 验证规则、预算与预测器、草稿缓存、prefill、长上下文状态、运行时（`kaifra_*`，Kaifra 是引擎的暂定名） |
| [`scripts/`](scripts) | 各实验的运行、采集和导出脚本，文件名带实验编号 |
| [`tests/`](tests) | CPU 单元测试，不需要 GPU |
| [`docs/`](docs) | 每个实验的方案和报告，以及[按主题的索引](docs/README.md) |
| [`results/`](results) | 每个实验的脱敏数据，可由对应导出脚本重新校验 |

`main` 汇总已完成的工作；仍在进行的分支和 PR 见[仓库导航](docs/REPOSITORY_STATUS.md)。

## 快速开始（不需要 GPU，不下载权重）

以下脚本只用 Python 标准库，**不会安装依赖或修改 GPU 设置**。研究环境以 Python 3.12 为目标。

```bash
git clone https://github.com/EdwinWang831/spark-infer-q.git
cd spark-infer-q

# 用公开 E17 预测器和真实预算规则处理合成特征，只需 Python 标准库。
# 不生成模型答案、不测吞吐；输入和成本口径见研究导览。
python3 -m examples.adaptive_budget

# 输出 JSON 环境报告，不导入 PyTorch，也不探测 torch.cuda。
python3 scripts/probe_environment.py --static-only

# 指向已有的本地 Hugging Face config.json，不下载检查点。
python3 scripts/inspect_hf_config.py /path/to/model/config.json

# 运行单元测试，无需 GPU；没有 PyTorch 或 NumPy 时会跳过部分测试。
python3 -m unittest discover -s tests -v
```

在 Spark 上运行完整模型需要准备好的研究环境、外部 TensorFold 源码和检查点，见 [prompt 分块执行器](docs/PROMPT_USAGE.md)、[K2 运行时](docs/K2_DFLASH_SCHEDULER_USAGE.md) 和[常驻 worker](docs/RESIDENT_RESEARCH_WORKER.md)。

**隐私：** 环境报告默认不提交。分享前先检查内容；不要提交 API key、token、模型权重、私有路径或个人标识信息。

## 上游项目与归属

- [DeepSeek DeepInfer](https://github.com/deepseek-ai/DeepInfer)（Apache-2.0）：C++ Scheduler 和 MTPVarlen 验证预算
- [TensorFold](https://github.com/ashhart/TensorFold)：模型执行与 CUDA 内核；须保留适用的上游许可证和声明
- [MiaAI-Lab 的 TensorFold 配方](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)（MIT）：E43/E44 改编了其中的 prefill 专家 tile，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- [Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)：外部提供的草稿检查点与参考实现
- [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1)（Apache-2.0）：E17 预测器补训用的提示来源，见[署名说明](docs/data/OASST1_ATTRIBUTION.md)
- [vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)：对照基线候选，尚未实测

**引入上游源码或权重前，须检查其许可证和声明。** 本仓库包含研究计划、诊断/构建/测试工具、实验请求执行器与运行时、对外部 TensorFold 模块和完整 target 的适配、DFlash2 桥接，以及脱敏实验记录。不包含完整的 Qwen 模型实现或模型权重；加载和其余 target 算子依赖外部提供的 TensorFold 源码和检查点。

许可证：[Apache 2.0](LICENSE)。
