# SparkInfer-Q

[English](README.md) | **简体中文**

**面向 NVIDIA DGX Spark（GB10 / SM121）的 Qwen 模型推理加速实验，借鉴 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)。**

> **研究进展：** 同机原生 TensorFold MTP / DFlash2 对照已完成 336 次计时，覆盖八个新英文提示文本、十四种设置。我们路径内部，冻结的 DeepInfer 动态预算达到 47.37 decode token/s，固定六/七草稿为 46.39 / 46.30，提升 2.12% / 2.31%。跨引擎六种设置的输出不同，且原生原始计时存在首轮效应、双方预热不对称；另行逐用例预热控制中，原生达到 **55.99 token/s**，与原始数据并列保留。该控制为事后顺序 session，仍有跨引擎输出差异；目前没有相对原生 TensorFold 的速度优势证据。当前冷 prompt 路径仍是端到端主要瓶颈；动态 decode 中 target 验证约占 80%，整个 drafter 约占 18%。详见[同机报告](docs/E4_MATCHED_REPORT.md)、[方法](docs/E4_MATCHED_PLAN.md)和[完整数据](results/e4/2026-10-20-matched.json)。
>
> 候选全部计时输出 IDs 与本路径串行 oracle 一致，另有 84 次独立非串行运行核对已提交 logits 和最终有效状态；主实验 240 次可客观判定的答案检查均通过。散文没有语义质量评分，截断代码仅作前缀观测。官方草稿 BF16 fidelity、target prefill 数值差异、长上下文和完整引擎验证仍待解决。此前[八行](docs/E3_WIDE_BLOCK_REPORT.md)、[局部策略](docs/E3_LOCAL_POLICY_REPORT.md)和[草稿快路径](docs/E3_DRAFT_FASTPATH_REPORT.md)是独立实验，收益不能相乘。

## 研究问题

能否将 DeepInfer 的引擎、调度、投机验证和内存管理优势用于 **单台 DGX Spark**，同时将模型执行器与 GPU 内核适配到 Qwen3.8-Flash-Next？或者，基于 vLLM / SGLang 的实现能否在已验证的速度与工程投入之间取得更好的平衡？

**推理底座由实验结果决定**，不在研究开始时预设。

## 当前实现

当前原型结合了 **TensorFold 模型执行路径**、我们适配的 **DFlash2 草稿模型**和 **DeepInfer 原生 MTPVarlen 验证预算机制**。我们编写的 Python 执行循环连接 target 特征、草稿候选、接受前缀的状态提交和实验性局部策略。主模型 CUDA 内核及 GDN/QSA 提交、重放操作来自 TensorFold。

目前尚未接入完整 DeepInfer 引擎。E3/E4 已报告速度对应 DFlash2 实验执行循环，不是下方独立调度桥接器或完整 DeepInfer 引擎的基准成绩。

另有[最小真实调度桥接](docs/E5_ENGINE_BRIDGE_REPORT.md)：DeepInfer 原版 C++ Scheduler 已实际驱动完整 Qwen target-only 路径，七组 GPU 场景通过，核对 364 行完整 logits 和 79 次有效状态。真实状态仍由 Qwen 后端管理，原生 block 仅作逻辑配额；GPU 请求仍逐个执行，尚未接入完整 Worker/异步流水，也不是吞吐结果。下一项为 DFlash2 草稿头成本拆测与压缩。

## 实验阶段

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| E0 | 硬件/软件环境指纹与上游兼容性审计 | PASS_STATIC；本阶段未执行 CUDA 测试 |
| E1 | 构建 DeepInfer 核心，独立检查 SM121 内核兼容性 | 最小 GPU 检查通过；原生 DeepGEMM BF16 路径受阻；完整后端兼容性尚未完成 |
| E2 | 最小模型适配器、前向执行、prefill 与 decode | 部分完成：完整 target 权重已加载，选定 token/logits 路径已运行；prefill/串行数值复核、独立整模参考及完整 DeepInfer 引擎集成仍待完成 |
| E3 | 草稿适配、投机验证、GDN/QSA 状态正确性 | 有界 greedy DFlash2 集成与原生 DeepInfer 动态预算检查通过；随机采样、完整调度器及原生 Qwen MTP 仍待完成 |
| E4 | 同条件对照 TensorFold / vLLM / SGLang 等引擎 | 部分完成：[短上下文 TensorFold / DFlash2 同机实测](docs/E4_MATCHED_REPORT.md)已完成，保留预热/数值限制；vLLM / SGLang、长上下文和并发未测 |
| E5 | 最小原生引擎桥接 | C++ Scheduler 驱动完整 Qwen 的有界 target-only 生命周期检查通过；完整 Worker/cache/异步与 GPU batch 未完成 |

详细通过标准和失败分类见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。另见[候选架构](docs/ARCHITECTURE.md)和[初始源码审计](docs/PORTING_MATRIX.md)。[适配接口清单](docs/ADAPTER_INTERFACES.md)区分引擎/缓存与 MTP 需求，以及已经完成的算子实验。

## 安全的 Phase-0 检查（不下载权重）

后续 DeepInfer 构建以 Python 3.12 为目标。以下检查脚本仅使用 Python 标准库，**不会安装依赖或修改 GPU 设置**。

```bash
git clone https://github.com/EdwinWang831/spark-infer-q.git
cd spark-infer-q

# 输出 JSON 环境报告，不导入 PyTorch，也不探测 torch.cuda。
python3 scripts/probe_environment.py --static-only

# 指向已有本地 Hugging Face config.json，不下载检查点。
python3 scripts/inspect_hf_config.py /path/to/model/config.json

# 运行标准库单元测试，无需 GPU。
python3 -m unittest discover -s tests -v
```

**隐私：** 环境报告默认不提交。分享前应检查内容；不要提交 API key、token、模型权重、私有路径或个人标识信息。

## 研究范围

- **范围内：** 验证 SM121 兼容性、Qwen 模型适配、GPU 投机验证、准确的 GDN/QSA 状态回滚与提交、延迟和吞吐分析。
- **初期范围外：** 重新训练 Qwen、搭建生产 Web 服务、假设 DeepInfer 已支持 GB10，以及未经实测就宣称加速。
- **基线：** 在同一台 Spark、相同输入、采样设置和检查点格式下对照 TensorFold、vLLM 与 SGLang。公开结果仅供参考；同条件性能和运行可用性需要实际验证。

## 上游项目与归属

- [DeepSeek DeepInfer](https://github.com/deepseek-ai/DeepInfer)（Apache-2.0）
- [TensorFold](https://github.com/ashhart/TensorFold)（模型执行与 CUDA 内核；须保留适用的上游许可证和声明）
- [Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)（外部提供的草稿检查点与参考实现）
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

**引入上游源码或权重前，须检查其许可证和声明。** 本仓库包含研究计划、诊断/构建/测试工具、有界实验请求执行器、对外部 TensorFold 模块和完整 target 的适配、有界 eager DFlash2 桥接，以及脱敏实验记录。不包含完整 Qwen 模型实现、第三方内核或模型权重；完整加载与冒烟测试依赖外部 TensorFold 源码和检查点文件。

许可证：[Apache 2.0](LICENSE)。
