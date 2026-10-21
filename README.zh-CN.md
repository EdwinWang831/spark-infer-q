# SparkInfer-Q

[English](README.md) | **简体中文**

**面向 NVIDIA DGX Spark（GB10 / SM121）的 Qwen 模型推理加速实验，借鉴 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)。**

> **研究进展：** 完整 SM121 Qwen target、真实 DFlash2 草稿模型和原生 DeepInfer MTPVarlen 已支持八行验证、最多七个草稿。在八个复用回归提示上的 200 次主实验计时中，固定三草稿、固定七草稿和动态预算分别达到 38.10 / 41.50 / 43.10 decode token/s。动态实际选择了多种长度，比固定七草稿快 3.86%；但另一组 72 次匹配对照发现，它与独立校准选定的固定六草稿基本持平（42.21 对 42.20 token/s）。更长草稿对这批任务有益，动态预算相对合适固定长度的优势尚未建立。全部计时的输出和有效状态均与串行一致，另有 56 次独立 logits 检查和 44 个窗口/前缀组合检查。这些结果仅覆盖有界 greedy 实验，不代表完整引擎、统计显著性、长上下文、通用质量或服务性能结论。官方草稿模型的 FP32 数学对照通过；BF16 数值一致性和 target prefill 差异仍待解决。详见[八行实验](docs/E3_WIDE_BLOCK_REPORT.md)、[局部策略](docs/E3_LOCAL_POLICY_REPORT.md)、[草稿快路径](docs/E3_DRAFT_FASTPATH_REPORT.md)和[数值复核](docs/E3_FOLLOWUP_REPORT.md)。不同实验的收益不能直接相乘。

## 研究问题

能否将 DeepInfer 的引擎、调度、投机验证和内存管理优势用于 **单台 DGX Spark**，同时将模型执行器与 GPU 内核适配到 Qwen3.8-Flash-Next？或者，基于 vLLM / SGLang 的实现能否在已验证的速度与工程投入之间取得更好的平衡？

**推理底座由实验结果决定**，不在研究开始时预设。

## 当前实现

当前原型结合了 **TensorFold 模型执行路径**、我们适配的 **DFlash2 草稿模型**和 **DeepInfer 原生 MTPVarlen 验证预算机制**。我们编写的 Python 执行循环连接 target 特征、草稿候选、接受前缀的状态提交和实验性局部策略。主模型 CUDA 内核及 GDN/QSA 提交、重放操作来自 TensorFold。

目前尚未接入完整 DeepInfer 引擎和 C++ 多请求调度器。已报告速度对应这条有界混合实现，不是完整 DeepInfer 引擎的基准成绩。

## 实验阶段

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| E0 | 硬件/软件环境指纹与上游兼容性审计 | PASS_STATIC；本阶段未执行 CUDA 测试 |
| E1 | 构建 DeepInfer 核心，独立检查 SM121 内核兼容性 | 最小 GPU 检查通过；原生 DeepGEMM BF16 路径受阻；完整后端兼容性尚未完成 |
| E2 | 最小模型适配器、前向执行、prefill 与 decode | 部分完成：完整 target 权重已加载，选定 token/logits 路径已运行；prefill/串行数值复核、独立整模参考及 DeepInfer 调度器集成仍待完成 |
| E3 | 草稿适配、投机验证、GDN/QSA 状态正确性 | 有界 greedy DFlash2 集成与原生 DeepInfer 动态预算检查通过；随机采样、完整调度器及原生 Qwen MTP 仍待完成 |
| E4 | 同条件对照 TensorFold / vLLM / SGLang 等引擎 | 尚未执行同机性能对照；已完成 [TensorFold 源码与证据对比](docs/E4_TENSORFOLD_COMPARISON.md) |

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
