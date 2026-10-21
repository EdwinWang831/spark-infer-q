# SparkInfer-Q

[English](README.md) | **简体中文**

**面向 NVIDIA DGX Spark（GB10 / SM121）的 Qwen 模型推理加速实验，借鉴 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)。**

## 最新实测结果

截至 **2026-10-21**，E4–E9 的实现、测试和脱敏证据已合入 `main`。最新完整生成执行器采用 **8 行 prompt 分块和完整词表 Q4 草稿头**。

| E8 同十条任务配对对照 | 原逐 token prompt | 8 行 prompt | 变化 |
| --- | ---: | ---: | ---: |
| wire 首 token 延迟（TTFT），各任务中位数之和 | 19.69 秒 | **4.23 秒** | **加速 4.66 倍** |
| 完整请求耗时，各任务中位数之和 | 37.92 秒 | **22.67 秒** | **加速 1.67 倍，耗时减少 40.22%** |
| 加权 decode 吞吐 | 45.91 token/s | 45.39 token/s | -1.14% |

表中耗时是各任务中位数之和，**不是单个请求的延迟**。收益来自 prompt 处理，decode 没有提速。一次模型加载、十条已见短上下文任务、四轮配对重复，共 80 次正式计时。40 次完整生成 logits/状态 oracle 对照（3,348 行已提交 logits）及全部 80 次计时输出 IDs/请求后状态检查通过。详见 [E8 报告](docs/E8_INTEGRATION_REPORT.md)、[数据](results/e8/2026-10-21-integrated-prompt.json)及[执行器用法](docs/PROMPT_USAGE.md)。

**最近一次原生对照（E7，在 prompt 接入之前）：** 200 次充分预热的均衡计时中，我们的完整 Q4 动态路径整体为 **45.62 decode token/s**，原生 TensorFold MTP 为 **49.18**；八条自然完成任务为 **45.57 / 43.58**。跨引擎 8/10 任务输出摘要一致；原生接口未提供完整 IDs/logits/状态，无法作精确对照。**E8 之后尚未重测原生 TensorFold**，不能宣称已整体反超。这是固定配方的同机对照，不代表对上游最新性能宣称的复测。[报告](docs/E7_TENSORFOLD_MATCH_REPORT.md) · [数据](results/e7/2026-10-21-tensorfold-match.json)。

**最新诊断（E9，没有新增提速）：** 完全未采样的 E8 请求中，target 验证占 decode 的 **83.81%**，草稿占 **14.39%**。三份 Nsight 样本中，**GPU 活动覆盖时间线的 91.76%**，这不是 SM 利用率。专家核、qmmf 及 BF16 矩阵乘占累计内核时间的 **78.6%**。CPU 同步区间约 99.1% 与 GPU 工作重叠，不能把同步耗时当成可直接消除的开销。采集失败、采样扰动及控制口径限制均保留在[报告](docs/E9_DECODE_PROFILE_REPORT.md)和[数据](results/e9/2026-10-21-decode-profile.json)中。

此前 [E6 草稿头实验](docs/E6_DRAFT_HEAD_REPORT.md)中，完整词表 Q4 动态 decode 比同任务 BF16-full 草稿头**快 9.23%**（45.54 / 41.70 token/s；256 次计时、64 次 oracle 检查）。Q4 现已用于完整执行器。量化仅覆盖草稿输出头；主模型使用 NVFP4，草稿 backbone、selector 和 KV 保留 BF16。这不代表通用的 BF16 等价质量。不同实验的收益不能相乘。

**E13 实验分支（可选路径）：** 三个1,024/1,536/1,920-token合成输入中，DFlash历史投影增量缓存让加权decode从 **39.73→42.22 tok/s（+6.27%）**，包含首轮构建和追加。24次计时的输出/live state及另外6次committed-logit oracle通过；12个较短任务则为 **−0.31%**，因此全局默认不变。这是最多生成96 token的同执行器decode对照，没有新增原生TensorFold排名或4K以上结论。独立接受率筛选的held-out Brier误差下降18.64%，尚非在线策略提速。研究worker在实验间保留现有权重常驻。[报告](docs/E13_DRAFT_CACHE_CONFIDENCE_REPORT.md) · [缓存用法](docs/DRAFT_CONTEXT_CACHE_USAGE.md)。

**E14 在线置信度预算（未确认提速）：** 9个新任务（含1K–1.8K输入）的108次计时及27次committed-logit/live-state oracle全部通过。margin策略为 **37.790 / 后置深度基线37.765 tok/s（+0.065%）**，耗时下降区间跨零；两组验证宽度序列完全相同。冻结曲线下32种分桶组合全选anchor+5草稿，更新实测成本后仍有30–32/32组合选同一宽度；新任务深层接受率也被高估。因此保留实验状态，下一步先补深度校准、检查信号能否改变动作，再作在线速度结论。沿用同一次常驻加载，没有新的原生TensorFold排名。[报告](docs/E14_CONFIDENCE_BUDGET_REPORT.md) · [数据](results/e14/2026-10-21-online-budget.json)。

## 研究问题

能否将 DeepInfer 的引擎、调度、投机验证和内存管理优势用于 **单台 DGX Spark**，同时将模型执行器与 GPU 内核适配到 Qwen3.8-Flash-Next？或者，基于 vLLM / SGLang 的实现能否在已验证的速度与工程投入之间取得更好的平衡？

**推理底座由实验结果决定**，不在研究开始时预设。

## 当前实现

当前原型结合了 **TensorFold 模型执行路径**、我们适配的 **DFlash2 草稿模型**和 **DeepInfer 原生 MTPVarlen 验证预算机制**。我们编写的 Python 执行循环连接 target 特征、草稿候选、接受前缀的状态提交和实验性局部策略。主模型 CUDA 内核及 GDN/QSA 提交、重放操作来自 TensorFold。

完整 Q4 SSE 研究执行器为 [`scripts/e8_ours.py`](scripts/e8_ours.py)，通过 [`spark_infer_q.prompt.prompt_state`](spark_infer_q/prompt.py) 默认采用 8 行 prompt 分块。prompt 处理沿用已验证的 decode 算术及提交路径。[用法说明](docs/PROMPT_USAGE.md)面向已准备好的研究环境，需要外部依赖与检查点。这仍是同步、单请求实验执行器。E3–E9 的 DFlash2 测量不是完整 DeepInfer 引擎的基准成绩。

独立的[原生调度桥接（E5）](docs/E5_ENGINE_BRIDGE_REPORT.md)已让 DeepInfer 原版 C++ Scheduler 驱动完整 Qwen target-only 路径，七组 GPU 场景通过，核对 364 行完整 logits 和 79 次有效状态。真实状态仍由 Qwen 后端管理，原生 block 仅作逻辑配额；GPU 请求仍逐个执行。完整 Worker/cache/异步流水及真实多请求 GPU batch 尚未接入。

下一步实测优先级为 target 内核效率、有效验证预算，再推进草稿 backbone 压缩。图执行和同步调整需要独立受控对照。E8 后仍需重新对照原生 TensorFold。E13已补充选定的1K–2K缓存/状态及decode检查；4K以上上下文、并发、随机采样及官方BF16草稿fidelity仍未验证。

## 实验阶段

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| E0 | 硬件/软件环境指纹与上游兼容性审计 | PASS_STATIC；本阶段未执行 CUDA 测试 |
| E1 | 构建 DeepInfer 核心，独立检查 SM121 内核兼容性 | 最小 GPU 检查通过；原生 DeepGEMM BF16 路径受阻；完整后端兼容性尚未完成 |
| E2 | 最小模型适配器、前向执行、prefill 与 decode | 部分完成：完整 target 权重已加载，选定 token/logits 路径已运行；沿用decode算术的prompt分块后续已在E8验证；独立整模参考及完整 DeepInfer 引擎集成仍待完成 |
| E3 | 草稿适配、投机验证、GDN/QSA 状态正确性 | 有界 greedy DFlash2 集成与原生 DeepInfer 动态预算检查通过；随机采样、完整调度器及原生 Qwen MTP 仍待完成 |
| E4 | 同条件对照 TensorFold / vLLM / SGLang 等引擎 | 部分完成：[短上下文 TensorFold / DFlash2 同机实测](docs/E4_MATCHED_REPORT.md)已完成，保留预热/数值限制；vLLM / SGLang、长上下文和并发未测 |
| E5 | 最小原生引擎桥接 | C++ Scheduler 驱动完整 Qwen 的有界 target-only 生命周期检查通过；完整 Worker/cache/异步与 GPU batch 未完成 |
| E6 | 可选草稿头压缩 | 256 次有界计时 / 64 次 oracle 通过；完整 Q4 动态相对同任务 BF16-full +9.23%；保留固定子词表覆盖与跨设备限制 |
| E7 | 充分预热的原生同机对照 | 200次计时 / 40次候选非串行oracle通过；E8之前整体动态45.62、原生49.18 tok/s；后续prompt接入已在E8验证 |
| E8 | prompt分块及完整Q4生成接入 | prompt：81组精确对照/160次计时，4.66倍；完整生成：40次oracle/80次计时，首token 4.66倍、完整请求1.67倍；decode -1.14% |
| E9 | decode CPU/GPU时间分布 | 3份有效时间线、12次生成oracle、15次输出/状态检查；GPU活动91.76%；诊断完成，未宣称新增提速 |

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
