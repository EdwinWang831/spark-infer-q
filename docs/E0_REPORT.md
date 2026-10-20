# E0：Spark / Qwen / DeepInfer 静态核验结果

状态：**PASS_STATIC**。已完成设备身份、当前服务、模型配置、版本与源码障碍的核验；这不表示 DeepInfer 已在 Spark 上运行。

本报告保留 E0 采集时的范围和待验证项。后续已完成 E1 和部分 E2 实验，最新进度见 [RESULTS.md](RESULTS.md) 与 [原生专家验证](E2_NATIVE_REPORT.md)。

采集时间：2026-10-20 12:30:51–12:39:20 UTC，即 America/Los_Angeles 当日 05:30:51–05:39:20。检查期间没有提交生成请求、加载模型或执行新 CUDA 内核。

机器可读记录：[2026-10-20.json](../results/e0/2026-10-20.json)。原始机器信息、实际路径及完整配置只保留于本地 Git 元数据目录，不放入待发布文件。

## 可以据此作出的结论

1. 设备确实是单卡 NVIDIA GB10，计算能力 **12.1 / SM121**，由 NVML 查询并用 nvidia-smi 交叉核对。不是根据产品名称推测。
2. 当前服务使用 **TensorFold + NVIDIA Qwen3.8 FP4 checkpoint**，并非 Huihui checkpoint，也不是已运行的 vLLM / SGLang。当前服务必须加入后续性能比较的基线。
3. 当前 checkpoint 的架构、配置、量化和 MTP 接口均与 DeepInfer 的 DeepSeek 专用实现不一致。**直接加载被明确的源码检查阻止；只增加模型注册或更名配置字段不够。**
4. SM121 不在上游声明的支持列表内，但 E0 没有编译或执行算子，因此不能得出“无法移植”的结论。
5. 服务前后快照的进程、启动时间、重启次数、配置与健康状态一致。没有新的性能提升或模型正确性结论。

## 实际设备与环境

| 项目 | 实际读取值 | 读取方式 / 边界 |
| --- | --- | --- |
| GPU | 1 × NVIDIA GB10，SM121 | NVML `nvmlDeviceGetCudaComputeCapability` 返回 `[12,1]`；nvidia-smi 返回 `12.1` |
| CPU 架构 | aarch64 | 宿主机与容器的平台信息 |
| OS / 内核 | Ubuntu 24.04.4 LTS / 6.17.0-1031-nvidia | 宿主机元数据 |
| 驱动 | 580.173.02 | nvidia-smi |
| 系统总内存 | 121.627 GiB | `/proc/meminfo`；不等于 GPU 可分配内存 |
| 两次可用内存快照 | 23.085 / 22.674 GiB | 共享主机上的瞬时值，不能将变化全部归因于本次检查 |
| Python | 宿主机与运行容器均为 3.12.3 | 解释器元数据 |
| GCC | 宿主机与运行容器均为 13.3.0 | `gcc -dumpfullversion` |
| 宿主机 CUDA Toolkit | 13.0.88 | `/usr/local/cuda/bin/nvcc --version`；原 SSH PATH 未包含 nvcc |
| 当前服务容器 CUDA Toolkit | 13.3.73 | 容器内 `nvcc --version` |
| 当前容器 PyTorch | 2.13.0a0+9186a08b2c.nv26.7.59513937，CUDA build 13.3 | 只读 distribution metadata 和 AST 解析 `torch/version.py`，没有导入 torch |
| 当前容器 Triton | 3.7.1+gitf797708c.nv26.7 | 包元数据 |
| 当前容器 Transformers | 5.17.0 | 包元数据 |
| 当前容器 ModelOpt | 0.44.0 | 包元数据 |

nvidia-smi 的 `memory.total` / `memory.used` 返回 N/A；未伪造为 128 GiB 显存。设备计算能力可由 NVML 读出，不需要调用 `torch.cuda`。

宿主机系统 Python 没有 torch；当前服务容器有上述 PyTorch。vLLM、SGLang、DeepInfer、TileLang 等在这两个被检查的 Python 环境中未发现对应 distribution，不能据此断言整台主机的所有虚拟环境都没有安装。

上游 NVIDIA requirements 固定 `torch==2.13.0+cu132` 和 `flash-attn-4==4.0.0b34`。当前服务容器的 CUDA/PyTorch/FlashAttention 不是这组准确依赖。E1 应使用独立环境；E0 未证明当前组合会构建失败，也没有在当前容器安装依赖。

## 当前实际模型

| 项目 | 读取值 |
| --- | --- |
| 模型 | `nvidia/Qwen3.8-Flash-Next-FP4-Local` |
| Snapshot revision | `b9ce18a6887e18c089fdea8509a33e63429a3300` |
| 架构 / model_type | `Qwen4ExpForConditionalGeneration` / `qwen4_exp` |
| 主干 | 48 层：36 × linear_attention，12 × full_attention |
| 专家 | 512 个，每 token 选择 10 个 |
| KV heads | 2 |
| 原生上下文配置 | 262,144 |
| 文本精度字段 | `text_config.dtype=bfloat16`；顶层没有 `dtype` |
| 原生 MTP | `mtp_num_hidden_layers=1` 且 `mtp.num_hidden_layers=1`，两者一致 |
| 量化 | ModelOpt `MIXED_PRECISION` |
| 配置中的量化模块条目 | NVFP4：49；MXFP8：438；W4A16_NVFP4：58；这是模块条目数，不是张量或参数数量 |
| 权重索引 | 375,487 个张量名、21 个分片，引用的分片均存在 |
| 分片文件字节数 | 102,192,011,716 bytes，约 95.174 GiB；不是运行时内存占用 |

`config.json` SHA256：`b50dbc52a0c83b0cd2a44a372c1fb6b5e0e6827d22f2611c97d4d030f92f6b28`。

权重索引 SHA256：`5fa166bbb1cf25adda455117563efa72a2072272001f459eedece131ffc913e1`。

**身份核验边界：** revision 来自正在运行的服务参数中的 snapshot 路径，配置和索引均读取了实际挂载文件并记录摘要。独立 Hugging Face API 核对连接超时；没有为本次 E0 重读约 95 GiB 权重来校验所有权重内容，也没有由“分片存在”推导“权重完整性已全部验证”。读取了约 45 MiB 的索引元数据，没有读取张量载荷。

## 当前服务基线

- 引擎：TensorFold native Zig/CUDA；镜像标签 `tensorfold-qwen38:zig-db28187`。
- 镜像 ID：`sha256:49a4adb03c4acdff4e4c54b6e8b491b01203cc9e00070ef0e47cea27ce249694`。
- 实际启动参数：上下文 262,144，最多 5 个并行请求，最大输出 32,768，FP8 KV，视觉开启。
- 环境配置：YaRN=0、MTP draft depth=6、product-streams=5、内存预留 10 GiB。
- **MTP 层数 1 与草稿深度 6 是不同量。** 前者来自 checkpoint 结构，后者来自运行配置；本次没有生成请求来测试实际接受率、草稿行为或精度。
- 服务配方 HEAD：`656879fd752749bbd11a6e1a065a326c00940911`。
- TensorFold 源码 HEAD：`db281878ddb836fd0df510d8771ecb7e0fe47d26`，存在已应用的 tracked patches。
- Model Optimizer HEAD：`d7cbff3486a3ffb1a357e32d9810da0bfc649709`。
- 实际读取的 15 个补丁 SHA256 均匹配安装清单；完整补丁摘要保存在机器可读记录中。

这些记录固定了镜像、源码工作区和安装清单的身份；本次没有重新构建镜像来证明二进制与源码逐字节一致。并发数和上下文是配置上限，不表示已经测过五路各满 256K。

## DeepInfer 的源码障碍

本次固定审核上游提交：`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`。项目原始规划提交：`bb5c010c29285bcf6a5f44c719abe84dfdbf1b08`。

| 边界 | 实际发现 | 结论 |
| --- | --- | --- |
| 模型注册 | 只注册 `DeepseekV41ForCausalLM`，没有当前 Qwen 架构 | 明确的加载障碍 |
| 配置解析 | 要求 `model_type=deepseek_v41`、顶层 BF16 dtype、KV heads=1；当前模型字段布局及 KV heads 不同 | 需要独立配置适配，不能把缺少顶层 dtype 误读为模型不含 BF16 |
| 权重量化 | 上游要求 FP8 / MXFP4 专家布局、`[32,32]` block、`ue8m0` scale；当前专家为 NVFP4、group16 / csf | 权重和 scale 真实格式不同，不是名称问题 |
| MTP | 上游只接受 DFlash head，当前是 Qwen 原生 MTP | 需要 head 和验证元数据适配 |
| 状态 / 缓存 | 当前模型具有大量线性注意力层，上游是 V4.1 专用缓存路径 | GDN/QSA 状态提交、回退仍需独立验证 |
| GPU 算子 | SM121 已实测；上游声明 NVIDIA SM100 / Ascend 950 | SM121 编译、执行、数值与性能均未验证 |

源码依据：

- [模型注册](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/__init__.py)
- [配置与量化检查](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/hugging_face/config.py#L237)
- [DFlash-only 检查](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/mtp/dflash.py#L92)
- [硬件要求](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md)
- [NVIDIA 依赖](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/requirements-nvidia.txt)

## 对现有服务的检查

12:32:45 和 12:39:20 UTC 的两次快照均满足：

- health HTTP 200、`status=ok`，模型列表 HTTP 200。
- 容器、镜像、服务 PID、启动时间相同；重启计数均为 0。
- 启动参数、允许记录的推理环境变量、挂载及模型配置 SHA256 不变。
- GPU 进程 PID 集合不变。

E0 没有安装软件、修改运行配置、重启容器或提交推理任务。容器内诊断明确记录 `torch_in_sys_modules=false`；新静态模式只解析 PyTorch 包文件，不执行它们。

**边界：**快照支持“前后服务保持运行且配置稳定”，不能证明两次快照之间每一毫秒均可用，也不能证明瞬时延迟影响绝对为零。本次没有为了验证扰动而额外施加生成负载。

## 下一阶段允许依据与待验证项

- E0 的设备和模型身份目标已经完成，可据此安排独立环境中的 E1 小算子 / C++ 扩展测试。
- 当前有约 22–23 GiB OS 可用内存的两次快照，不能当作 CUDA 可分配容量或额外模型加载预算；后续测试必须重新核对资源。
- 先进行不加载完整权重的 SM121 编译、执行、数值测试，再决定完整引擎移植是否值得。
- 后续性能比较必须包含现有 TensorFold 基线；vLLM / SGLang 的可用性与性能仍需分别建立。
- 本次没有 DeepInfer 前向、MTP 正确性、吞吐、延迟或加速比数据；E1–E4 保持 NOT_RUN。
