# E1：SM121 GPU 执行与局部数值验证

**最小 GPU 验证门槛通过，同时确认一个原生后端障碍。** DeepInfer 的通用 RMSNorm 能在 GB10 上实际执行；DeepGEMM 的 cuBLASLt BF16/NVFP4 路径也通过小规模数值核对。但固定版本的 DeepGEMM 原生 BF16 路径拒绝 SM121，不能据此宣布完整 DeepInfer 或 Qwen 推理兼容。

后续的独立 CSF 对照、完整专家和激活量化差异见 [E2_EXPERT_REPORT.md](E2_EXPERT_REPORT.md)。本报告保留 E1 当时的小切片范围和未完成项，不代表最新阶段仍未核对 CSF。

2026-10-20 实验。完整时间、工件摘要、逐项结果及误差见 [机器可读记录](../results/e1/2026-10-20-gpu-kernels.json)。前一阶段的核心构建和 CPU 测试见 [E1_BUILD_REPORT.md](E1_BUILD_REPORT.md)。

## 固定环境和隔离

| 项目 | 实际观测 |
| --- | --- |
| DeepInfer | `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` |
| DeepGEMM | `1fc9c66975ee322d628b3288534ca0e9ce12a64d`，构建分发版 `2.8.2+1fc9c66` |
| GPU | NVIDIA GB10，实际 capability 12.1 |
| PyTorch | 独立环境 `2.13.0+cu132`，CUDA build 13.2 |
| 编译工具链 | CUDA 13.3；不是完整 CUDA 13.2 SDK 复现 |
| TileLang / TileKernels | `0.1.15` / `2.1.0`，与 DeepInfer requirements 一致 |
| 资源限制 | 独立容器，2 CPU、6 GiB RAM、无额外 swap；只挂载实验工作目录 |
| PyTorch allocator | 每个测试进程上限 512 MiB；不包含 CUDA context、驱动和外部库分配 |

这次容器具有 GPU 访问权限。它与现有推理服务共享同一颗 GPU，并非硬件分区；小型测试仍可能造成短暂资源竞争。没有挂载完整 checkpoint，没有加载第二份模型，没有向服务发送生成请求。仅将下述微小权重切片复制到私有实验目录。

实验结束后容器已停止。所有固定上游和已初始化依赖的 tracked source 均未修改；未向当前推理容器或宿主机安装软件。DeepGEMM 初次构建因缺少 `elfutils/libdwfl.h` 失败，随后只在实验容器补装 `libdw-dev` 等头文件，重新构建成功。最终 `pip check` 通过，进程映射未加载基础镜像的全局 Torch 动态库。

## 实际执行结果

| 测试 | 结果 | 能支持的结论 |
| --- | --- | --- |
| Torch BF16 GEMM 控制组，M/N/K=32/64/128 | PASS | 本实验环境能在 GPU 执行基本矩阵计算 |
| DeepInfer 原版 RMSNorm：12 组 | 全部 PASS | 所选通用 TileLang 内核可在 SM121 编译、执行，并符合 CPU FP32 参考 |
| RMSNorm CUDA Graph：捕获后改变输入，再重放 | PASS | 该算子的所选执行形态能重放并产生新输入对应的输出 |
| DeepGEMM cuBLASLt BF16 GEMM | PASS | 该库的 cuBLASLt 接口可以在所选形状运行 |
| DeepGEMM 原生 BF16 GEMM，关闭机会性 cuBLASLt 后备路径 | **KERNEL_FAIL** | 实际在 host dispatch 报 `csrc/apis/gemm.hpp:533: Unsupported architecture`，尚未发射原生计算内核 |
| DeepGEMM cuBLASLt NVFP4：合成数据 | PASS | 所选 E2M1 数据及 per-16 E4M3 缩放布局可执行 |
| 同一 NVFP4 路径：真实 Qwen 小切片，2 个输入种子 | 全部 PASS | 所选 checkpoint 局部数据经 CSF 解码、缩放布局转换后，计算符合解量化参考 |

控制组在两个独立进程运行，不能算作两种独立覆盖。此前 14 项 CPU sequence 用例也与本次 GPU 用例分开计数。

### RMSNorm 的数值边界

12 组组合为：FP32/BF16 × `(1,128)` / `(17,2560)` / `(2,17,2560)` × 有/无 residual。2560 是实际 Qwen 配置的 hidden size；输入和 norm 权重为合成随机数据。

CPU 参考独立执行残差相加、FP32 均方及归一化，并按输出 dtype 舍入；有 residual 时还要求 residual 输出逐元素完全相等。

- FP32 最大绝对误差：`1.9073486328125e-6`；逐元素容差 `atol=rtol=2e-5`。
- BF16 最大绝对误差：`0.0009765625`；逐元素容差 `atol=0.03125, rtol=0.008`。
- CUDA Graph 用例重放后与参考一致；不代表完整调度器或模型 CUDA Graph 正确。
- 此进程 Torch 峰值 allocated/reserved：`34,957,312 / 35,651,584` bytes。它不是进程总 GPU 内存或完整模型内存。

### 真实 Qwen NVFP4 切片

checkpoint revision：`b9ce18a6887e18c089fdea8509a33e63429a3300`；config/index SHA256 与 E0 一致，未做完整权重哈希校验。

取主模型 `model.language_model.layers.0.mlp.experts.0.gate_proj` 的前 64 行、前 128 个输入通道。原矩阵为 `640×2560`，本次仅计算 `32×128 @ (64×128).T`，是局部点积，不是完整 gate projection 或 MoE 层。

权重实际使用 packed E2M1、CSF 压缩 per-16 E4M3 缩放和 FP32 全局缩放。提取脚本按现有 TensorFold `cuda_csf.zig` 的格式说明解码 CSF，再将缩放排列为 cuBLASLt vec16 布局。该解码器尚未与现有 TensorFold 运行时的解码输出独立对照，因此不声称完整 loader 语义已经验证。

| 输入种子 | 相对 L2 误差 | 乘全局缩放后折算的最大绝对误差 |
| --- | --- | --- |
| 23 | `0.001683825976215303`，约 0.168% | `0.005451747216284275` |
| 47 | `0.001715665915980935`，约 0.172% | `0.007579258643090725` |

两次均使用合成的已量化输入，GPU 输出 BF16；CPU 参考解包相同的已存储量化值并用 FP32 计算。逐元素容差为 `atol=0.0625, rtol=0.008`，应用于全局缩放之前。全局标量 `6.648472481174394e-5` 在 GEMM 外用于折算误差；GPU 测试没有复现完整模型的输入校准、真实激活和后续算子。

这些误差衡量的是计算路径相对于同一量化表示的差异，**不是 NVFP4 相对原始高精度模型的量化损失，也不是模型质量指标**。

提取过程读取 `60,120` bytes tensor payload，其中包括该 projection 的完整 CSF 小型缩放流，以恢复 exceptions；另读取约 48.4 MB 的 index/header 元数据。实际带入 GPU 实验的权重/缩放工件合计 `4,612` bytes，无完整 shard 或模型复制。本进程 Torch 峰值 allocated/reserved 为 `16,789,504 / 18,874,368` bytes。

## 只完成源码检查的组件

- **DeepSelect**，固定 `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18`：`setup.py` 显式传入 `compute_100a/sm_100a` 和 `compute_103a/sm_103a`。本次未构建或发射它的内核。单改 `TORCH_CUDA_ARCH_LIST` 不会替换这些显式参数。
- **FlashMLA**，固定 `2e5429fc5653bab6e081f09477126f731882a6a9`：`CUDA_ARCHS` 只含 `sm_100a`、`sm_103a`。本次未构建或运行；V4.1 MLA 也不能直接替代 Qwen QSA/GDN。
- **DeepGEMM 原生 FP8/FP4**：相关源码有 arch major 10 的 guard，但本次仅实测了原生 BF16 的架构拒绝。不能把源码推断写成所有量化内核均已执行失败。
- **DeepEP**：本轮单 GPU 小型验证未运行多 GPU 通信；尚未证明完整引擎可以无修改绕过其导入或运行依赖。

因此，目前的实际方向是：保留可复用的核心/元数据组件，继续验证通用 TileLang 和适合 SM121 的计算后端。尚无证据支持沿用整套原生 SM100 算子。也没有证据证明改几个架构编译参数就足够。

## 现有服务与完成边界

实验前后及结束后的服务快照均为健康，PID、启动时间、重启次数（0）、镜像、启动参数、环境和挂载不变。实验容器 cgroup 记录 peak `4,263,796,736` bytes，OOM / OOM kill / max 事件均为 0，结束后只停止实验容器。

快照不证明整个区间连续可用，也不量化瞬时延迟影响。本次没有吞吐、延迟、功耗或加速比结果。

**已完成：** 核心扩展之外的最小 GPU 执行验证、所选局部数值核对，以及原生 DeepGEMM 架构障碍定位。

**仍未完成：** 全推理依赖兼容、完整 Qwen adapter/weight loader、端到端 logits、QSA/GDN 状态、原生 MTP 验证和性能比较。下一阶段首先应独立对照 CSF/缩放与现有 TensorFold 的解释，再扩展到完整单专家投影及真实 BF16 激活的数值对照；不能直接跳到全模型速度结论。

## 复现入口

- [e1_gpu_smoke.py](../scripts/e1_gpu_smoke.py)：原版 RMSNorm、CUDA Graph 和两类 BF16 dispatch。
- [e1_run_gpu_smoke.sh](../scripts/e1_run_gpu_smoke.sh)：使用同一独立 Torch 环境和动态库路径，进程内层 600 秒超时。
- [e1_extract_nvfp4_slice.py](../scripts/e1_extract_nvfp4_slice.py)：只读提取 CSF 编码的微小权重切片，输出必须保持私有。
- [e1_nvfp4_smoke.py](../scripts/e1_nvfp4_smoke.py)：NVFP4 cuBLASLt 与 CPU FP32 解量化参考对照。

需要先准备与记录一致的独立容器、依赖、固定源码和构建产物。可将 runner 复制为实验目录的 `gpu-smoke.py` 后运行 `bash /workspace/e1/run-gpu-smoke.sh tilelang` 或 `deepgemm`。原生 BF16 的已知失败会让后一个命令返回非零，属于保留的实验结果。原始日志和源快照在私有 evidence 目录，权重字节仅留在远程实验区，未加入 Git。
