# E1 核心扩展构建与 CPU 接口验证

本轮范围：构建 DeepInfer 核心 C++ 扩展、验证导入、运行不需要权重的上游 CPU 序列测试。**核心子项通过；本报告对应的阶段未执行 GPU 算子。** 后续完成的所选 GPU 验证及后端障碍见 [E1_GPU_REPORT.md](E1_GPU_REPORT.md)。

固定上游提交：`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`。构建和测试期间没有修改上游 tracked source。

实验区间：2026-10-20 05:51:26–06:18:20 America/Los_Angeles。机器可读记录：[2026-10-20-core-build.json](../results/e1/2026-10-20-core-build.json)。

## 构建隔离

- 创建独立实验容器，基于已存在的 TensorFold 镜像；没有进入当前推理容器安装软件。
- 实验容器使用 `runc`，无 GPU DeviceRequests、无 GPU 设备映射。
- 只挂载实验工作目录，没有挂载模型权重，没有开放服务端口。
- 上限：2 CPU、6 GiB RAM、2 个并行编译任务。
- `CUDA_VISIBLE_DEVICES` 为空；未加载第二份模型，未提交生成请求。

## Bootstrap 环境的实际结果

| 项目 | 实际结果 |
| --- | --- |
| 平台 | Linux aarch64 / Python 3.12.3 / GCC 13.3 |
| CUDA 编译工具链 | 13.3 |
| PyTorch 运行版本字符串 | `2.13.0a0+9186a08b2c.nv26.07` |
| PyTorch C++11 ABI | true |
| 核心编译 | 13 个 C++ 编译单元及共享库链接成功 |
| 初次导入 | 缺少 `libcuda.so.1`，IMPORT_FAIL |
| 提供实际驱动库后 | 导入通过，`deepinfer._C.engine` 存在 |
| 上游 CPU 测试 | `tests/engine/test_sequence.py`，14 passed |
| CUDA 初始化状态 | 导入前、测试后均为 false |

构建命令为 `python setup.py build_ext --inplace`。本次只构建核心扩展，没有运行包括全部 vendor 的顶层 `build.sh`，也未声称完整推理依赖已安装。

Bootstrap `.so` SHA256：`cf1abda2b27ca1eaf2aec7ea187f2c52ad75e1af7335b59610ca31406fdcc39e`。

## 固定 PyTorch 版本的干净环境复验

另建 `include-system-site-packages=false` 的虚拟环境，安装上游固定的 `torch==2.13.0+cu132`，重新编译全部 13 个单元，而非复用 Bootstrap 的目标文件。

| 项目 | 实际结果 |
| --- | --- |
| PyTorch | `2.13.0+cu132`，来自独立虚拟环境 |
| PyTorch CUDA build | 13.2 |
| 编译器 / headers | CUDA 13.3；不是精确的 CUDA 13.2 SDK 复现 |
| 核心扩展重新构建 | BUILD_PASS，returncode=0 |
| 扩展导入 | PASS |
| 同一组上游 CPU 测试 | 14 passed，0 failed，0 skipped |
| CUDA 初始化 | 测试前后均为 false |
| 动态库来源检查 | 未加载基础镜像的全局 PyTorch 动态库 |
| `pip check` | No broken requirements found |

固定版本 `.so` SHA256：`f6aa14b790eb7bc729251eea152fd9c83cf215bc5ab912e25ee19c57d8fc1768`。

PyTorch ARM64 wheel 的实际 SHA256：`184f88e91546a2087aee2e5e71012ea6182aaf9d6bc81c28375be2816c349a49`。它来自官方 PyTorch 下载源，与该次 pip 下载记录的摘要一致；这是观测到的工件固定值，不是独立签名的上游清单。

环境准备时，普通 Python 包从 files.pythonhosted.org 下载发生读取超时；随后使用已缓存的官方 PyTorch/NVIDIA wheel 与清华 PyPI 镜像完成安装。此故障分类为依赖下载问题，不是平台编译失败。最终依赖版本表保存在 JSON 记录中。

两套环境运行的是同一组 14 项用例，不能计为 28 种独立覆盖。固定 PyTorch 版本环境仍只准备了核心构建/CPU 测试依赖；FlashAttention 4 等完整推理依赖未安装验证。

## 导入失败的定位与处理

无 GPU 设备的构建容器没有自动挂载驱动动态库，因此初次导入的错误是 `libcuda.so.1: cannot open shared object file`。这不是 C++ 编译失败，也不能据此判断 SM121 不支持。

随后只向实验目录复制实际安装的 `libcuda.so.580.173.02`，通过该实验进程的 `LD_LIBRARY_PATH` 使用；没有把 CUDA stub 当作运行时驱动，也没有挂载 GPU 设备。

驱动库 SHA256：`a13f3101b105930ba0796b1ab99721cbee3e19360ced21253e2a09e1a102fb99`。

注意：链接阶段使用 SDK 的 driver stub 是标准构建行为；它不等于已经具有可工作的驱动运行环境。

## 14 项测试实际覆盖什么

使用固定提交的原始 `tests/engine/test_sequence.py`，覆盖：

- 序列由 int32 token 数组构造，包括空数组和 frombuffer view。
- 拒绝错误 dtype、非连续布局和不符合接口约定的输入。
- 压缩 token lookup、越界与负值 sentinel、空输入。
- 输出数组独立性以及 lookup/input 的类型和维度检查。

测试未加载模型，也没有下载 DeepSeek checkpoint。可重用的 [CPU smoke runner](../scripts/e1_core_smoke.py) 会检查源提交、禁止 GPU 设备挂载、启用 HF offline、记录实际测试数与 CUDA 初始化状态。

## 这一结果的有效边界

已证明：原版核心扩展能够在上述 ARM64 工具链上构建、导入，所选 CPU 序列接口测试通过。这为复用 C++ 序列/元数据等组件提供了实际证据。

仅凭本核心构建阶段尚未证明（后续进展见 GPU 报告）：

- DeepGEMM、FlashMLA、DeepEP、DeepSelect 或 TileLang 的 SM121 GPU 算子可用。
- C++ 调度器完整运行、CUDA IPC、CUDA Graph、GPU 缓存管理正确。
- Qwen 权重能由 DeepInfer 加载，或 MTP/GDN/QSA 状态语义正确。
- 推理吞吐、延迟、精度优于现有 TensorFold。

核心编译单元是 C++，没有通过这次构建生成并执行 SM121 算子。即使设置了 `TORCH_CUDA_ARCH_LIST=12.1`，也不能把它当成 SM121 算子验证结果。

## 现有服务与资源复查

实验前后模型服务均为 health HTTP 200 / status=ok，容器与镜像身份、PID、启动时间、启动参数、环境摘要和挂载均不变，重启计数均为 0。模型配置 SHA256 仍与 E0 一致。没有重启或修改现有推理容器。

构建期间一次 SSH 通道中断，经备用连接读取到远程构建成功记录并确认服务状态；不把 SSH 超时写成编译失败或模型故障。

实验容器的 cgroup 记录 OOM / OOM kill 均为 0；存在触达 6 GiB 内存限额的 max 事件。内存统计包含依赖安装和文件页缓存，不代表 GPU 显存用量。结束后只停止实验容器，工作目录、虚拟环境和产物保留供后续复验。

前后快照不能量化瞬时延迟影响，也不能证明整个区间每一毫秒的可用性；本轮未通过额外生成负载测试服务扰动。

## 复现入口

在无 GPU、无权重挂载的实验容器中准备固定提交和 Python 环境，使用：

```bash
bash scripts/e1_build_core.sh
LD_LIBRARY_PATH="/workspace/e1/driver-libs:$LD_LIBRARY_PATH" \
  /workspace/e1/venv/bin/python scripts/e1_core_smoke.py \
  --source /workspace/e1/DeepInfer \
  --report /workspace/e1/artifacts/core-smoke.json
```

脚本应先复制进实验容器可访问的工作目录。Bootstrap 构建复用了基础镜像中的 PyTorch；它不是上游要求的空环境全量安装。不同 PyTorch 版本应重新构建扩展，并使动态库搜索路径优先指向同一环境，避免混用二进制。
