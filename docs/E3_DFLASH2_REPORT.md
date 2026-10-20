# E3：DFlash2 与 DeepInfer 动态验证桥接

**真实 DFlash2 已在 GB10/SM121 上接通完整 Qwen target，并调用 DeepInfer 原生 `MTPVarlen` 动态选择验证长度。** 6 种受控状态检查与 3 个短文本的固定/动态 greedy 对照通过。证据见 [公开 JSON](../results/e3/2026-10-20-dflash2.json)。这是单请求、短上下文、greedy 的实验；完整 DeepInfer scheduler、GPU sampler、MTPInfoPool 和生产服务不属于已接入范围。

## 固定来源

- [DFlash2 权重及配置](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/tree/a22a1744e85c17364dcc54a6a420af7135d4c7f6)：603,065,600 个 BF16 参数，5 层；共享 target embedding/head。
- [SGLang DFlash 实现](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/models/dflash.py)及同版本 `dflash_worker_v2.py`：核对层结构、卷积、selector、anchor、位置和 proposal 行。
- [DeepInfer MTPVarlen](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/mtp_varlen.hpp)：原始头文件由外部固定源码提供，没有复制/重写其调度算法。
- target 继续使用 E2 固定的 NVIDIA NVFP4 checkpoint、BF16 激活和外部 TensorFold 执行源。原生 Qwen MTP 不加载执行。

官方 drafter 发布说明使用 GB300/FP8 target 和 SGLang；这些结果不能当成 GB10/NVFP4 结果。本实验自行验证 Spark 上的权重、状态和耗时。

## 实际实现边界

[参考适配器](../spark_infer_q/dflash2_reference.py)使用真实 BF16 草稿参数、PyTorch 线性层和显式 FP32 attention。它从外部固定 SGLang 文件抽取 `_grouped_conv`、`_score_edges`、`_follow_maps` 的原函数体，去除编译 decorator 后 eager 执行；不安装或复制整个 SGLang runtime。卷积和 selector 另用独立小循环检查。其余模块按固定源码接口组合，**没有完成与完整 SGLang draft forward 的数值对照**，不能将该原型的性能视为最终优化性能。

- target 捕获完成的零基层 `[5,15,25,35,45]`，每层四路 HC stream 求均值，拼成 `[rows,12800]`。原生 forward 的 pending MoE branch 在 shadow 上补齐后才取特征。
- 每轮输入 `[anchor,mask,mask,mask]`，位置从已处理 prefix 长度开始；draft hidden 的第 1～3 行给出草稿，第 0 行是 anchor。
- 草稿的 2048 sliding window 在本次最大 256 行范围内不截断；只实现短上下文、TP1、4 行 block、greedy。没有随机采样的无偏性结论。
- target 验证 `[anchor,*drafts]`。只提交确实处理且保留的行；纠正/额外 token 已输出但仍 pending。GDN 用外部 native `commit(keep<R)` 的 accepted-prefix replay；拒绝后的特征也裁到有效 prefix。
- [C++ 薄接口](../scripts/e3_varlen_bindings.cpp)构造原生 `Sequence/MtpState` 并调用原始 `MTPVarlen::schedule/update_runtime`。不代表完整 scheduler 的请求生命周期、并发、取消和 cache 分配已完成接入。
- 动态预算使用 Spark 实测验证成本及滞后经验接收率；后者是探索性估计，未经独立数据校准，也不是训练好的 confidence head。DFlash2 selector 分数不充当接受概率。
- DFlash2 仍一次生成三个 proposal；动态预算裁剪 target 验证长度，本轮没有动态裁剪 drafter 自身计算深度。

## 验证设计

1. 原生预算器：高/零/中间接收率、双请求分配、滞后轮次、非法计数器及运行时估计。
2. 完整 target：K=0、首/中/末拒绝、全接受、受控停止 token；比较有效 KV/索引/池、GDN recurrent/conv、PLE 历史与窗口、验证及后续 logits。强制草稿用于覆盖状态边界，不计入真实接收率。
3. 三个短文本 prompt：target-only、固定 3 草稿、动态验证，分别输出 16 个 token；同一 decode 算术处理 prompt，保留 E2 prefill 数值问题。每个自然请求逐 token 比较输出与验证 logits，最终比较已提交状态。
4. 计时只包含生成阶段，排除加载、prompt 和校准；原型模式每种每 prompt 一次，属于探索性耗时而非正式基准，不能推断跨任务稳定收益。

已完成的控制测试：原生预算器的高/零/中间接收率分别选择 4/1/2 个验证输入行，双请求得到 `[4,1]`；滞后轮次、非法计数器与运行时估计检查通过。这里验证行数包含 1 个 anchor。

完整 48 层 target 的六种受控状态情况均通过：K=0、首/中/末拒绝、全接受、受控停止。保留前缀的 logits、捕获特征、有效 KV/索引/池、GDN recurrent/conv、PLE 历史/窗口，以及继续执行的 logits，均与逐 token 参考逐 bit 一致。这些受控草稿不计入真实模型接收率。

## 真实权重观测

2026-10-20 的三个自然 prompt，每种模式生成 16 个 token。第一个 token 在计时前已成为 pending anchor，因此每条计时对应后续 15 个新 token；加载、prompt 和校准不计入。各模式的输出 IDs、验证行 logits 和最终有效状态均与 target-only 逐 bit 一致。真实动态运行只观察到 3/4 行预算，不能把控制测试的 1/2 行选择说成自然负载实测。

| 短 prompt | target-only ms | 固定 3 草稿 ms | 动态验证 ms | 固定比值 | 动态比值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 法国首都 | 834.83 | 377.65 | 334.94 | 2.211× | 2.492× |
| 17 + 25 | 705.42 | 388.78 | 417.33 | 1.814× | 1.690× |
| Python 两整数相加 | 786.52 | 458.08 | 394.15 | 1.717× | 1.995× |

按三条总耗时计算，固定模式比值 **1.900×**，动态模式 **2.030×**。固定接收 30/42 个被验证草稿，动态接收 28/39 个；受控强制草稿不在分母里。draft forward/selector 通常约 14–16 ms；选定 startup 曲线的 target 验证+提交约 42–55 ms（1～4 行）。具体逐步数据保留在 JSON。

**这是一次探索性计时，每种模式每个 prompt 仅一次。** 三条输出都还处在 `<think>` 的开头，未形成完整答案，不能据此声称任务正确率、长答案收益或稳定 2× 加速。运行顺序固定为 target-only→固定→动态，没有随机交错；动态 runtime estimator 已吸收此前固定模式成本样本，接收率估计则重新初始化。动态模式在加法样本里反而慢于固定模式；它值得继续校准，尚不能宣布稳定胜出。没有与完整 SGLang draft forward、原生 Qwen MTP 服务或其他服务框架做同条件基准。

## 内存与结束状态

完整 target CUDA 权重仍为 69.8511 GiB。603,065,600 个 drafter BF16 参数约 1.1233 GiB，embedding/head 直接共享；参考实现另有 fused projection 副本和工作区。包含状态分支/诊断的 Torch allocated 峰值 **72.2886 GiB**，reserved 峰值 **72.4063 GiB**。这仅证明本次短上下文、单请求实验能放下；26.8222 GiB n-gram 映射仍按需访问，长上下文、并发和图缓存容量没有结论。

运行结束前记录了 cgroup `memory.events`，全部为 0；其 `memory.peak` 约 3.983 GiB 的记账范围不包含全部 CUDA 统一内存，不能代替 Torch/主机数据判断容量。进程释放权重后已停止实验容器，Docker OOMKilled=false；未部署新服务。原始日志、失败尝试、外部源、二进制及权重留在私有实验区。

本地标准库测试 33 项通过。JSON 保留实际 GPU 运行时的源码摘要；运行后仅修正 controller 在“长度上限之前未输出 EOS”时的停止原因标记，新增回归测试通过。本次 GPU 用例没有同时触发该分支，已测行为不变；未宣称另做 GPU 复跑，修改前后摘要单独记录。

## 下一阶段

先完成 DFlash2 forward 对照，并以完整答案、不同任务/长度及多次交错复测检验收益；经验接收率需要独立数据校准。随后再接 GPU 验证/采样和完整请求调度边界。target 的独立精度参考与 prefill 数值复核继续作为单独主线，不因本次同路径 greedy 一致而关闭。

## 环境问题与复现

本次直接下载网络不可达，本机下载也曾提前截断，经跳板上传过慢。随后使用 Spark 可直连的公开镜像；镜像只是传输途径，元数据及固定文件标识仍来自原站。下载器改用有边界检查和重试的分段续传，最终必须核对完整 LFS SHA256；未校验完成的 partial 不能用来实验。

先前复制的 CUDA driver 路径在当前容器中 `cuInit` 触发 SIGBUS；容器实际选用的 `libcuda.so.610.43.02` compatibility 路径正常，release venv 的 `libtorch_cuda/libcudart` 也已核实并记录摘要。本阶段 launcher 去除历史 `driver-libs` 优先级，保留 release PyTorch/CUDA 依赖路径。没有改系统驱动或全局配置。tokenizers 0.23.3 / safetensors 0.8.0 wheel 安装到 E3 私有依赖目录；SHA256 另存公开 JSON。Torch 为 2.13.0+cu132，TF32 关闭。

复现使用 [下载器](../scripts/e3_fetch_draft.py)、[预算器构建](../scripts/e3_build_varlen.py)、[环境入口](../scripts/e3_run.sh)、[完整 target 适配](../scripts/e3_target.py)和[实测脚本](../scripts/e3_dflash_probe.py)。需要已有 E2 外部源码、扩展、CSF 库、checkpoint 只读挂载，以及本轮固定 SGLang 源；它们不入库。
