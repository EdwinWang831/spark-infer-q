# E2：完整 target 权重加载

**完整 48 层 target 权重已在 SM121 上成功加载。** 主干包含 24,576 个 routed experts、共享专家、GDN/QSA、hyper-connections、embedding、输出 head 和 PLE。n-gram 表使用按需主机映射；MTP/视觉权重按 checkpoint 原格式另行存储，尚未适配执行。

证据见 [load JSON](../results/e2/2026-10-20-full-load.json)。这是权重与选定执行路径验证，不能替代独立整模参考、任务质量、长上下文、多请求容量或性能评测。

最终复测为 2026-10-20 09:41:06–09:45:45 America/Los_Angeles。主干加载约 267.43 秒；本地标准库测试共 28 项通过，其中新增 5 项 checkpoint 边界测试。

## 加载实现

- [Catalog](../spark_infer_q/checkpoint.py) 不读取 tensor payload，先验证固定 config/index 身份、所有 shard/header 对应关系、dtype、shape、字节跨度和总量，拒绝丢失、重叠、截断或未索引数据。
- [CSF bridge](../spark_infer_q/csf_loader.py) 给原版 TensorFold reader 补充虚拟 `weight_scale`：用前轮已验证的原版 Zig C ABI 解码 CSF，再交给原有 CUDA native packer。关闭 read-ahead，限制临时驻留；没有修改外部 loader、forward 或 CUDA 源码。
- 全部 73,728 个主干专家投影标度完成解码。按每层 gate/up/down 的首尾等位置选 5 个投影，共 240 个，与独立 Python CSF 解码逐字节核对；这是抽样独立核对，不是所有投影的双实现比对。
- 保留 NVFP4 routed weights、MXFP8 普通投影及共享专家、BF16 hyper-connection、embedding/head 等原有格式，target 激活继续用 BF16。checkpoint 中的 73,728 个静态 input scales 另外保留，当前 BF16 激活路径不应用它们。
- 使用 101 个固定摘要的外部源码文件，沿用已有 SM121 扩展；没有下载模型、安装包或重新构建内核。首次预检发现 Hugging Face blob 存在跨目录符号链接，改为只读挂载整个 cache hub，并支持索引内安全的 `tensors/` 相对路径。原始失败记录保留。

原版 `Weights.nbytes()` 没有递归统计普通 Python matrix face 对象，会低估 MXFP8 等存储。本轮另外遍历持有的 CUDA tensor storage 并按地址去重；公开 JSON 同时保留原版计数和修正计数，不用原版低值判断容量。

## “完整加载”的含义

| 范围 | 实际处理 |
| --- | --- |
| target 48 层及 embedding/head/PLE | 同时持有可供原生 forward 使用的 CUDA 权重对象 |
| n-gram 的 128 个 shard | 全部只读映射，合计 26.8222 GiB；逐 shard 读取首尾，共 256 行，检查 BF16 解码有限值及摘要 |
| MTP 和视觉的 5,136 个 tensor | 按原 checkpoint 表示读入 CUDA 并保留；没有装配/执行 proposer 或视觉模型 |
| input scales | 原样读取保留，不额外量化 target 激活 |

全部 375,487 个源 tensor 都有明确归属，未覆盖数量为 0。其中 256 个是 n-gram codes/scales 的映射，其余实际读取 payload。映射全表不等于读取或固定驻留所有表页；本轮也未计算全部 checkpoint payload 的密码学哈希，不能宣称做过全文件校验。

MTP 权重读取、MTP 模型适配、投机验证和动态预算是独立里程碑。本轮 MTP 执行关闭，不存在接受率或 MTP 加速结论。动态方向见[接口说明](ADAPTER_INTERFACES.md)。

## 完整 target token/logit 执行与数值复核

固定 token IDs `[100,101,102,103]` 分别走逐 token、4-token decode window 和 4-token prefill；执行真实 embedding、全部 48 层、PLE/n-gram、最终 mixer 与输出 head，词表 248,320。这里没有文本任务、独立整模 oracle 或质量判定，也没有接上 DeepInfer scheduler 或前轮最小请求执行器。

| 对照 | 实际结果 |
| --- | --- |
| 逐 token 和 prefill logits | 所检查值全部有限 |
| 相同 decode 算子，4-token window vs serial | 全部 logits 逐 bit 一致，相对 L2 差 0 |
| 第 2 个 token 后 snapshot→执行→restore→重放 | 后两步 logits 逐 bit 一致 |
| prefill vs serial 的末行 logits | 相对 L2 差 **10.68136%**，触发 2% numerical-review 标记 |
| 末行 argmax | 两路径都为 token 487；仅此不能证明分布或质量一致 |

为定位较大的差异，另记录两路径末行在每层完成后的 hidden states 和 Top-10 专家选择。观察用 shadow hidden/独立 scratch，不改变原始执行；带观测的最终 logits 与原始路径逐 bit 一致。

- layer 0 hidden L2 差约 **0.19118%**，该层专家集合相同。
- 含 PLE 的 layer 1 约 **0.20981%**；差异在进入 PLE 前已经出现，不能只归因于 PLE。
- layer 6 约 **2.51921%**，首次出现末行专家集合变化；所测末行一共有 **15/48 层**选到不同的专家集合。
- layer 47 的四路 residual hidden L2 差约 **14.13741%**，最终 mixer/head 后的 logits 差为上述 10.68136%。逐层数据全部保留。

因此，相同 decode 算子的窗口/串行对照目前没有暴露行映射错误；prefill 差异从浅层出现，向深层累积并跨越路由边界。原生 prefill 与 decode 的 MoE 输出存储/舍入等路径存在已知差别，但这些事实还不足以把 10.68% 宣布为可接受舍入，也没有确定每个算子的贡献。

该百分比不是质量下降比例。当前结论是 **加载通过、选定执行/重放通过、prefill 数值仍待复核**。下一步应以真实输入和独立精度参考定位、评估影响，不能要求参考永远优于候选，也不能未经质量评测把差异解释为改进。

## 内存和运行条件

现有服务已按用户授权停止，启动配置留在私有实验区供恢复。与前轮共存实验不同，本轮没有要求维持原服务在线。实验使用独立 4 CPU/104 GiB 容器、禁止额外 swap，Torch allocator 上限 88 GiB，并在加载边界检查主机至少保留 8 GiB 可用内存。

target 唯一 CUDA storage 为 **69.8511 GiB**。含原格式 MTP/视觉副本和额外标度的加载阶段 Torch allocated 峰值约 **71.7046 GiB**，reserved 约 **72.0410 GiB**；加入 forward/状态副本/诊断后峰值约 **72.8191 GiB / 73.1230 GiB**。最终所记录主机可用内存最低约 **29.6936 GiB**。n-gram 映射的 26.8222 GiB 另记，既不能当成已占用同等物理内存，也不能忽略将来热访问产生的页缓存压力。

结束后实验容器已停止，Docker 的 OOMKilled 为 false。中途采样的 cgroup events 为零，但停止前未成功收集最终 peak/events，公开记录将它们标为 unavailable，不把早期采样充作最终审计。最终核验时旧服务容器已不存在，无法比较其最终配置；此前保存的启动快照仍在私有实验区，本轮未重建旧服务。

这些计数不保证 256K、多请求、MTP、图缓存及所有工作区同时运行时的容量。CUDA allocator、cgroup、主机 MemAvailable 的记账范围不同；尤其不能用较小的 cgroup 数值替代统一内存里的实际 CUDA 分配。加载后进程退出会释放权重，不代表已经部署了新推理服务。

## 复现与下一阶段

[source staging](../scripts/e2_stage_block_sources.py) 使用 `--include-loader`，沿用前轮私有实验区的外部源码、扩展和 CSF C ABI 库。将本项目 `spark_infer_q/`、[runner](../scripts/e2_load_smoke.py)、[环境入口](../scripts/e2_run_load.sh)、[token/logit smoke](../scripts/e2_target_forward.py) 和 CSF Python reference 放入新 `e2-load` 目录；只读挂载 checkpoint cache hub 后，设置 `E2_CHECKPOINT`，运行 shell 的 `--auxiliary --forward-smoke`。第三方源码、权重、二进制及原始运行日志不入库。

后续继续完整 target 的真实输入/logits 对照与调度桥接；再接 Qwen 原生 MTP 的固定 K proposer、验证及 accepted-prefix 状态契约，最后增加动态长度与跨请求预算。独立质量和实际端到端收益分别评估。
