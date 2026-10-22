# E5：真实 DeepInfer Scheduler 到完整 Qwen 的最小桥接

日期：2026-10-21（America/Los_Angeles）。[固定范围](E5_ENGINE_BRIDGE_PLAN.md)、[完整证据](../results/e5/2026-10-21-engine-bridge.json)。状态：PASS（限定同步、target-only 桥接）。

本轮新增的是 **DeepInfer 原版 C++ Scheduler 实际产生请求计划，并由现有完整 Qwen target 执行，完成后提交回原生输出账本**。它已超出只调用 `MTPVarlen` 或使用自有 Python 请求队列的范围；仍不是完整 `LLMEngine`、Worker、GPU sampler、异步流水或 GPU KV/cache 管理移植。E3/E4 的 DFlash2 性能路径没有因此自动切换到新桥接器。

## 实际接线

```text
RequestHandle(slot, generation)
  -> native Scheduler.schedule / SequenceMetadata
  -> Qwen backend advances actual GDN/QSA state
  -> backend completion / synchronous GPU fence
  -> native Scheduler.process_outputs
  -> EOS / cap / cancel / native block release + device state release
```

| 项目 | 实际配置 |
| --- | --- |
| DeepInfer 源码 | `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` |
| 实际核心扩展 SHA256 | `f6aa14b790eb7bc729251eea152fd9c83cf215bc5ab912e25ee19c57d8fc1768`，沿用 E1 release 构建 |
| Torch / 设备 | `2.13.0+cu132` / DGX Spark GB10 SM121 |
| 主模型 | 完整 48 层 NVIDIA FP4 Qwen checkpoint，revision `b9ce18a6887e18c089fdea8509a33e63429a3300` |
| 请求 / token budget | 最多 2 个活跃句柄，原生每步总 query budget 32，逻辑 block size 16 |
| CPU / GPU context | 128 / 256；EOS 与 token 上界使用实际模型输出矩阵的 ID 空间 |
| 模式 | target-only、greedy、BF16 KV；`DEEPINFER_USE_NEXTN_DECODE=0` |
| 原生进程组 | 单 rank Gloo，满足原版 Scheduler 接口；不使用 NCCL/多卡 |

原生 BlockManager 管理的是逻辑配额与 token 计数，**这些 block ID 尚未映射为 Qwen 的 GPU KV 页**。真实 recurrent/conv、KV/index 状态由 Qwen backend 独立拥有；不调用 V4.1 的 clone/zero/fork 路径。`log_stats=False`，FlopsEstimator 不提供；未伪造 Qwen 的 DeepSeek 架构元数据或 MFU 数字。

prepare / execute / complete 分开：同一时刻只允许一份未完成计划，execute 返回前后端必须完成同步，complete 才能发布 token。取消后若已有计划仍持有槽位，必须收尾后释放；重复完成与旧 generation 句柄会被拒收。原生 abort 可以处理队列中的请求；已被取出队列、尚未完成的 prompt 由适配器直接标记其持有的原生 Sequence 为 aborted。不是对异步 Worker 的完整取消竞态验收。

## 检查结果

CPU 使用真实 C++ Scheduler + toy backend，七组场景通过：**110 个调度请求片段、12 个完成/取消请求**，无 CUDA 初始化，所有逻辑 block 归还。不是 mock Scheduler，也不计入 GPU 模型检查。

另用真实 checkpoint tokenizer 和冻结的 E4 输出作 CPU 回放，同七组场景通过：**79 个请求片段、12 个完成/取消请求**。它核对真实 token ID、模板、词表/EOS 边界及调度路径，没有 GPU forward；不是额外七种独立场景或模型生成证据。

GPU 检查状态：PASS。GPU 事件轨迹、输出 IDs、停止状态和最终原生处理位置与真实 token CPU 回放逐项一致。七组场景合计 **79 个调度请求片段、364 行完整 logits 检查、79 次有效状态检查、12 个完成/取消请求**。

| 场景 | 检查重点 |
| --- | --- |
| 单请求 EOS | 请求推进、完整生成、提前 complete 拒收、重复 execute/complete 拒收 |
| 两请求分块 prompt / 交错 decode | JSON 与纯 Python 函数，chunk 进度、交错请求、独立状态 |
| 长度上限 | 只生成 3-token 代码前缀，停止准确；不称完整正确程序 |
| waiting 取消后复用 | 未执行请求不发布输出，代次递增 |
| prepared prompt 取消后复用 | 已计算但未发布的结果被丢弃 |
| prepared decode 取消后复用 | 已有 prefix 的请求取消后不追加结果，旧通知不能用于新句柄 |
| 非法准入 / 满槽 | 空/越界 token、无效长度、上下文溢出、重复 ID 与超额请求拒收；本组不执行 GPU forward |

GPU 复用 E4 的整数、JSON 和 `sum_positive` 三个英文 direct 提示作为回归，不是新的质量/泛化测试。每个实际 forward 的输入 IDs/位置对照独立生成的本路径 serial 轨迹；全部输出 logits 对照保存的 serial logits，并由单独 shadow 状态按期望 token 推进，再检查有效 GDN/QSA 内容。活跃请求在调度准备及完成时都核对原生 `num_cached_tokens` 与真实设备位置。取消后私有设备计算可以比未提交的原生位置多一步，但不发布，安全收尾后整份丢弃。

这些检查验证调度接线与状态连续性，**不构成独立整模数值正确性证明**。此前 target prefill/serial 数值差异和官方 drafter BF16 fidelity 问题保持开放。

## 故障与修正

初次 CPU 预检因容器与源码目录 owner 不同，Git 拒绝读取 revision；改为对该命令显式指定唯一源码目录的 `safe.directory`，没有更改全局 Git 配置或上游源码。

第一次完整权重加载后，GPU 场景尚未 dispatch 就被 EOS 准入检查拒绝：runner 错用 `tokenizer.vocab_size`，该基础词表不含新增特殊 token。修正为实际 `weights.head.b.weight.shape[0]`，CPU/GPU 使用同一新 runner 哈希重新检查。未更改 checkpoint、EOS 定义、输入或目标 logits。初次失败保留私有原始记录，不计入通过次数。

第二次加载后，serial oracle 准备暴露了另一个 runner 错误：`apply_chat_template` 的默认返回是字典，却被当成 token 列表使用。修正为显式 `return_dict=False`；增加了完整权重加载前的真实 tokenizer/ID 检查，以及前述 E4 输出 CPU 回放。预检同时修正了读取本 checkpoint 嵌套 `text_config.vocab_size` 的字段位置。两次 GPU 尝试均未完成任何场景，不算通过；最终数据来自修正后的独立重跑，而非拼接失败尝试。

## 资源、范围与下一项

成功运行 target load 286.04 秒，总 runner 349.52 秒；这些包含加载、serial oracle 和状态诊断，**不是速度基准**。PyTorch peak allocated/reserved 为 70.90 / 71.77 GiB，不是统一内存的全机峰值。结束资源状态：模型 worker 已退出；cgroup memory events 各项为零；实验容器 exited、OOMKilled=false。

分块调度仍逐 token 执行 prompt；多请求计划在 GPU 上仍逐请求执行。未证明批量 prefill、GPU ragged batch、CUDA Graph、异步流水、抢占重算、随机采样、GPU 故障恢复、长上下文或服务性能。异常执行会隔离该 bridge，不能据此宣称已实现设备故障恢复。完整 DeepInfer Worker/cache/sampler 与 DFlash2 此桥接路径尚未接通。

本轮是对 [全局调度 RFC PR #2](https://github.com/EdwinWang831/spark-infer-q/pull/2) 的早期引擎接口验证，不等于实现了 RFC 的全局 Cost-Aware 调度。下一项按约定进入 **DFlash2 草稿头专项**：先分离 head 成本，再依次比较缩词表、量化与组合，检查提案/接受率、完整耗时、总内存和输出/logits/状态。后续以相同执行器对照策略，分离引擎优化与预算优化的收益。

真实批量执行的后续入口已找到：固定 TensorFold `forward.py` 中 `stage(windows)` 支持多请求分段，`compute(segs)` 统一执行，`commit(..., at=...)` 按偏移提交。先从两个请求各一行 decode 开始，检查专家联合分组以及 GDN/QSA 请求边界，再扩展统一长度草稿验证、不同长度分段与 CUDA Graph bucket。函数存在不等于完整模型批量路径已经通过；总吞吐、单请求 TPOT 与尾延迟需同时测量。

代码：[桥接器](../spark_infer_q/deepinfer_bridge.py)、[真实调度/模型检查](../scripts/e5_bridge_smoke.py)、[证据导出](../scripts/e5_export.py)。外部内核与权重不随仓库发布。
