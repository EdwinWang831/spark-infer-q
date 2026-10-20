# Qwen / DeepInfer 接口清单

此清单来自固定 DeepInfer `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` 的源码检查，**是适配设计与待测契约，不是已经实现或验证的 Qwen 接口**。后续已完成[两类原版 TensorFold block 的选定状态检查](E2_BLOCK_REPORT.md)，以及[自有最小执行器的真实 block 生命周期检查](E2_EXECUTOR_REPORT.md)。DeepInfer C++ scheduler 接入、完整模型和 MTP 正确性仍未证明。

## 引擎与模型边界

| 边界 | 当前 DeepInfer 来源 | Qwen 适配要求 |
| --- | --- | --- |
| 请求生命周期 | `csrc/engine/sequence.hpp`、`scheduler.hpp` | 稳定 request ID、prefill/decode/cancel/recompute 状态、slot 分配回收和异步完成代次 |
| 模型输入 | `model_executor/input_metadata/input_metadata.py` | 明确 token IDs、positions、请求 query 范围、有效 token mask、state handle；不把 V4.1 压缩比和 KV 布局写入通用接口 |
| 模型输出 | `model_executor/layers/sampler.py` | logits 与输入行一一对应，明确哪些状态已经计算；采样输出与模型状态提交分别管理 |
| 设备存储 | `worker/cache_engine.py` | 当前 `StateBlock` 是 V4.1 SWA/YoCo/compression 布局，需独立定义 Qwen GDN recurrent/conv state 和 QSA KV/索引等状态的分配、保存及恢复 |
| CUDA Graph | metadata 的 copy/clean 方法及 graph runners | 稳定 buffer 地址、有效请求范围、padding 不修改真实 state；RMSNorm 图测试不覆盖整个引擎 |
| 并行依赖 | metadata/parallel utils 与 V4 模块直接导入 FlashMLA/DeepEP | 新 Qwen 适配边界应避免因无关模块导入而强制依赖未适配的 SM100/多 GPU 后端；单 GPU 运行标志不等于已经解除导入依赖 |

建议模型适配接口显式提供 `prefill`、`decode/verify`、`snapshot`、`commit_prefix`、`restore` 和 `release`；方法名是提案。每个 state handle 应带请求身份、代次和实际已处理位置，防止请求取消/slot 复用后应用过期 GPU 结果。先实现 MTP 关闭时的单步状态连续性，再测试分支执行。

## 已实现的最小边界

`Executor` 已实现 `admit/prepare/dispatch/finish/cancel/checkpoint/restore`，`Row` 显式携带请求 handle 和位置；handle 包含 owner、slot 和 generation。后端隐藏状态先私有推进，设备完成后发布逻辑位置；取消后槽位排空再释放。`TensorFoldBlockBackend` 执行真实独立 GDN/QSA block，压紧有效行并散回输出，完整快照包含已提交 KV prefix；graph 绑定 state 版本/容量/代次/parity，恢复/扩容/释放使其失效。输入生命周期和返回 tensor 的消费 stream 有显式依赖。

最终 GPU 记录含 16 项状态/生命周期检查、2 项数值诊断和 2 项局部计时。此原型串行执行、默认 2 请求/context 64，每次 dispatch 复制完整前缀；没有 token IDs、embedding、lm_head、采样或 DeepInfer metadata 桥接。它为上表的接口提供已测实现材料，不能把上表整体标为完成。详见[实现及证据](E2_EXECUTOR_REPORT.md)。

## MTP 数据流

固定源码中的实际契约：

1. `MTPInfoPool` 的 slot 0 保留为 padding，真实 request slot 各持有 `nextn` 行。保存 proposal IDs、proposal logprobs、可选 top-k ID 映射及可选 confidence；不能只有草稿 token 而丢失随机验证需要的 proposal 分布。
2. `MTPDecodeInput.first_token_positions` 将请求映射到 main varlen 行，`-1` 表示 CUDA Graph padding。
3. `MTPDriverInput.last_valid_offset` 表示本轮接受的**草稿数量**，`golden_tokens` 是验证后采样、用于下一轮草稿的 token，`MTPSamplingInput` 携带采样参数、seed/position 和 pool 写槽。
4. C++ `MtpState` 保存 accepted/draft tokens、两轮接受概率估计及 schedule/process counters。native vector 字段暴露给 Python 时是 list 快照，不能靠原地修改该 list 更新 C++ 状态。
5. `mtp_varlen.hpp` 用滞后概率和运行时成本估计选总 query 预算，再分配各请求长度。保留其避免后续草稿影响更早验证资格的约束，或独立证明替代方案。

适配时须区分 **已输出长度** 和 **已由 target forward 处理的长度**。验证后采样出的纠正/额外 token，可能尚未作为输入经过 target；不能将“接受 k 个草稿并输出一个新 token”直接解释为 target state 已推进 k+1。对每个阶段记录 anchor/base position、accepted prefix、pending token 和实际 processed length，并与逐 token 执行对照。

Qwen GDN 状态不能仅靠删去拒绝 token 的 KV 恢复。需要目标状态和草稿状态分别保存有效 checkpoint，提交接受前缀或从有效 checkpoint 精确重放。V4.1 的 bounded SWA replay 不自动具备 Qwen GDN 的等价语义。Qwen 原生 MTP 也不能直接使用 DFlash 的 forward/已训练 confidence 权重。

## 后续验证门槛

- 引擎：请求进入/取消/回收/重算，slot 复用与异步过期结果，padding 不写入真实状态。
- 无 MTP 状态：相同前缀分块 prefill 和逐 token decode 的 logits/state 对照。
- MTP：K=0、全接受、首/中/末拒绝、EOS；每种情况下 processed length、pending token 和 GDN/QSA 状态与参考一致。
- 采样：greedy 一致性；随机采样使用分布检验，单独标明上游 `DRAFT_PROB_PENALTY` 等近似。
- 调度：固定 K 后再验证自适应预算的正确性、校准与实际总耗时收益。

以上完整 DeepInfer 适配器验收仍未执行。原版 block 的前轮测试覆盖 prefill/decode、snapshot/restore、reset/growth 和两请求布局；后续自有执行器进一步验证取消/复用、padding、完整 prefix 恢复、连续图解码及图失效。历史 CPU 全路径的一行 GDN 专家集合差异保留；新独立 GPU 参考在同一批输入上集合一致，但仍有数值差异，且未验证整模 logits。既有 14 项 CPU sequence 测试和这些局部测试都不能替代引擎接入验收。

源码入口：[sampling inputs](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/sampling_inputs.py)、[MTP pool](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/mtp_info_pool.py)、[sampler](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/sampler.py)、[cache engine](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/cache_engine.py)、[scheduler](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/scheduler.hpp)。
