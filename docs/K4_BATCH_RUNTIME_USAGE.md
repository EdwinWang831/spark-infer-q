# K4 双请求异步服务

该入口功能链路通过；[修复前复核](K4_MATCHED_SCALING_DIAGNOSIS.md)发现的主要轮询损失已修复，[同期SSE短组32.57→45.00 token/s](K4_WAIT_FIX_REPORT.md)。目前仍缺广泛负载和长期性能验收。它是显式启用的双请求研究服务，沿用 E33 精确 greedy 辅助头、W8 草稿、E17 NumPy 预测器和当前草稿后动态预算。每个请求独立持有 Target 状态、DFlash 投影缓存、输出和 generation；联合 Decode 使用一次 48 层 Target forward。预填充仍按请求执行八行块。

在已准备好的研究环境中启动：

```bash
E3_ROOT=/path/to/checkout/scripts bash /path/to/prepared/e3_run.sh \
  k4batch_serve.py --root /path/to/prepared --checkpoint /path/to/target \
  --mode adaptive --execution graph --capacity 1536 --port 18084
```

`--mode none` 是 NEXTN=0 Target-only；`--execution eager` 仅关闭 Target backbone Graph，精确筛选头仍使用自己的预备图。两种 mode 不在一个正在运行的服务中切换。没有固定草稿参数。`--capacity` 限制 prompt+output 准入，物理 Target/DFlash 为 2048；范围为 16～1536，正式 GPU 对照的最大提示/输出范围见[报告](K4_BATCH_RUNTIME_REPORT.md)。

默认 legacy Target 启动时加载一次模型，准备 W8、辅助头、两个固定 state slot 和 80 张 Target 图后才输出 READY。Graph 按每个 slot 的 1/2/4/8 行 bucket 与独立 GDN parity 预备。实际 q 仍为动态 1～8；padding 不进入有效状态。缺失/重分配的图走明确的 eager fallback，不在请求中 capture。精确候选子集的新大小可能触发 JIT；Graph READY 不代表所有未知输入的子集形状都已编译。

服务仅监听回环地址。两个终端同时执行以下请求即可提交两个流：

```bash
curl -N http://127.0.0.1:18084/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"请用三点解释缓存与队列的区别。"}],"max_tokens":64,"temperature":0,"stream":true}'
```

这是项目自己的 SSE 格式。响应头 `X-Request-Id` 与首个 `accepted` 事件返回请求 ID；`output` 事件包含该次新增 `token_ids`、完整 `text` 快照、累计计数、计时和 `finish_reason`，终止后发送 `[DONE]`。字节级 tokenizer 的不完整 Unicode 尾部可能修订，所以 `text` 是快照。只支持 greedy、文本 messages 和流式输出；没有完整 OpenAI 协议兼容承诺。

```bash
curl http://127.0.0.1:18084/health
curl http://127.0.0.1:18084/cancel -H 'Content-Type: application/json' \
  -d '{"request_id":"<X-Request-Id 返回值>"}'
```

- 每个请求有自己的有界输出队列；最多两个尚未关闭的流。满额/准入超时返回 409，输入错误 400，worker 故障 503。已经开始的流用 `error` 事件报告错误。
- HTTP handler 只提交、接收自己的输出和取消。唯一 owner 线程驱动原生调度器，轮询真实 CUDA event；verify/commit fence 完成后才向 native 发布和回收 slot。仍保留草稿/预算等必要主机同步。
- 显式取消、断连或慢消费者会取消对应 generation。慢消费者收到 `slow_consumer` 错误，不会静默丢失输出后报告成功。断连通过写失败检测，GPU 忙时每秒发 SSE 心跳。
- API 的同 ID 重用受完整 handle/generation 保护。HTTP ID 由服务生成，`/cancel` 作用于该 ID 的当前流。
- 停止服务时先停止 worker 并完成 fence，再回收 Target 和 native Gloo。关闭超时明确报错并保留在途状态；native owner guard 防止活跃 worker 的进程组被提前释放。worker 故障后隔离，需创建新 owner；不会自动重试旧请求。

默认预算仍保留单请求成本目标，`global_budget_optimized=false`。新增的[实测批次联合预算](K4_JOINT_BUDGET_REPORT.md)已通过正确性检查，但正式SSE性能未过关，只作为显式实验选项：在上述启动命令追加以下参数。

```bash
--batch-costs /path/to/checkout/results/k4/2026-10-24-joint-costs.json \
--batch-costs-sha256 494337bd1dcfa027041e65d3d2669687d8b01fb26d4e36f458143670f842390a
```

两个参数必须同时提供，且要求adaptive与Graph模式。加载时核对文件SHA、E17/W8成本指纹和Target源码；不匹配会明确失败。启用时`verification_budget=joint_current_e17`、`global_budget_optimized=true`，这里的联合优化仅指当前已调度decode批次，不包含公平性、准入或完整K5。E17权重和预测特征不变，q仍动态选择。

新增[自有批量 DFlash 研究开关](K4_OWN_BATCH_DRAFT_REPORT.md)：`--draft-execution batch`，配合显式 W8 batch library/SHA。它保留 E17 与原动态预算，拒绝旧联合成本表；草稿数值与部分候选会改变，性能未达到晋升目标，因此默认仍为 sequential。四请求仅在草稿组件检查，服务仍只有两个 slot。

新增[自有紧凑 Target 研究开关](K4_OWN_TARGET_REPORT.md)：

```bash
--target-backend compact \
--target-state-library /path/to/spark_q_target_state_batch.so \
--target-state-sha256 b63a8c527c56b2aec964fba81352bed9bb16b510db32cfe14b4d3f19ee2b64e2
```

该 SHA 是本轮已测二进制；重新构建必须传入实际新 SHA 并重新验收。默认仍为 `--target-backend legacy`；指定 compact 可与 batch draft 组合，E17/动态预算继续保留。旧 `--batch-costs` 表不适用于 compact，CLI 会拒绝。Target 状态库只能与 compact 一起提供。

compact 仅支持当前 TP1 文本、BF16 KV、2048 内 dense-budget QSA。按实际总验证行压紧，不为每个请求单独补到 bucket；自有状态和 GPU 指针表隔离请求。Graph 按实际行数/context 复用，默认最多 32 张，第二次遇到形状才捕获并执行，满额按 LRU 驱逐。未知形状可在请求中触发 eager、捕获或头部 JIT；READY 不表示这些冷开销已经支付。正式性能对照先预热，并要求计时期间没有新增 capture/JIT。prefill 与部分算子仍通过 TensorFold provider。

后续[Prefill 推进修复与服务测量](K4_PREFILL_PROGRESS_REPORT.md)增加以下显式选项，可在上述 compact/batch 参数上追加：

```bash
--interleave-prefill \
--prefill-execution owned \
--prefill-chunk-rows 128 \
--prompt-token-budget 128 \
--worker-cpu performance
```

`--interleave-prefill` 在长 prompt 分块之间给已 ready 的 decode 留出轮次，仍使用 DeepInfer 原生计划/记账，仅支持双 slot 单 rank。owned prefill 使用已提交的 E40 executor，保留当前模型的精确 logits/features/state 交接；底层权重/算子依赖仍显式存在。当前完整服务测试的 prompt quantum 为128；允许参数范围不等于所有长度都已经验收。

`--worker-cpu performance` 仅在 Linux 将 owner 线程限制到其现有允许集合中 `cpuinfo_max_freq` 最高的一组 CPU，退出时恢复旧 mask；不改进程其他线程、频率或 governor。在本轮 Spark 上选择 CPU5～9/15～19。默认 `--worker-cpu default` 不触碰 affinity。无法读取拓扑或设置 mask 会明确失败。该开关的收益与异构 CPU/系统放置有关，不是通用 GPU kernel 加速承诺。

默认仍为 rows8、prompt budget32、原 prompt 优先和系统 CPU 放置；上述方案按可选研究配置提供，E17 与动态预算不变。完整请求/SSE cohort 的口径、固定到达顺序与五轮数据见报告，不与单流 decode 数字混用。

本入口仍限定单卡双请求；4请求完整服务、4K+、随机采样、跨卡、生产压力和公网部署尚未验证。
