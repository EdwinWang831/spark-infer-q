# K4 双请求异步服务

该入口功能链路通过；[修复前复核](K4_MATCHED_SCALING_DIAGNOSIS.md)发现的主要轮询损失已修复，[同期SSE短组32.57→45.00 token/s](K4_WAIT_FIX_REPORT.md)。目前仍缺广泛负载和长期性能验收。它是显式启用的双请求研究服务，沿用 E33 精确 greedy 辅助头、W8 草稿、E17 NumPy 预测器和当前草稿后动态预算。每个请求独立持有 Target 状态、DFlash 投影缓存、输出和 generation；联合 Decode 使用一次 48 层 Target forward。预填充仍按请求执行八行块。

在已准备好的研究环境中启动：

```bash
E3_ROOT=/path/to/checkout/scripts bash /path/to/prepared/e3_run.sh \
  k4batch_serve.py --root /path/to/prepared --checkpoint /path/to/target \
  --mode adaptive --execution graph --capacity 1536 --port 18084
```

`--mode none` 是 NEXTN=0 Target-only；`--execution eager` 仅关闭 Target backbone Graph，精确筛选头仍使用自己的预备图。两种 mode 不在一个正在运行的服务中切换。没有固定草稿参数。`--capacity` 限制 prompt+output 准入，物理 Target/DFlash 为 2048；范围为 16～1536，正式 GPU 对照的最大提示/输出范围见[报告](K4_BATCH_RUNTIME_REPORT.md)。

启动时加载一次模型，准备 W8、辅助头、两个固定 state slot 和 80 张 Target 图后才输出 READY。Graph 按每个 slot 的 1/2/4/8 行 bucket 与独立 GDN parity 预备。实际 q 仍为动态 1～8；padding 不进入有效状态。缺失/重分配的图走明确的 eager fallback，不在请求中 capture。精确候选子集的新大小可能触发 JIT；Graph READY 不代表所有未知输入的子集形状都已编译。

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

每请求预算保留原有单请求成本目标；`global_budget_optimized=false`。批次成本建模、公平性和全局行数分配属于 K5。本入口限定单卡双请求；4K+、随机采样、跨卡、生产压力和公网部署尚未验证。
