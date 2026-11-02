# Kaifra v0.1：统一推理服务

[English](ENGINE_V01_USAGE.en.md) · [验收方案](ENGINE_V01_PLAN.md) · [真实验收报告](ENGINE_V01_REPORT.md)

v0.1 将已提交的 K4 异步运行时与主线整合，提供统一命令行和聊天 API。当前支持单台 DGX Spark、已准备好的 NVFP4 Qwen/DFlash2 环境、BF16 target KV、文本 greedy、两个运行请求，以及有界等待队列。动态模式继续使用 E17 预测器与当前草稿后验证预算。

**能力边界：** 单请求的 prompt+output 准入上限为1536，物理状态容量2048。已有256K适配器、四并发未提交实验、压缩KV和RSI方案保持各自的研究状态，未自动成为此服务的能力。模型加载、算子和原生调度仍依赖准备好的 TensorFold/DeepInfer 环境。

## 启动

在已有 GPU 研究环境中使用 Python3.12及以上版本：

```bash
python -m pip install --no-deps -e .
kaifra doctor --root /path/to/prepared/runtime --checkpoint /path/to/local/target
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target --port 8000
```

也可直接从仓库运行 `python -m spark_infer_q doctor ...` 和 `python -m spark_infer_q serve ...`。`--help`、`--version` 和 doctor 只用标准库；doctor 检查本地文件，不导入 PyTorch、探测 CUDA、加载权重或下载文件。启动时还会执行原生源码/扩展、预测器和成本artifact的身份检查。

`--root` 是已经准备好的运行环境，包含 TensorFold/target 适配器、DFlash checkpoint、Q4/W8/varlen 扩展和冻结的 E17/E29 artifacts；不是空白的模型下载目录。现有研究目录的准备方式见[常驻 worker](RESIDENT_RESEARCH_WORKER.md)。若需要其 linker 环境，保留原 prepared `e3_run.sh`，只把入口换为本分支的脚本：

```bash
E3_ROOT=/path/to/checkout/scripts bash /path/to/prepared/runtime/e3_run.sh \
  serve.py serve --root /path/to/prepared/runtime \
  --checkpoint /path/to/local/target --port 8000
```

启动按一次 session 加载、Graph/工作区准备、owner启动、HTTP监听的顺序执行，最终输出 READY。模型在服务存活期间常驻，正常请求不会重新加载。现有 resident session 可通过 `open_service(session, ServiceConfig(...))` 直接复用，见[服务工厂](../spark_infer_q/engine_server.py)。

## 聊天 API

默认监听 `127.0.0.1`。支持 OpenAI chat 的文本 greedy 子集：

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"spark-infer-q","messages":[{"role":"user","content":"解释缓存与队列的区别。"}],"temperature":0,"max_tokens":64}'

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"spark-infer-q","messages":[{"role":"user","content":"解释缓存与队列的区别。"}],"temperature":0,"max_tokens":64,"stream":true,"stream_options":{"include_usage":true}}'
```

非流式返回 `chat.completion` 和 usage；流式返回 `chat.completion.chunk`、增量 `delta.content`、结束原因以及 `[DONE]`。可选 `include_usage` 在最后返回用量块。解码器尚未完成的Unicode尾部会延迟发布，避免先输出替换字符后无法撤回。当前聊天模板关闭 thinking。

支持字段为 `model/messages/max_tokens/temperature/stream/stream_options.include_usage`。仅接受 `temperature=0`；额外采样、工具、结构化输出、多模态和完整 OpenAI API 兼容不属于此版本。

流式响应头 `X-Request-Id` 可用于取消：

```bash
curl http://127.0.0.1:8000/cancel -H 'Content-Type: application/json' \
  -d '{"request_id":"<X-Request-Id>"}'
```

队列满或准入超时返回429，输入错误400，故障owner返回503。断连、显式取消和慢消费者只取消所属generation；状态在device fence完成后释放。取消的聊天回复以 `stop` 结束；非流式扩展字段 `kaifra_finish_reason` 保留原生取消原因。

## 配置、指标与停止

默认两个运行请求，另允许四个等待/未关闭流位置。`--queue-size 0` 关闭额外等待容量，`--admission-timeout 30` 限定准入等待，`--capacity` 限定 prompt+output。`--mode none` 启动同路径 target-only 服务；`--execution eager` 关闭 target backbone Graph。默认 owned prefill chunk128、prompt budget128，并在长prompt分块之间保留ready-decode机会。

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/metrics
```

health包含实际能力与owner状态；指标使用 Prometheus 文本格式，记录运行/等待请求、准入拒绝、完成/取消/失败、已发布token和延迟汇总。TTFT/request延迟从引擎准入后计算，不含等待队列时间，不是客户端端到端延迟；累计token计数包含取消前已发布部分。HTTP字段校验失败不进入worker准入计数。

Ctrl-C或SIGTERM先停止接收，取消/完成在途工作，等待GPU fence及owner线程，再关闭handler、工作区和原生进程组。关闭超时会保留仍在使用的状态并明确报错。服务故障后需要新owner，不会静默重试已发布请求。

原 `scripts/k4batch_serve.py` 和默认 `kaifra_service.make_server` 继续保留研究 SSE 格式；新统一入口显式选择 OpenAI chat 子集，两种协议不混用同一个监听端口。

## 验证入口

```bash
python -m unittest discover -s tests -p 'test_engine*.py' -v
python -m unittest discover -s tests -v
```

CPU HTTP测试使用真实服务和owner loop，设备边界使用确定性fixture。独立的完整模型、状态和冷启动验收均已完成，范围见[报告](ENGINE_V01_REPORT.md)；CPU测试和打包结果仍单独报告。
