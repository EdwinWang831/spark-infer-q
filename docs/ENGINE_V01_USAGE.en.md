# Kaifra v0.1: unified local inference service

[简体中文](ENGINE_V01_USAGE.md) · [Validation plan](ENGINE_V01_PLAN.md) · [Validation report](ENGINE_V01_REPORT.md)

This integration combines the committed K4 asynchronous runtime with current main. It targets the prepared NVFP4 Qwen/DFlash2 deployment on one DGX Spark: BF16 target KV, greedy text, two running requests and a bounded admission queue. Adaptive mode retains E17 and the current post-draft verification budget.

The public context admission limit is1536 tokens including output, with2048 physical capacity. The256K adapter, uncommitted four-request work, compressed KV and RSI proposal remain separate research capabilities. TensorFold execution primitives and DeepInfer's native scheduler remain external dependencies.

## Start

Inside the prepared GPU environment, with Python3.12 or newer:

```bash
python -m pip install --no-deps -e .
kaifra doctor --root /path/to/prepared/runtime --checkpoint /path/to/local/target
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target --port 8000
```

The equivalent source entry is `python -m spark_infer_q`. Help, version and doctor require only the standard library. Doctor checks files, without importing Torch, probing CUDA or loading/downloading weights; startup performs additional pinned-source, extension and artifact checks.

The prepared root contains the external target adapters, DFlash checkpoint, Q4/W8/varlen extensions and frozen prediction/cost artifacts. If the deployment needs its existing linker environment, retain that launcher and select the new entry:

```bash
E3_ROOT=/path/to/checkout/scripts bash /path/to/prepared/runtime/e3_run.sh \
  serve.py serve --root /path/to/prepared/runtime \
  --checkpoint /path/to/local/target --port 8000
```

The CLI loads a session once, prepares workspaces/Graphs and starts the owner before reporting READY. Requests reuse that model. An already resident session can use `open_service(session, ServiceConfig(...))` directly; see [the implementation](../spark_infer_q/engine_server.py).

## HTTP boundary

The listener is loopback-only. `GET /v1/models` lists the configured model. `POST /v1/chat/completions` supports text messages, greedy generation and both streaming and nonstreaming responses:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"spark-infer-q","messages":[{"role":"user","content":"Explain caches and queues."}],"max_tokens":64,"temperature":0}'
```

Use `"stream":true` for incremental `chat.completion.chunk` events and `[DONE]`; optional `"stream_options":{"include_usage":true}` adds a final usage chunk. Incomplete Unicode suffixes are delayed until they can be published without revision. The current template disables thinking.

Supported fields are `model/messages/max_tokens/temperature/stream/stream_options.include_usage`. Only temperature0 is supported. Sampling, tool calls, structured output, multimodal inputs and complete OpenAI compatibility are outside this version.

Streaming response header `X-Request-Id` identifies a generation for `POST /cancel` with `{"request_id":"..."}`. A cancelled chat ends with `stop`; nonstreaming extension field `kaifra_finish_reason` retains the native reason. Queue saturation/admission timeout returns429, invalid input400, and unavailable owners503. Cancellation, disconnect and slow-consumer handling retain GPU state until the relevant fence completes.

## Bounds and lifecycle

There are two running slots plus four additional admission/open-stream slots by default. Configure `--queue-size`, `--admission-timeout` and `--capacity` explicitly. `--mode none` selects target-only execution; `--execution eager` disables target-backbone Graphs. Defaults use owned prefill with128-row chunks, prompt budget128 and ready-decode interleaving.

`GET /health`, `/healthz` and `/ready` expose owner state and actual capabilities. `/metrics` exposes Prometheus text counters for admission, completion/cancellation/failure, published tokens and engine latency. TTFT/request durations start at engine admission and exclude queue wait; they are not client end-to-end timings. Tokens published before cancellation remain counted; HTTP schema rejections do not enter worker admission counters.

SIGINT/SIGTERM stops acceptance, drains/cancels device work and joins the owner before releasing buffers and native ownership. A timed-out owner retains its in-use state and reports an error. A failed owner requires explicit recreation; published requests are not silently retried.

The old K4 script and default service factory retain their research SSE format. The new public entry explicitly selects the OpenAI chat subset.

## Validation

```bash
python -m unittest discover -s tests -p 'test_engine*.py' -v
python -m unittest discover -s tests -v
```

CPU tests exercise real HTTP and the real owner loop with deterministic device fences. Separate full-model/state and cold-start validation passed within the documented scope; see the [report](ENGINE_V01_REPORT.md). Packaging and CPU checks are reported independently.
