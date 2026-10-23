# K1 Target-only 运行时

`KaifraEngine` 扩展 E5 桥接，继续使用真实 DeepInfer C++ Scheduler / BlockManager。`QwenTargetBackend` 复用已加载的 TensorFold Target，设备权重和共享 buffers 由外部 session 持有，请求自己的 GDN/QSA/PLE state 在结束或取消后释放。不得同时在其他线程/实验中直接使用该 Target；运行时锁只能保护经过自身 API 的调用。

## 已有 session 中调用

```python
from spark_infer_q.native_scheduler import NativeScheduler
from spark_infer_q.kaifra_runtime import KaifraEngine

with NativeScheduler('/workspace/e1/DeepInfer',
                     '/workspace/e1/build-release/lib/deepinfer') as native:
    with KaifraEngine.from_native(native, session.target, eos_ids=session.eos,
            tokenizer=session.tokenizer, capacity=256, max_requests=2) as engine:
        handle = engine.submit('example', [{'role': 'user', 'content': '你好'}],
                               max_tokens=32)
        for event in engine.drain():
            print(event.token_ids, event.text, event.finish_reason)
```

也可用 `submit_tokens()`，但 token ID 上界以真实模型 head 行数为准。`step()` 返回本次调度后新增的输出事件；`cancel(handle)` 在同步 step 边界生效，下一次 step/drain 产生 `cancelled` 终止事件。旧 slot/generation 句柄会被拒绝。终止历史由调用方保存，引擎自身不累积全部历史输出。`close()` 取消并回收剩余请求，但不继续向调用方输出取消事件。

事件的 `token_ids` 只包含本次新增、已由原生账本确认的 token；`text` 是累计文本快照，不能当作增量字符串追加。最后输出的 token 尚未作为 Target 输入处理，因此 `processed_position` 通常等于 prompt 长度加输出长度减1。EOS token 计入 completion_tokens。

TTFT 从 token 化结束、进入原生准入前开始，到首次已提交输出，包含设备状态分配及准入后的排队；不包含 HTTP 收包、token 化或等待运行时锁。decode_seconds 从首次输出到当前 step 的墙钟时间，包含请求交错和调用者未调用 step 的间隔。事件的 elapsed_seconds 同样从此准入边界计时，不是完整 HTTP 请求耗时。这些不是纯 GPU 时间，也不是性能基准。错误隔离后拒绝新请求；只有成功完成设备 fence 才允许清理请求状态。设备 context 故障会明确失败，不能靠重新提交恢复。

## 本地 HTTP / SSE

单独启动时通过现有 prepared root 加载一次 Target；不启动 TensorFold HTTP 服务，不加载 drafter。**已有常驻模型占用 GPU 时应复用 session，不要同时另开此入口。**

```bash
E3_ROOT=/path/to/checkout/scripts bash /path/to/prepared/e3_run.sh \
  kaifra_serve.py --root /path/to/prepared --checkpoint /path/to/checkpoint \
  --capacity 1024 --prefill-rows 8 --port 18083

curl -N http://127.0.0.1:18083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":32,"stream":true}'
```

只监听 loopback。响应头 `X-Request-Id` 和首个 SSE `accepted` 事件包含请求 ID。`POST /cancel`，JSON `{"request_id":"..."}`，可在下一同步边界取消；客户端断开后也会取消并回收。`GET /health` 返回运行状态和能力。HTTP 入口一次只允许一条 generation，第二条返回409；同进程 API 允许两个请求交错，GPU 仍逐请求执行。

SSE 是本项目的快照事件协议，字段包含 token_ids、text、finish_reason 和指标；路径名称不代表完整 OpenAI 协议兼容。greedy/text-only，temperature 必须为0；unsupported 字段会拒绝。没有 DFlash2、随机采样、异步流水、整模 Graph、GPU 页映射或跨请求 GPU batch。

## 复现验证

```bash
python3 -m unittest discover -s tests -p 'test_kaifra*.py' -v
```

真实模型检查通过 `scripts/k1_runtime_validate.py` 的 `run(session, arguments)` 在已有常驻 worker 中执行，传入封存源码目录 `source_root`。目录必须带逐文件 SHA256 的 manifest.json。先传 `preflight=true` 检查真实扩展、重复打开/关闭和借用进程组，再执行 GPU 检查。具体运行范围和失败保留规则见 [事前方案](K1_RUNTIME_PLAN.md)。
