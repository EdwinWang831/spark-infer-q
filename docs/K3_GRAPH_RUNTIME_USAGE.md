# K3 Graph 执行候选

在 K2 的动态推测解码中可显式选择 Graph：

```bash
E3_ROOT=/path/to/checkout/scripts bash /path/to/prepared/e3_run.sh \
  k2_serve.py --root /path/to/prepared --checkpoint /path/to/target \
  --mode adaptive --execution graph --capacity 256 --port 18083
```

默认 `--execution eager` 保持现有行为；Graph 不恢复固定草稿模式，也不更换 E17 预测器/成本。现有 `lagged` / `confidence` 动态对照也可选择 Graph。`none --execution graph` 在加载前拒绝，Target-only 仍为 K1 eager。

Graph 的启动阶段预热1～8行×两种GDN奇偶，共16个bucket，再报告READY。prompt保持eager，Graph只替换decode compute；stage/commit/greedy及必要的host完成同步保留。未捕获width或不兼容state显式回退eager，普通请求不动态capture。指标公开capture/replay/fallback计数、预热耗时和单请求范围，不宣称异步或联合GPU batch。

session API可复用已有模型：

```python
from spark_infer_q.kaifra_graph import GraphTarget
from spark_infer_q.kaifra_speculative import session_provider, speculative_engine

execution = GraphTarget(session.target)
try:
    execution.warmup()
    provider, curve, metadata = session_provider(session, 'adaptive')
    with speculative_engine(native, execution, provider, mode='adaptive',
            eos_ids=session.eos, prior=session.calibration['conditional_prior'],
            capacity=256, tokenizer=session.tokenizer, curve=curve) as engine:
        engine.submit('example', [{'role': 'user', 'content': '你好'}], max_tokens=32)
        for event in engine.drain():
            print(event.token_ids, event.finish_reason)
finally:
    execution.close()
```

`native` 必须是 K2 的 NEXTN=7 NativeScheduler 上下文。一个GraphTarget持有一个固定地址workspace，当前只能由一个活动请求独占；完成/取消释放lease，下次复用时reset，图保留。先关闭engine，再关闭GraphTarget。调用者不得在Graph请求执行中并发使用同一个eager target/buffers；resident作业与服务本身均串行执行。capability计数在每次engine.step后更新，反映实际重放/回退。

Graph是显式候选。研究中的E29收益和E33头优化不能当作这套服务的速度；K3接入结果见[报告](K3_GRAPH_RUNTIME_REPORT.md)。没有重新加载权重或训练预测器，尚无长上下文、多请求同时重放、随机采样及全主机异步执行结论。


完整辅助头/W8/NumPy/Graph配置使用独立[profile e33入口](K3_E33_RUNTIME_USAGE.md)，同期匹配结果见[迁入报告](K3_E33_RUNTIME_REPORT.md)。本页baseline Graph不包含这些优化，勿将其44.78直接与完整配置比较。
