# Kaifra v0.2 使用说明

[English](ENGINE_V02_USAGE.en.md) · [验收方案](ENGINE_V02_PLAN.md) · [v0.1 历史报告](ENGINE_V01_REPORT.md)

此入口继续面向已经准备好的单台 DGX Spark 环境，复用已有 NVFP4 target、
DFlash2、DeepInfer 和 TensorFold。不下载模型，不是任意模型的一键安装器。

```bash
python -m pip install .
kaifra doctor --root /path/to/prepared/runtime --checkpoint /path/to/local/target
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target
```

默认每请求容量 **32768 tokens，输入和输出合计**；四个运行请求，
额外四个等待/未关闭流名额。启动时依实际测量的私有状态/Graph 增量、
草稿缓存和暂存余量检查内存，保留 88 GiB 进程上限与 8 GiB 主机余量。
`doctor` 只检查文件，成功不代表 GPU 已验收。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"spark-infer-q","messages":[{"role":"user","content":"请解释请求缓存隔离。"}],"max_tokens":128,"stream":true,"stream_options":{"include_usage":true}}'
```

接口、取消、指标和关闭操作见 [v0.1 用法](ENGINE_V01_USAGE.md)。
请求体最大 2 MiB，token 准入另计。`--capacity` 和 `--max-requests`
分别设置单请求总长度与运行槽，`--queue-size` 设置额外名额；
等待超过 `--admission-timeout`（默认 30 秒）返回 429。长请求排队可能超过此时间。
总流数上限为运行槽与额外名额之和（默认 8）。`pending_submissions` 还包括
等待下一计划边界准入空闲运行槽的命令，起步时可暂时超过 4；四槽占满后额外等待为 4。

长路径保留 BF16 target KV、原生 Scheduler/MTPVarlen、动态 DFlash、
E54 的 2040 行滚动特征及请求私有投影缓存。四份私有状态和 Graph 共用
串行 GPU 工作区：HTTP 请求可并发在途，`gpu_batch=false`。
`asynchronous=false` 描述同步 device 执行器，`http_async_owner=true` 描述服务线程。
动态成本仍沿用冻结的 E29 曲线，未按多路长请求重新校准。

原有短路径仍可显式启用：

```bash
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --capacity 1536 --max-requests 2
```

该路径保留双请求 target GPU batch。v0.2 没有导入另一工作区的未提交四路
CompactTarget 实现。`--mode none` 可作 target-only 正确性对照，
`--execution eager` 可作诊断；默认采用 adaptive 与 Graph。
本阶段没有把 256K、八并发、压缩 KV 或长期生产压力宣布为统一服务能力。
