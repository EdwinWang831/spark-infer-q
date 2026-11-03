# 四请求 128K prefill 合批配置

[English](ENGINE_V07_USAGE.en.md) · [报告](ENGINE_V07_REPORT.md) · [方案](ENGINE_V07_PLAN.md)

在已准备好的单台 DGX Spark 环境中，显式选择每请求 1024 行、至多
4096 行的 prompt 前向。该配置要求四个 128K 运行槽、adaptive DFlash、
Graph decode、E44 专家扩展、linear 特征窗口和已校验的原生控制扩展。

```bash
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --capacity 131072 --max-requests 4 --mode adaptive --execution graph \
  --prefill-experts /path/to/spark_q_e44_pipeline_v1.so \
  --prefill-chunk-rows 1024 --feature-history-layout linear \
  --prefill-batch-control /path/to/kaifra_prompt_control_v1.so
```

原生控制扩展的构建参数在
[`k4batch_engine_batch_memory.py`](../scripts/k4batch_engine_batch_memory.py) 中，
源文件为 [`k4batch_prompt_control.cpp`](../scripts/k4batch_prompt_control.cpp)。
加载器校验本轮封存的 DeepInfer revision、扩展、三个头文件和控制
二进制身份。该二进制对应报告中的准备环境。

每请求 131072 tokens 包含输入和输出。四请求各输出 64 tokens 时，
各自输入上限为 131008。运行期间维持 88 GiB 分配器上限、8 GiB 主机
余量及至少 1 GiB 暂存预算，准入使用当前进程的实际内存测量。

`/health` 的 `prefill_gpu_batch`、`prefill_batch_requests` 和
`prefill_workspace_rows` 分别为 `true`、`4`、`4096`。
`gpu_batch=false` 描述 decode 的逐请求串行执行，`prefill_cuda_graph=false`
描述 prompt 的 eager 算子执行。单轮可以处理一至四个有效请求，取消的
请求在调用共享前向前移除。状态更新后完成设备同步，再发布原生输出。

完整四长请求的平均总耗时减少 10.29%，较早请求的首 token 延迟增加，
具体数值见[延迟表](ENGINE_V07_REPORT.md#每请求延迟)。聊天、排队、取消、
健康和指标接口见 [128K 用法](ENGINE_V04_USAGE.md)。
普通启动使用[镜像／512 行配置](ENGINE_V05_USAGE.md)，单请求 1024 行
配置通过同一用法中的参数选择。
