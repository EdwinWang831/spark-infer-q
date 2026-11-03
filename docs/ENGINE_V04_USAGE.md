# Kaifra v0.4 使用说明

[English](ENGINE_V04_USAGE.en.md) · [验收报告](ENGINE_V04_REPORT.md) · [验证方案](ENGINE_V04_PLAN.md)

当前 NVFP4 target / DFlash2 的单台 DGX Spark 配置。每请求输入加输出
最多 131072 tokens，最多四个运行请求。模型、DeepInfer、TensorFold 和
CUDA 扩展沿用已准备好的本地环境。

```bash
python -m pip install .
kaifra doctor --root /path/to/prepared/runtime --checkpoint /path/to/local/target
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target
```

四请求 128K 默认使用 512 行 prefill，完整 target K/V 保持 BF16。
raw QSA 索引采用已有 2048 行环，全部池化历史与绝对位置保留。
草稿原参数与已有融合矩阵共享精确视图，以减少重复存储。DFlash 的
W8 MLP、Q4 输出头、接受率预测器与动态验证预算沿用已验证配置。

容量包含输出。生成 64 tokens 时，每条输入最多 131008 tokens。
公开 API 沿用聊天流式、非流式、取消、健康检查和指标接口。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"spark-infer-q","messages":[{"role":"user","content":"解释长请求的缓存隔离。"}],"max_tokens":64,"stream":true,"stream_options":{"include_usage":true}}'
```

每个运行请求拥有私有缓存和 Graph，共用串行 GPU 工作区。
`/health` 的 `gpu_batch` 为 false。四个额外流名额用于等待，总在途上限
为八。默认准入等待 30 秒，长请求排队可能超时并返回 429。
请求体上限为 2 MiB，token 数另行检查。

启动先测共享区和单槽增长，再预算全部请求、镜像特征环、草稿投影及
暂存余量。88 GiB 分配上限与 8 GiB 主机余量共同决定是否准入。
不同模型配置或显式分块可能触发启动拒绝。

64K 的较大 prefill 配置可显式选择。

```bash
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --capacity 65536 --prefill-experts /path/to/spark_q_e44_pipeline_v1.so

# 短双请求 GPU batch
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --capacity 1536 --max-requests 2
```

64K 配合已校验的 E44 扩展时，自动使用 1024 行 prefill。四请求 128K
自动选择 512 行，即使传入该扩展，少于 1024 行的 MoE 调用仍走原算子。
`--prefill-chunk-rows 128|256|512|1024` 可覆盖默认值，仍需通过启动预算。
不同容量与分块的耗时分别测量。
