# Kaifra v0.3 使用说明

[English](ENGINE_V03_USAGE.en.md) · [验收报告](ENGINE_V03_REPORT.md) · [方案](ENGINE_V03_PLAN.md)

面向已准备好的单台 DGX Spark 环境和当前 NVFP4 target / DFlash2。
默认四个运行槽、每请求 **65536 tokens，输入与输出合计**；target KV 保持 BF16。
当前执行仍依赖 DeepInfer 和 TensorFold，安装此 Python 包不下载权重或自动准备 CUDA 扩展。

```bash
python -m pip install .
kaifra doctor --root /path/to/prepared/runtime --checkpoint /path/to/local/target

# 兼容配置：512行prefill，原专家算子。
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target

# 推荐用于长prefill的已验收配置：已有E44扩展时，自动选择1024行分块。
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --prefill-experts /path/to/spark_q_e44_pipeline_v1.so
```

本轮 Spark 常驻服务启用第二种配置。扩展是已有的
[E44 专家 tile](E40_PREFILL_EXECUTOR_USAGE.md#optional-mia-shared-input-expert-tile-e44)，
不会在请求中编译或下载。`serve` 在加载模型前核对其 SHA256；
不传扩展时继续使用原算子。少于1024行的调用仍走原MoE。
`--prefill-chunk-rows 128|256|512|1024` 可显式覆盖分块。

接口沿用 [v0.2](ENGINE_V02_USAGE.md)：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"spark-infer-q","messages":[{"role":"user","content":"请解释请求缓存隔离。"}],"max_tokens":128,"stream":true,"stream_options":{"include_usage":true}}'
```

`--capacity` 设置输入加输出的总上限，最大65536；`--max-requests` 为1–4。
四个额外流名额使默认总在途上限为8，起步阶段的待准入命令数可暂时大于4。
准入等待默认30秒，长请求排队可能触发429；此超时未被放宽。请求体上限2 MiB，
token准入单独检查。`/cancel`、`/health`、`/ready` 和 `/metrics` 保持原接口。

启动按实际测得的每槽状态/Graph增量，加上草稿投影、完整镜像特征环与暂存余量准入，
保留88 GiB分配上限和8 GiB主机余量。`doctor` 是静态文件检查，不代表 GPU 已验收。

原配置仍可显式选择：

```bash
# v0.2 的32K长路径
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --capacity 32768 --prefill-chunk-rows 512

# 原双请求短上下文GPU batch
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --capacity 1536 --max-requests 2
```

长路径的四份状态和Graph共用串行GPU工作区，`gpu_batch=false`。
1024行配置改善完整cohort和首token时间，但有prefill等待时单次输出间隔会变长；
需要更短间隔时可选512行。动态成本仍使用冻结E29曲线，没有重训预测器。
本轮没有证明长请求GPU合批、1M、八请求或长期生产压力。
