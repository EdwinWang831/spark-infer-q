# Kaifra v0.4.1 prefill 配置

[English](ENGINE_V05_USAGE.en.md) · [报告](ENGINE_V05_REPORT.md) · [验证方案](ENGINE_V05_PLAN.md)

当前单台 DGX Spark、已准备好的 NVFP4 target 与 DFlash2 环境。
普通入口采用四请求 128K、512 行 prefill 和镜像特征窗口。

```bash
kaifra doctor --root /path/to/prepared/runtime --checkpoint /path/to/local/target
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target
```

具备已校验的 E44 专家扩展时，可显式选择 1024 行 prefill 与单缓冲特征。

```bash
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --capacity 131072 --max-requests 4 \
  --prefill-experts /path/to/spark_q_e44_pipeline_v1.so \
  --prefill-chunk-rows 1024 --feature-history-layout linear
```

`--feature-history-layout mirrored|linear` 选择草稿已提交特征的存储布局。
两种布局都保留最近 2040 行，并提供连续视图。单缓冲滚动时复制有效后缀，
每请求持久存储从 104448000 降至 52224000 bytes。追加时返回的视图借用
窗口存储，有效期至下次追加。原始输入参数和完整 target K/V 继续使用原格式。

线性布局用于带有界草稿历史的长 adaptive 路径。target-only 和短双请求
GPU batch 采用原配置。target KV 为 BF16，动态预算使用冻结 E17/E29。
`/health` 暴露 `feature_history_layout`、`prefill_chunk_rows`、专家后端与 SHA256。

启动仍要求 88 GiB 分配器上限、8 GiB 主机余量及至少 1 GiB 暂存空间。
是否准入按当前进程实测。每请求 131072 tokens 包含输入和输出，生成
64 tokens 时，输入上限为 131008。四个运行槽加四个额外流名额，默认
准入等待为 30 秒。聊天、取消、健康与指标接口沿用
[128K 使用说明](ENGINE_V04_USAGE.md)。

1024 行分块与专家 tile、特征布局组成一个配置。完整请求耗时、首 token、
输出间隔及解码阶段分别计时。长请求共用串行 GPU 工作区。
