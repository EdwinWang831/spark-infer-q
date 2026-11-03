# Kaifra v0.1：统一引擎验收

2026-11-02。**统一服务的 session 集成与单命令冷启动均通过；服务保持常驻。** 该结论限定双请求、文本 greedy 和1536-token公开准入范围，不包含256K HTTP服务或四路生产能力。

[冻结方案](ENGINE_V01_PLAN.md) · [中文用法](ENGINE_V01_USAGE.md) · [English usage](ENGINE_V01_USAGE.en.md) · [可校验证据](../results/engine/2026-11-02-v01.json)

## 交付内容

本分支从 main `99592f6` 整合已提交 K4 `74f2199`，保留主线长上下文研究接口、两种既有调度策略和输出缓存。另一工作区未提交的四并发/Graph实验没有移入。

- `python -m spark_infer_q` / 可安装的 `kaifra` 命令，提供doctor和serve。
- 一次模型加载、唯一GPU owner、两个运行请求、默认四个额外等待/流位置。
- 原生DeepInfer调度、E17当前草稿后动态预算、W8草稿MLP和Q4输出头，顺序草稿。
- owned prefill chunk128 / prompt budget128，ready-decode交错机会。
- 模型列表、OpenAI chat文本子集、非流式和流式回复、usage、取消、健康/readiness、Prometheus格式指标、正常停止。
- 旧K4研究SSE接口保持兼容；新增服务显式选择聊天协议。

TensorFold加载/算子、DeepInfer Scheduler/MTPVarlen与当前checkpoint仍是外部依赖；没有宣称已独立替换全部执行底座。

## CPU和打包

本机Python3.14.8运行484项测试：**443通过，41因可选环境依赖跳过**。新检查覆盖CLI标准库入口、参数边界、真实HTTP上的聊天schema、流式重建、Unicode尾部、usage、取消指标和429准入拒绝。设备边界使用已有确定性fixture，CPU结果不替代GPU证据。

wheel构建成功，在独立虚拟环境安装后，离开源码目录运行 `kaifra --version` 和 `kaifra serve --help` 通过。GitHub CPU workflow使用Python3.12、完整git历史和NumPy，不需要模型权重。

## 常驻模型上的整合验收

先确认原worker空闲，串行执行两个封存版本，均保持原Target/DFlash/projection对象、进程PID46235和load_count1。每个版本分别启动target-only与adaptive服务，各处理9个请求。

| 稳定版本 v2 检查 | 结果 |
| --- | --- |
| 完整decode logits / features | 166行 / 166行，与独立eager参考逐位一致 |
| 有效状态 | 88次检查一致 |
| Prefill交接 | 18次logits/features/state检查通过 |
| 两种模式的完整输出 | 代码、中文请求的全部token IDs与文本一致 |
| 流式、非流式、EOS、单token cap | 通过 |
| 显式取消、断连及后续恢复 | 两种模式均通过 |
| 实际共享target计算 | 两种模式均观察到joint forward |
| 关闭 | 无活跃lease/native request，owner和listener线程均退出 |

v1使用新建CUDA stream，关闭后allocated比进入前多32MiB；原始记录保留，没有清空allocator或提高限额，也没有单独确认该差额的底层来源。v2改为复用调用方已有stream，开始/结束allocated均为 **77,426,962,432 bytes**。这说明本轮稳定版本没有新增分配漂移，不外推为无限次服务重建的证明。

## 单命令冷启动

环境doctor通过、session验收完成后，才切换原研究worker。原worker已退出为zombie状态，最初控制脚本因其 `/proc` 目录仍存在而等待；确认State=Z、地址空间已释放后继续，没有强杀或再次加载旧worker。

新进程从公开入口完整加载48层target、准备运行时并输出一次READY；启动日志只观察到一轮48层加载。使用原prepared linker环境和既有本地权重，保留load_target的88GiB分配上限，没有下载或更换checkpoint。

| 冷启动后的真实HTTP检查 | 结果 |
| --- | --- |
| 代码与中文非流式回复 | 与session验收的完整16-token参考及usage一致 |
| 代码与中文流式回复 | 重建文本与参考一致 |
| 六请求队列 | 2运行+4等待，重复输入的完整32-token输出一致 |
| 队列满时额外请求 | HTTP429，已有六请求继续完成 |
| 完成统计 | 10请求、256个输出token，0失败，1次准入拒绝 |
| 验收结束 | active_requests/open_streams均0，health=ready，观察到36次joint forward |

新服务在Spark运行环境的回环地址 `127.0.0.1:18085` 保持运行，PID55633。部署状态随进程变化，以health为准。旧文件队列式研究worker已停止，HTTP服务不消费其jobs目录。

## 结论范围

v0.1完成的是“加载、调度、执行、发布、取消、观测、退出”的本地服务闭环。本轮没有吞吐排名、长期压力、随机采样、工具调用、多模态、跨模型泛化或公网部署结果。完整模型代码和CPU/GPU验收不能替代这些独立能力门槛。

已有256K执行和双长请求适配器继续保留；下一阶段可接入统一服务边界，但当前v0.1并未把其上下文上限从1536提升到256K。

## 证据导出

```bash
python scripts/export_engine_v01.py --raw results/raw/engine-v01 \
  --out results/engine/2026-11-02-v01.json
```

导出器校验两轮session状态、输出、资源起止、冷启动记录和当前生产代码SHA；CLI冷启动版本与v2服务工厂版本之间的生产源码差异仅为启动入口。公开记录省略部署绝对路径；完整启动日志和控制记录留在本地/原机器。

English: the two-request greedy service passed full-model session integration and a separate cold start. CPU checks, GPU/state oracles, queue saturation and startup are reported separately. Long-context serving, four-way concurrency and production stress remain outside this v0.1 result; no new performance claim is made.
