# 引擎 v0.1 集成与验收方案

2026-11-02。集成起点：main `99592f6` 与 K4 已提交版本 `74f2199`；另一 K4 工作区的未提交四并发和 Graph 实验保持独立。本方案在首次整合 GPU 作业前冻结。

## 交付范围

- 单台 Spark、当前 NVFP4 Qwen checkpoint、BF16 target KV、greedy 文本生成。
- 一个公开命令入口，单次加载；也支持复用已加载 session。
- 双请求原生调度与共享 target 验证，E17 当前草稿后预算；默认顺序草稿。
- 已有 owned prefill，chunk/token budget128，ready-decode 交错机会。
- OpenAI chat 子集：模型列表、非流式和流式回复、usage；另保留原 K4 SSE API。
- 有界等待、取消、断连、慢消费者隔离、health/readiness、基础服务指标、正常停止。
- 完整模型、算子和 DeepInfer 仍是明确的外部依赖；当前物理容量2048、公开 prompt+output 准入最大1536。

## 冻结验收

1. CPU：现有回归全通过；新 CLI/API 用真实 HTTP 和真实 owner loop 检查，设备边界仅用确定性的已有 fixture。
2. 仅复用常驻 worker，确认队列空闲再提交单个串行整合作业，保存完整源码 manifest。
3. GPU：分别以 target-only 与 E17 adaptive 打开同一个服务工厂，各测双请求，提示固定为英文去重函数与中文缓存解释，输出 cap16，并确认真实 joint forward。两模式比较完整输出 token IDs/文本，独立参考验证完整 logits、已提交 features 和有效 state。另用英文整数计算检查自然 EOS。
4. 检查非流式/流式重建与 usage、模型列表、health、指标、无效请求；额外测单 token cap、取消、断连后请求继续执行与资源回收。
5. 关闭后没有活跃 Target lease/native request，owner/listener 线程退出；原 session 对象身份、Gloo/NEXTN 环境和加载计数不变。保存 allocated 前后值，任何差额保留解释，不能清空 allocator 或提高限额。
6. 单命令冷启动属于独立验收：环境 doctor 通过、session 工厂验收通过后，才考虑一次必要的常驻研究进程到统一服务的切换。未执行前不得把 session 复用验证称作冷启动通过。

本轮验证集成与服务行为，不测性能排名，不训练预测器、不改草稿/target 精度、不执行固定草稿对照。先冻结输入和门槛；失败保留、修复后使用新版本号重跑。
