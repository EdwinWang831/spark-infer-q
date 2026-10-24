# E33 原生运行时开销优化

**同期实测：原生运行时 47.2815 → 47.9935 token/s（+1.5059%）；研究路径 48.4255 token/s，差距由 2.3623% 缩小到 0.8919%。**

此轮沿用 K3/E33 的冻结模型、W8 草稿、E17 NumPy 预测器、当前草稿后动态验证预算与原 BF16 精算规则。固定草稿模式不恢复。

## 改动

- 原生 CPU 元数据由桥接器持有有界 NumPy 存储和共享 Tensor 视图。宽度、草稿、概率、accepted flat buffer 和输出 token 在同步原生调用之间复用，取消行清零。原生调用立即读取或复制数值；不跨轮持有输入存储。bridge 仍只允许一个 pending plan。
- 单 rank、单准入 slot 时，decode 期间不可能还有另一个 waiting prompt。把 native prefill 检查间隔设为 `capacity+1`，每次新准入通过公开 `decode_step` 属性重启检查相位；原生每次 prefill 本来也会重置该相位。取消/复用后下一步立即处理新 prompt。它是 prompt 轮询计数，不改变 MTP `schedule_cnt/process_cnt`。多 rank/多 slot 仍每轮检查。
- 构造 speculative work row 用原生 `get_len/get_last_token_id` 读取唯一 pending anchor，避免在 reserve/finalize 时两次复制整个已发布 token 历史；仍检查 pending 数量、q/k 长度与 committed position。对外输出事件继续核对完整原生前缀。
- 不改变原生 schedule/process 计数、EOS/输出上限、取消、handle generation、state ownership 和发布前检查；没有将第一轮草稿移出 decode timer。

## 对照协议

使用原 E33 的八个工程 root，四轮自然 EOS/output_cap；已分词 token API。三组交替轮换：原研究循环、冻结的迁入前运行时、当前运行时。旧运行时来自上一轮完整 source manifest，独立导入并检查 SHA256。三个 Target 各预备 16 张图，同一个 resident worker 顺序执行，不重新加载共享模型。

decode 从首 token 准备完成后开始，包括第一轮草稿、预测/预算、筛选验证、同步、commit、调度/记账/事件与终止回收。预热、正确性诊断、模型/图准备和 prefill 在 decode timer 外。所有计时核对输出、最终有效状态和完整决策轨迹；预热另逐行核对原 BF16 logits 与 DFlash 特征。性能剖析为另外两条 root 的诊断任务，不混入正式测速。

正式结果由 `scripts/k4_overhead_export.py` 重新检查三组的加权总 token/总耗时、root bootstrap、历史轨迹与运行时代码身份。历史报告中的 47.46/48.38 不用于替代这轮同期对照。

## 正式结果

2026-10-23，`PASS_K4_RUNTIME_OVERHEAD`。[公开计时与校验证据](../results/k4/2026-10-23-runtime-overhead.json)。8 个 root × 4 轮 × 3 组，共 **96 次正式计时**，每组 **8588 decode token**。下列完整请求/首 token 准备时间均为 32 次请求之和，提交前的 engine 初始化不在请求 timer 内。

| 配置 | decode token/s | decode 总秒数 | 完整请求总秒数 | 首 token 准备总秒数 |
| --- | ---: | ---: | ---: | ---: |
| legacy | 48.4255 | 177.345 | 265.088 | 87.744 |
| native_before | 47.2815 | 181.635 | 270.866 | 89.231 |
| native | 47.9935 | 178.941 | 268.227 | 89.286 |

相对同轮旧运行时 **+1.5059%**，按 8-root 成对 bootstrap 的 95% 区间 **0.0621%～2.6454%**。四轮整体变化为 **-0.1464% / +4.7810% / +0.2672% / +1.1944%**；其中一轮略慢，收益较小，存在明显轮间波动。

同期研究路径差距由 **2.3623%** 缩小到 **0.8919%**。新路径相对研究路径的 95% 区间为 **-2.2146%～-0.0414%**，仍存在剩余开销。旧运行时到新运行时合计省下 **2.6947 秒**，约 **0.3138 ms/token**；与研究路径剩余差额约 **0.1858 ms/token**。这是整套改动的净效果，没有按 CPU 函数拆分归因，也不把旧轮次 47.46/48.38 与这轮混算。

| 工作量（每组） | 数量 |
| --- | ---: |
| 验证轮次 | 2596 |
| 验证行数 | 12748 |
| accepted / observed draft | 6012 / 7784 |
| 自然筛选头回退 | 0 |

所有组的完整 context/rates/width/emitted/accepted/observed 轨迹一致，并与冻结 E33 历史记录逐步一致；每次输出 hash 和最终有效 state 均核对。每组 Graph 16 张，计时中没有新增捕获或 subset JIT。逐 root 的计时保留在公开证据，不将 token/s 按任务简单平均。

## 验证与回收

- 本地 211 项：199 通过、12 因本机依赖缺失跳过；增加三组统计的加权计时、成对比较和缺失/重复运行拒绝检查。
- 原生 CPU：15 组 K2 回归、4 组 deferred-draft/预测 ID 契约；同一个 engine 连续复用 5 次，长度 1/3/8/24/2，每次准入后的第一步立即给出首 token，资源全部归还。
- GPU：11 组头功能、13 组 native/HTTP，511 行完整 committed logits、511 行特征、116 次有效 state 检查；覆盖全接受/拒绝、EOS/cap、三阶段取消复用、真实 SSE 和取消。
- 正式对照的预热另核对 **6441 行完整 logits、6441 行特征、4984 行 prefill 特征**；三组预热与计时共 **120 次最终有效状态**检查。
- worker 全程 PID 319、load_count=1；未重新载入或停止模型。最终 READY，Gloo group/env 恢复；GC 后 allocated 回到 77,594,734,592 bytes，未调用 empty_cache。

## 首轮尝试与适用范围

第一轮将 CPU 元数据复用与借用 Graph hidden/省去中间 fence 一起测试。该轮旧运行时 47.9206、新路径 47.4940 token/s，变化 -0.8902%，95% 区间 -2.1508%～+0.5126%，未证明收益。Graph 改动已撤回；最终版保留原 Graph 复制和同步，使用上述 CPU/调度优化。初次两 root 的 cProfile 仅作诊断，包含剖析器开销，不加入正式速度或被用来拆分 1.51% 的原因。原始数据同时保留两次正式尝试和旧路径复测，避免只保存最快结果。

这是同一组八个工程 root 上的小幅改进；默认 profile 仍为 baseline，E33 仍显式启用。单请求 greedy、E17 当前动态预算、自然 EOS/cap 的范围与 K3/E33 相同。未测速 HTTP 网络/逐事件文本解码、冷 JIT、4K+、联合 GPU batch 或随机采样。没有删除状态、取消、输出前缀检查来换取速度。

## 复现

沿用 [E33 运行入口](K3_E33_RUNTIME_USAGE.md)，无需新的模型、kernel 或 CLI 参数。常驻队列使用 `k3e33_cpu_validate.py`、`k3e33_gpu_validate.py`、`k3e33_overhead_online_validate.py` 与 `k3e33_resident_validate.py`；每次封存新 source_root。三组 runner 强制检查旧 source manifest SHA `a8848be2fb59096ae30087a1ca3b728f823aad520f6a96cc2dd69f6f1c63c9ad`，缺失或变化即拒绝，不自动重新加载模型。

```bash
python3 scripts/k4_overhead_export.py results/raw/k4-overhead results/k4/2026-10-23-runtime-overhead.json
python3 -m unittest discover -s tests -q
```

导出器重新验证加权统计、历史轨迹、源身份、所有正确性和回收条件；公开结果保留每次标量计时、工作量、决策 trace SHA 和源码/内核身份，不包含原提示、权重或私有绝对路径。完整逐步记录位于 Git 忽略的 raw 目录；原始 `evidence.tar` SHA256 为 `64ef62f460ecf4b09766eebf587956d06644a9701e2b173cbf5863741f0f4950`。
