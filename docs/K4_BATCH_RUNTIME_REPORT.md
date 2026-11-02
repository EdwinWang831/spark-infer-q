# K4：双请求 GPU 联合验证与异步服务

**2026-10-24更新：固定sleep轮询导致CPU草稿提交变慢，已改为当前事件等待并通过真实GPU复测。** 同一短组SSE32.57→45.00 token/s；cap128混合30.78→38.04。见[原因与修复报告](K4_WAIT_FIX_REPORT.md)。下文保留最初功能及split/joint证据；其中+24.1%仅比较新backend内部，不代表超过原优化引擎。修复前[同负载回退复核](K4_MATCHED_SCALING_DIAGNOSIS.md)同样保留，广泛性能晋升仍未完成。

路线图 K4 的双请求链路已实现并通过受控功能检查：两个请求共享一次完整 48 层 Target forward，E17 当前草稿后动态预算、W8 草稿与 E33 精确筛选保持不变，异步 owner 驱动 DeepInfer 原生调度器，服务支持两个独立 SSE 流。入口与限制见[使用说明](K4_BATCH_RUNTIME_USAGE.md)，[公开证据](../results/k4/2026-10-23-batch-runtime.json)由导出器重新校验。

之前名为 K4 的[单流开销优化](K4_RUNTIME_OVERHEAD.md)只是运行时开销实验；本报告对应实施路线中的真正双请求 batch 阶段。

## 执行与状态

- `JointTarget` 使用 TensorFold stage、联合 compute 和按窗口偏移的 commit；`kaifra_batch_target.py` 是同一实现的兼容别名。两 slot 的 GDN/QSA/PLE 状态独立，窗口按所拥有的 state 映射，不依赖 native 行顺序。
- 每请求自然选择 q=1～8，各自补齐到 1/2/4/8 bucket；80 张 Target 图覆盖 slot 布局和独立 GDN parity。commit 使用物理窗口宽度与实际接受前缀，padding 不进入有效 state。预填充仍按请求执行八行块。
- `RequestDraftProvider` 只复制每个 generation 的可变投影缓存和使用记录，冻结权重/W8 包/预测器/预算共享。每请求保留 E17 NumPy + E29 W8 的原成本目标，未实现 K5 的全局费用优化。
- `AsyncWorker` 是唯一原生 scheduler owner。Target/head/commit 用 CUDA event 分阶段 launch/poll；设备 fence 后才发布 native output、回收 slot。取消登记与发布边界串行化；准入在在途计划结束后处理。仍存在必要的主机草稿、预算和候选索引同步。
- 每请求队列有界，慢消费者明确失败并取消；断连由 SSE 写失败检测。关闭超时保留在途设备状态；native group 在 owner 活着时拒绝释放。worker 故障隔离后完成 fence，另建 owner 可继续使用原模型。

## 正确性证据

本地 226 项测试，214 通过、12 项因本机依赖缺失跳过。新增事务预检、GPU 完成前禁止回收的确定性生命周期、真实 HTTP/SSE 并发/取消/断连、统计计时口径测试。

resident 上使用实际 DeepInfer 扩展；没有并行启动其他 GPU 任务或重新加载模型：

- CPU native：NEXTN=0 和 7；动态 q 覆盖 1～8；全接受、首/中/末拒绝、EOS/cap、等待/已准备/在途取消、存活请求和同 ID 新 generation。
- 最终 Graph probe：四组同宽/异宽，包括 q=(3,6)/(8,8)/(2,5)，34 个实际行、40 个物理行；完整 BF16 logits、五层特征与接受前缀后的有效 state 一致。
- 动态 GPU split/joint 对照：两组英中请求，一组包括长上下文和自然三 token EOS；每请求输出与 context/rates/q/emitted/accepted/observed 轨迹一致。检查 **1048 行 logits、964 行接受特征、164 次有效状态**，均有独立 eager 缓冲与即时 reference commit，避免共享临时缓冲污染对照。
- 真实 GPU NEXTN=0/7 的 14 项补充生命周期检查覆盖双请求 EOS/长度混合、单 token pending position、三个取消阶段、两个真实 GPU 窗口中取消一个而另一个继续、同 ID 复用、关闭超时、native owner guard、在实际在途 GPU 作业上注入 worker 故障、故障后新 owner 恢复。
- 实际并发 SSE 输出正确；在途取消后旧 generation 不能复用 slot；真实 HTTP 断连与慢消费者回收通过。HTTP 的显式 `/cancel` 路由另由本地真实 HTTP/确定性 fence 测试覆盖。

## 匹配的双请求服务对照

split 和 joint 使用同一 native scheduler、E17/W8/E33、bucket、worker、HTTP handler 和提示。split 仍准入两个请求，但逐请求发起 Target ticket；joint 合并原生计划中的 decode 行。两个英中固定 pair × 三次交替重复 × 两种执行方式，共 **12 次双请求计时**。这是双请求受控范围，不是并发容量或生产稳定性结论。

计时包含准入、文本处理、SSE 与事件输出。完整请求从两个客户端提交到两个流结束；decode cohort 从两者都收到首个输出后到两者结束，计数只包含 cohort 后观察到的 token。每种方式的 cohort token 数相同，按总 token / 总 wall time 计算。长短请求有尾部单 slot 执行；不把整个请求时间当作双 slot 满载时间。

每个计时逐请求核对预算/接受轨迹和 tokens；实际/物理验证行与对照一致；正式计时没有新增 Target capture 或头部候选子集 JIT 形状。预热与诊断不进入计时。

| 指标 | split | joint |
| --- | ---: | ---: |
| 双请求合计 decode token/s | 31.0947 | 38.5850 |
| 完整请求合计 token/s | 13.4133 | 14.8738 |
| TTFT p50 / p95，ms | 491.07 / 3398.79 | 482.80 / 3311.91 |
| SSE 摊销 TPOT p50 / p95，ms | 27.83 / 177.03 | 22.69 / 116.22 |

每组正式计时共有 249 个 cohort decode token、261 个完整输出 token；三次整体 decode 收益分别为 +22.75%、+35.46%、+15.21%。合计 decode **+24.09%**，完整请求吞吐 **+10.89%**。TPOT 是相邻 SSE 输出块的时间间隔除以新增 token 数，并按 token 加权；speculative token 一次输出多个，不能解读为每个 token 均匀到达。长提示预填充使 p95 TTFT 达到约 3.3 秒。

初次探索对照为 32.36 → 36.72 token/s（+13.48%）；它未逐次记录 JIT/预算检查，独立保留，不并入正式分母。两轮之间存在明显波动。本轮只有两个固定 pair，未给泛化置信区间或晋升默认配置，也不能与先前单流约 48 token/s 的不同任务/计时口径直接比较。

## 回收、失败记录与边界

同一 resident PID 319，load_count=1，最终 READY；Gloo 与 NEXTN 环境恢复，没有调用 `empty_cache` 或清空共享库缓存。K4 Python 对象、请求 lease、原生 blocks 和在途 ticket 均已回收，第二次 GC 后显存不再变化。

最终 allocated 为 **77,695,397,888 bytes**，比新 stream 之前的 **77,594,734,592** 多 **96 MiB**。分配器中存在无 Tensor 持有者的非默认 stream 32 MiB 块，与三次新增 stream 的库工作区缓存一致；归因为分配器/持有者证据推断。没有强行清空进程级 BLAS 工作区来制造旧基线相等，不能声称 allocated 完全回到旧值。资源报告保留这一差额。

保留的失败：首版 probe 在两个 reference 之间复用了 eager 临时缓冲，误报 joint state 不一致；修正为各 reference 即时 commit 后，完整 Graph 版本通过。两次 audit 错把新 stream 的库缓存也要求回到旧 allocated，保留失败记录；最终 audit 分开检查请求对象、状态回收和库缓存。一次 GC 审计遇到非字符串 `__module__` 的检查错误，修正类型判断，未改执行器。

容量限定两请求、greedy、loopback。最长正式提示 324 token、最大输出 cap 32；Target 物理容量 2048、CLI 逻辑上限1536不等于所有容量点已测。尚无 4K+、随机采样、公网部署、长期压力、全局批次成本模型或更大并发性能结论。草稿仍由 owner 依次计算；本阶段的联合执行发生在 Target 验证。

## 复现

```bash
python3 -m unittest discover -s tests -q
python3 scripts/k4batch_stage.py unique-cpu unique-cpu-job.json k4batch_cpu_validate.py
python3 scripts/k4batch_stage.py unique-probe unique-probe-job.json k4batch_probe_validate.py --graphs
python3 scripts/k4batch_stage.py unique-gpu unique-gpu-job.json k4batch_gpu_validate.py --graphs
python3 scripts/k4batch_stage.py unique-lifecycle unique-lifecycle-job.json k4batch_lifecycle_validate.py --graphs
```

job basename 要用小写字母/数字/连字符；每次均需换成唯一标识。stage 封存源码并校验 SHA256，通过共享 resident 队列顺序执行。导出器验证固定本轮证据名、源码身份、预算/输出一致、计时配对和资源条件：

```bash
python3 scripts/k4batch_export.py results/raw/k4-batch results/k4/2026-10-23-batch-runtime.json
```

完整 raw 结果、失败、封存源码和 manifests 保留在被 Git 忽略的目录；`evidence.tar` SHA256 为 `052cc65567a5d35bafad1e72574d875c643275c7c404cd9290a88c6a2f52f79e`。公开 JSON 不包含提示、权重或私有绝对路径。当前服务为显式入口，没有替换既有单流默认配置。
