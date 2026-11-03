# K4 固定轮询导致草稿提交变慢：原因、修复与复测

**已经找到并修复一个主要原因。** Target 阶段之间固定休眠轮询，会使随后 eager DFlash 的 CPU 提交明显变慢，GPU 操作之间出现更多空档。只改等待方式、保留相同 GPU 计算与所有完成 fence 后，真实双请求 SSE 短组从 **32.57 到 45.00 token/s（+38.17%）**，cap128 混合组从 **30.78 到 38.04（+23.57%）**。E17 预测器、当前草稿后动态预算、输出和逐请求接受轨迹不变。

这是三组固定负载、每个模式两次计时的诊断结果，尚不构成默认服务晋升或普遍双流扩展结论。原[同负载回退报告](K4_MATCHED_SCALING_DIAGNOSIS.md)保留历史数据；本轮[公开证据](../results/k4/2026-10-24-wait-fix.json)由 `k4batch_wait_fix_export.py` 从封存结果重算。

## 为什么确定是等待方式

作业261只改变新 backend 不完整 poll 之后的等待：固定0.5ms sleep、立即重试（仅诊断）、等待当前事件。相同中文提示、cap64，四个模式各三次均衡交替；输出和完整 E17 决策轨迹逐次相等，正式计时无新增 head JIT。

| 模式 | decode token/s | 每请求草稿合计 wall，ms | 草稿调用中位数，ms |
| --- | ---: | ---: | ---: |
| 原优化 single | 34.51 | 333.77 | 11.15 |
| 新 backend，固定 sleep | 25.57 | 827.54 | 35.22 |
| 新 backend，立即 poll | 32.88 | 341.50 | 11.37 |
| 新 backend，当前事件等待 | 32.98 | 340.20 | 11.67 |

等待的直接0.5ms不是全部损失；它还改变随后 CPU 执行与提交的行为。独立作业262用 Kineto 对照原 single 和 sleep backend：各18次 DFlash，每次1024个 kernel、78种名称，逐次 kernel 多重集合完全一致，没有跨 proposal 边界的 kernel。两边都包含522个 GPU copy/memset 操作，合计仅约0.52/0.56ms。

| 18次草稿合计 | 原优化 single | sleep backend |
| --- | ---: | ---: |
| 调用 wall，ms | 270.91 | 493.80 |
| 调用线程 CPU，ms | 270.73 | 492.42 |
| kernel 时长之和，ms | 141.57 | 139.55 |
| GPU kernel 间隔合计，ms | 122.23 | 346.48 |
| GPU kernel/copy/memset 操作间空档合计，ms | 122.30 | 346.61 |

额外 wall 约222.89ms，伴随约224ms的操作间空档增加，GPU 算术耗时没有增加。这里是独立带剖析的归因运行，不把它混入正式吞吐分母，也不把有嵌套的 API 时间相加。

作业263另做只读 CPU 观察：每个模式两次，共36次草稿。原 single 调用期间入口/出口 CPU 不同次数为0，sleep 为22，事件等待为2；草稿中位调用11.39/25.71/11.70ms。sleep 观察到更多低 capacity、最高约2.808GHz的异构核；原 single 主要在最高3.9GHz的核，事件等待主要恢复到后者。这支持 CPU 调度/迁移参与提交变慢，但不是连续硬件频率跟踪，也没有单独隔离迁移与频率各自贡献。未改 affinity、governor、时钟或 CUDA 调度参数。

此前缓存分配/复制的独立控制见旧报告：各模式都约10～11ms，不能解释交替 Target 后数倍的草稿调用时间。

## 修复与安全边界

`AsyncBatchMixin.wait_for_device(plan)` 选择当前 verify/head ticket 或 commit event，只等待这个计划的当前事件。`execute()` 与异步 owner 在 poll 未完成时使用它；无事件的通用 bridge 保留原等待回退。生产没有采用 Python 忙轮询，也没有新增全设备同步。

事件等待本身不发布结果、不设 `plan.executed`、不退休 ticket、不释放 slot。随后仍由原 poll 验证和推进阶段，完成后才发布到 native。owner 在命令锁外等待，其他线程可登记取消或关闭；下轮取消仍在发布前应用。关闭超时继续保留在途状态，等待事件失败会隔离 bridge，回收前仍需 fence。驱动如何等待不等于已经测得 CPU 功耗下降，本轮没有功耗结论。

保留 E17 NumPy float64、W8 MLP/Q4草稿输出头、BF16 attention/selector、E33精确 Target 头、每请求当前成本目标、自然 EOS 和 q=1～8。没有固定草稿模式，没有为了速度去掉 generation、取消或 lease 保护。

## 同时复测优化 single 与新 backend

作业264使用相同 resident、共享 JointTarget/provider，封存 sleep backend 与修复 backend 并存；每组每模式两次，五种模式共30次正式计时。single 是两请求独立顺序处理的容量基线。所有模式同 tokens/预算轨迹、无计时内 JIT。单位为首 token 后到最终退休的合计 decode token/s，排除 prefill。

| 负载 | 原优化 single | sleep single | 修复 single | sleep joint | 修复 joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原短英中，32/32输出 | 40.30 | 32.75 | 38.11 | 34.67 | 45.75 |
| 原长短，20/3自然输出 | 56.56 | 47.49 | 53.29 | 49.72 | 55.94 |
| 短提示cap128，57/128实际输出 | 37.78 | 27.51 | 36.82 | 30.12 | 39.00 |

修复 joint 相对同期原优化 single：短组+13.51%，长短组-1.09%，cap128组+3.23%。修复 single仍低2.53%～5.79%，没有消除全部接入开销。长短组输出很少、重复仅两次，不对约1%差异作稳定优劣判断。

## 真实异步双请求 SSE

作业266对照封存的原 worker/backend 与修复 worker/backend，共三组×两模式×两重复=12次双请求正式计时；共用 Target、provider、HTTP handler，并在 owner 显式选择同一 resident 默认 CUDA stream。两次模式顺序反转。每条请求比对完整 tokens 与逐轮 E17/接受轨迹，正式运行无新 head JIT；预热记录独立保留。80个 Target Graph，无 fallback。

decode cohort 从两个客户端均观察到首个输出开始，到两个流终止；只计 cohort 后观察到的 token。完整请求指标另从客户端提交算起，包含预填充和服务开销。两模式各组 cohort token 数相等，全部由总 token / 总 wall 重算。

| 负载 | 原 SSE decode | 修复 SSE decode | 变化 | 原完整请求 | 修复完整请求 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原短英中 | 32.57 | 45.00 | +38.17% | 26.95 | 34.69 |
| 原长短 | 51.22 | 59.46 | +16.09% | 6.26 | 6.38 |
| cap128混合 | 30.78 | 38.04 | +23.57% | 28.81 | 34.95 |

单位均为合计 token/s，不是每条流各自速率。长提示预填充仍主导长短组完整请求时间。上述数值不与历史八任务单流约48、上轮不同 cohort 的 SSE38.58直接计算增益。不同提示与自然 EOS 的长尾仍明显；本轮未新增长期压力、延迟分布或更大并发验收。

## 正确性、资源与剩余工作

本地230项测试，218通过、12因环境跳过；新增事件选择/禁止提前发布、异步等待期间取消和有界关闭、事件故障后的 fence/隔离检查。作业265用修复版重新通过14项真实 GPU NEXTN=0/7 生命周期检查：EOS/cap、pending position、等待/已准备/在途取消与 survivor、同ID新 generation、关闭超时/native owner guard、实际GPU故障隔离及新 owner 恢复。

所有 GPU 作业由同一 resident 串行处理，PID319、load_count=1，最后 READY。作业269检查没有活跃 K4 对象，Gloo/NEXTN 环境恢复，第二次GC后 allocated稳定在77,728,952,320 bytes，比前轮77,695,397,888新增32MiB。分配器显示非 Tensor 持有的32MiB工作区块，符合新增 worker/default-stream 库工作区缓存；这是分配器证据归因，没有调用 `empty_cache` 或清空共享库缓存来追平旧值。

保留失败作业260：初版剖析把 CPU `user_annotation` 和 `gpu_user_annotation` 的 DFLASH 范围都计入，36个范围对应18次调用，解析断言失败；262限定 CPU 范围后通过。封存262脚本中的频率采样描述不准确，实际频率证据只来自263；当前脚本已订正说明，并显式保留 sleep 对照，避免以后修复版重跑时误换对照。

当前仍是两请求显式研究入口。DFlash逐请求串行、Target按1/2/4/8补齐、自然EOS造成singleton尾部，以及固定的每请求预算成本目标都还在。下一步优先量化并改善草稿串行与singleton效率，再决定是否进入K5全局成本调度；本轮没有把K5或广泛性能验收标为完成。

## 复现与封存

以下命令通过现有串行 resident 队列执行，必须使用新的唯一版本/作业名；不能并行另起GPU进程。旧 backend 对照使用已封存 `wait-v1`，必须保持其 manifest 完整。

```bash
python3 scripts/k4batch_stage.py unique-wait unique-wait.json k4batch_wait_probe.py --graphs
python3 scripts/k4batch_stage.py unique-core unique-core.json k4batch_core_probe.py --graphs
python3 scripts/k4batch_stage.py unique-dispatch unique-dispatch.json k4batch_dispatch_profile.py --graphs
python3 scripts/k4batch_stage.py unique-direct unique-direct.json k4batch_wait_fix_validate.py --graphs
python3 scripts/k4batch_stage.py unique-service unique-service.json k4batch_wait_service_validate.py --graphs
python3 scripts/k4batch_stage.py unique-lifecycle unique-lifecycle.json k4batch_lifecycle_validate.py --graphs
python3 scripts/k4batch_stage.py unique-audit unique-audit.json k4batch_wait_fix_audit.py --graphs
python3 scripts/k4batch_wait_fix_export.py results/raw/k4-wait-fix results/k4/2026-10-24-wait-fix.json
```

导出器核对原始封存源码、当前生产包源码、同一worker、配对重复/计数、预算一致断言、无计时内JIT、真实kernel多重集合、Graph与资源条件。原始作业、失败、源码、manifests及压缩私有trace保留在忽略目录；公开JSON仅包含标量、计时行与指纹。`evidence.tar.gz` SHA256：`5c25ff3574f1c68e3fe4f972d816b70015c7a17d1b377b560cf1c4adc0349619`。
