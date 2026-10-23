# E31：专家 scale 成对解码，未通过性能门槛

2026-10-23 UTC，`REJECT_E31_EXPERT_PERFORMANCE`。成对解码减少了编译后的转换指令，数值检查全部通过，但真实专家调用没有可靠提速。不晋升，不做完整 decode 长测，不修改默认执行器。[事前方案](E31_EXPERT_SCALE_PLAN.md) · [完整脱敏证据](../results/e31/2026-10-23-expert-scale.json)。

## 改动及创新边界

原专家核将相邻两个 E4M3 scale 分别通过 FP8→FP16→FP32 转换。候选使用 CUDA 已有的 `__nv_cvt_fp8x2_to_halfraw2` 一次转换这两个值，再转为 FP32。保留原 NVFP4 权重布局、BF16 激活、MMA、逐块 FP32 FMA 顺序、全局 scale、epilogue、4 warp、双预取。没有新增专家权重副本，也不更改精度。

这是既有 packed conversion 技术在当前内核上的应用，不能称为新的算法、指令或模型架构。没有做完整文献/专利检索，也不主张首创。它是否值得保留由当前执行收益决定。

静态 SASS 观察如下。计数包含编译展开代码，**不是动态执行次数或 GPU 硬件计数器**。

| 内核 | 原 FP8 转换指令数 | 候选 | 全部静态指令数：原→候选 |
| --- | ---: | ---: | ---: |
| gate/up，SwiGLU | 64 | 32 | 1,896→1,772 |
| down，FP32 输出 | 32 | 16 | 768→710 |
| down，BF16 输出 | 32 | 16 | 776→718 |

因此原编译产物没有自动把这些转换合并。BF16 down 仅做静态检查；本轮真实 decode pair 使用 FP32 down。不能从静态减少量直接推断运行时间减少量。

## 正确性和真实输入

冻结 E28 W8 草稿、E17 NumPy 线性预算、E29 W8 费用和完整 target Graph，复用 E30 的四条校准 root（英文问答、英文代码、英文长输入、中文长输入），每条前两轮，在第 0、23、47 层检查专家 pair，共 24 例。实际验证宽度为 2、4、5、6、8，context 为 31～771；没有覆盖所有宽度、所有层或新的独立任务集。

GPU 穷举全部 65,536 对 FP8 字节：有限值与有符号零按位一致，NaN 位置一致。24 例原核输出首先核对真实前向的 routed 输出，再与候选逐位核对 gate/up BF16 和 down FP32 输出；shared slot 与边界哨兵保持不变。候选只在隔离 pair 回放运行，不替换全局 dispatch。

另外以实例局部的诊断遍历读取真实路由，8 次完整诊断前向均与冻结 Graph 的 logits、五处 DFlash 特征和提交 live state 一致。这验证采样/回放没有改变原轨迹；**不是整模型换成候选后的生成验证**。

## 性能结果

每例为原核和候选分别捕获只含 gate/up、down 的两节点 CUDA Graph，再交替 replay 计时。每种缓存条件 6 次配对：热复用每区间 10 次 pair，128 MiB 缓存扰动每区间 1 次，避免区间内重新变热。共 576 个 CUDA-event 区间、3,168 次计时 pair 调用。capture、缓存扰动、clone、数值检查和预热不计入区间。

下表是 **24 例各组中位数之和**，不是一次完整 target 验证耗时。独立回放和缓存扰动也不等于整模型的实际缓存分布。

| 条件 | 原核合计 | 成对解码合计 | 速度变化 | 按 root bootstrap 95% 区间 |
| --- | ---: | ---: | ---: | ---: |
| 热复用 | 13.225680 ms | 13.195328 ms | +0.230% | −0.604%～+1.040% |
| 128 MiB 缓存扰动 | 13.377968 ms | 13.401424 ms | −0.175% | −0.629%～+0.244% |

两个区间均跨零，且距离事前至少 10% 专家 pair 节时门槛很远，停止本候选。指令减少没有形成可确认的实际收益；这不证明专家核已无空间，也不证明 DRAM 带宽饱和。没有新增完整 decode TPS，也不将本轮微基准换算成整模收益。

## 复现与常驻状态

实际候选在与 worker 相同的 release venv / PyTorch `2.13.0+cu132` 下构建，SM121。最初误用了容器默认 PyTorch 的构建已隔离，未加载到常驻 worker；发现环境不符后改用 worker 的解释器和动态库环境完成正式构建。

正式二进制 SHA256：`888ca81196404a2effcc5a4fd9584365dd7e88462fcd759ea9eac19dc7b13b95`。

`106-e31-scale.json` 作业 PASS，性能结论 REJECT；同一 worker、target/draft 对象，load_count=1。作业前后 completed_jobs 104→105，failed_jobs 保持 6（提交前已有的其他任务失败不属于本实验）。allocated 快照 77,561,180,160→77,695,397,888 bytes，多 128 MiB；没有声称实际省内存。权重未重载、worker 未重启，默认路径不变。

原始证据 171 文件逐一校验，归档 SHA256 `abb12f590f4255357e48263050d180ec9e53463551cc4a5f0eff53d85ace2ba5`。存于 Git 忽略的 `results/raw/e31/final/worker`；含封存源码、输入元数据、SASS 计数、全部计时和作业结果，公开 JSON 不含提示词、模型 payload、私有绝对路径或对象 ID。

```bash
python3 scripts/e31_export.py --raw results/raw/e31/final/worker --out results/e31/2026-10-23-expert-scale.json
python3 -m unittest discover -s tests -q
```

本机 210 项测试：195 通过、15 缺少可选 PyTorch 跳过。GPU 数值和真实路由检查由上述常驻作业完成。

本轮不再扩大转换或几何参数扫描。下一次结构候选应能减少更大比例的实际工作，例如重复读取或中间搬运，并先说明可减少哪些操作；当前没有足够证据承诺具体收益。
