# K4：自有紧凑 Target、状态事务与有界 Graph

**Target 研究后端已迁入，两请求正确性通过，但完整服务没有加速，默认仍为 legacy。** 本轮没有证明超过原服务，也没有完成四请求服务或彻底移除 TensorFold。公开结果为 [2026-10-24-owned-target.json](../results/k4/2026-10-24-owned-target.json)，可由 `scripts/k4batch_target_export.py` 从封存证据重算。

## 1. 迁出了什么

`CompactTarget` 自行维护 compact staging、48 层执行循环、设备请求/位置/状态指针表、commit 和 Graph 生命周期，不调用 TensorFold 的 `stage/layer_forward/commit/gdn_multi/attn_multi` 控制入口。

- `QwenState` 拥有固定容量 BF16 KV、GDN committed/candidate recurrence、conv、PLE tail/history 和位置；按有效前缀更新，拒绝区不推进有效位置。
- 自有 CUDA GDN recurrence 跨请求执行。部分接受时，一次 kernel 在所有 GDN 层/请求上重放有效前缀；另用 conv 与 PLE/位置 kernel 提交，避免按请求/层重复调度。归约顺序和 `--fmad=false` 是本轮数值契约。
- 自有 Triton attention 处理当前文本/BF16、2048-key dense-budget QSA；动态位置、因果有效长度及请求隔离由设备表提供。固定 512-key chunks、BF16 probability 和合并顺序用于保留数值。
- 自有 PLE 卷积通过设备表选取请求分段和 tail；table embed、gate 和投影仍是显式注入的兼容算子。
- 每个请求实际验证 1～8 行，合并后最多 16 行；不再分别补到 1/2/4/8。E17、Q4/W8 和当前 E29 动态预算均保留，没有新增固定草稿运行模式。

**剩余依赖：**TensorFold 权重加载、eager prefill、工作区分配，以及 dense/MoE/HC、GDN front/back、PLE embed/gate 等低层 provider；草稿 Q4 输出投影也仍注入。DeepInfer 继续承担原生 Scheduler/准入及逻辑块记账；本轮 Qwen 专属状态和计算由我们实现，不能称为 DeepInfer 已自带完整 Qwen 模型后端。

支持范围限定当前 TP1 文本模型、BF16 KV、上下文不超过 2048 的 dense-budget QSA。稀疏 QSA、视觉、量化 KV、跨卡和其他模型几何没有验收。

## 2. Graph 和状态正确性

权重/工作区与执行器生命周期绑定，固定容量 state 不重分配；地址、parity、请求布局和位置经持久设备表间接读取。Graph key 为实际总行数与 context bucket，默认最多 32 张，第二次见到形状才捕获，LRU 驱逐只在无在途 ticket 时执行。捕获后显式 replay，避免读取尚未执行的新输出。未知形状的 eager、capture、头部 JIT 是冷请求成本；正式 warm 测量排除了新增 capture/JIT，不代表冷延迟保证。

早期暴露的关键错误是 PLE 卷积仍在 Graph 中捕获 Python 分段：3+5 和 5+3 共八行却不能安全复用。自有卷积改为设备分段后，交换布局逐位一致。不能只按总行数改 key 而保留静态分段控制。

最终组件检查包含 24 组完整 48 层用例、6 轮连续拒绝/部分接受后继续生成、discard 与 slot 复用。不同 parity、单一路活跃、keep=0/partial/full 和跨 512-key attention 分块均通过；跨块探针用合成位置及零间隙 KV，只是算术检查，不是自然长文本质量或性能证据。小型四图缓存实际捕获 9 次、驱逐 5 次，状态/输出均一致。

完整动态服务诊断有 1,728 logits 行、1,424 已接受 feature 行、328 次有效状态检查通过。两种 Target 的 1,470 次同位置 E17 检查最大差异为 0。此结论仅比较本轮相同 owned batch draft 下的两个 Target；**此前 batch draft 相对原草稿的数值差异仍存在**，不能被这些 Target 检查抵消。

14 项真实 GPU 生命周期全部通过：EOS/cap、pending position、各取消阶段、在途 survivor/同 ID 复用、关闭超时/native guard、故障 fence 与新 owner 恢复。正式 compact 服务保留 27 张 Graph，未超 32 张限制，无 fallback；legacy 保留 80 张。

## 3. 固定候选执行对照

两边先恢复相同真实 prompt 状态，使用相同人工候选、验证长度和 accepted-prefix；不含 DFlash、预测器、预算或 prefill。计时从 staging 到精确头和 commit retirement。每形状每边 7 次交错，全部 logits/features/有效状态匹配，正式期间无新 capture/JIT。

| 两个 slot 的实际行数 | legacy 中位 ms | compact 中位 ms |
| --- | ---: | ---: |
| 3+5 | 67.87 | 62.55 |
| 5+3 | 68.93 | 62.05 |
| 7+2 | 65.53 | 64.69 |
| 8+8 | 73.42 | 72.48 |
| 3+0 | 49.94 | 47.40 |
| 0+8 | 56.85 | 56.40 |

消除 padding 的部分形状收益较大；本来就不补齐的 8+8 只下降约 1.3%。这只证明选定 Target 执行窗口的改善，不是服务加速或四请求扩展证据。

## 4. 五轮完整动态路径对照

相同模型、BF16 KV、owned batch DFlash、Q4/W8、E17 和原 E29 控制，每组每边五次交错运行。online 允许自然 EOS 和动态预算，逐次保留全部结果。direct 从两请求都取得首 token 后计时，包含首轮草稿与 native 回收；SSE 从 cohort 最早首 token 到全部请求结束，扣除每请求恰好一个首 token，可能包含另一请求尚未结束的 prefill。

| 两请求负载 | legacy direct | compact direct | legacy SSE cohort | compact SSE cohort |
| --- | ---: | ---: | ---: | ---: |
| 短输入，cap32 | 44.37 | 47.27 | 38.98 | 38.06 |
| 长短输入混合，cap24/16 | 60.28 | 61.01 | 57.06 | 9.57 |
| 短输入，cap128 | 39.28 | 40.13 | 36.96 | 36.75 |

单位 token/s，均为总 tokens / 总墙钟。短组 direct 约 +6.53%，95% 配对 cohort bootstrap 为 +3.24%～+10.58%；长输出 direct +2.17%，区间 −2.63%～+7.72%。SSE 短组 −2.37%，区间 −7.03%～+3.76%；长输出 −0.57%，区间 −6.39%～+5.47%。没有通过 +10% 服务目标；不能把 direct 的改善写成 SSE 胜利。

长短 SSE 的大幅差额必须按计时/到达和 prefill 解释：legacy 五次两个首 token 都约 3.1～3.3 秒才出现，cohort 只剩约 0.34～0.40 秒。compact 有三次短请求约 0.27～0.30 秒先出首 token，之后等待长 prefill，cohort 拉长到约 3.4 秒；另两次仍约 0.35～0.37 秒。原始记录全部保留，不能把这组差额直接归因为 Target kernel 慢了六倍，也不能忽略短请求后续输出的真实长等待。上一版独立服务运行 legacy/compact cohort 为 13.88/9.82，亦说明此口径受 prefill/输出时机显著影响。这里只验证到时间关联，尚未通过独立调度消融确定最小修复点。

包含 prefill 的完整请求吞吐：短组 33.83→34.03，长短组 6.35→6.26，长输出 35.51→35.30 tok/s。短组 TTFT p95 517.88→496.03ms，长输出 request TPOT p95 38.84→41.48ms，后者超过 +5% 初始约束。每边每组仅十个请求，不能据此建立稳定的尾延迟界限。**功能迁入通过，性能晋升失败，默认保持不变。**

## 5. 为什么省了计算却没有服务收益

长输出 direct 的平均分摊墙钟：verify 3670.16→3555.54ms，commit 73.76→55.34ms，但 draft 865.93→899.71ms，抵消部分收益。SSE 的 verify 3677.43→3571.44ms、commit 73.11→55.61ms，draft 则 903.56→1055.14ms。后者仍是同一草稿源码/权重；本轮不能从该现象直接断言 E17 不适配并发或某个 CUDA kernel 变慢，需进一步区分 worker/主机间隙与 GPU 计算。

长输出五轮合计两边同为 325 次 Target forward，仅 50 次 joint；actual rows 同为 1,465，legacy physical 为 1,615，compact 为 1,465。消除了约 9.3% 的物理行，却没有减少验证轮数，且多数轮次只有一个 survivor。部分 PLE/GDN/QSA 和 commit 迁移没有改变 dense/MoE 主体计算；本轮没有逐层 profiler，不能把剩余时间精确归因给某个算子。

旧异步路径的 commit_ms 一直为零。本轮改为在 commit fence 完成后，按参与 native 行数分摊一次共享墙钟；专门测试验证未完成 fence 不记账、不回收，两个结果之和等于批次时间。阶段数值是墙钟归因，含主机/等待，不是纯 GPU kernel 时长。此前版本的结果独立保存，不与本轮正式样本混合。

后续优先处理两项已暴露的系统问题：长 prefill 期间的 decode 推进/公平性，以及服务路径 draft 墙钟上涨；逐层 profile 确认余下计算瓶颈后再替换 provider。同时继续扩展四请求完整链路、重校准批次成本。当前不能只提高请求上限或重训预测器来宣称解决并发性能。

## 6. 验证、资源和复现

本地 250 项测试，235 通过、15 个可选环境跳过。远端 CPU/Torch 真正执行 7 项 state/transaction 检查，全部通过。最终 GPU 正式作业为 component369、service373、trace374、lifecycle375、CPU376、audit377。早期 362 的参考缓冲 alias、363 的 capture 未 replay、364 的静态 PLE 分段，以及 366/367 的工作区接口登记失败均保留。370 是补齐 commit 计时前的独立完整服务结果，不能择优替换 373。

所有 GPU 作业均沿共享 resident 串行；PID319、load_count=1，没有模型重载、停止或 empty_cache。最终审计无活跃批次对象，Gloo/NEXTN 环境恢复，二次 GC 后 allocated 稳定为 77,896,724,480 bytes，worker 回到 READY。该值只对应审计时点，不能把共享进程以后的显存变化归因给本轮。

证据包 `results/raw/k4-owned-target/evidence.tar.gz` SHA256：`358aa3e73e4e9d096599c1319d9a05f13177eb713d2b796006953f7e40e508e1`。自有 CUDA state 源码 SHA256：`80899a88984f7cf395f02a8f1f6ab264699dec9bf3e8091924916641cc30f7f1`；已测库 SHA256：`b63a8c527c56b2aec964fba81352bed9bb16b510db32cfe14b4d3f19ee2b64e2`。包中含初始失败、各版封存源码/manifest、正式结果和 state 二进制；旧 batch draft 的库/证据另见[前一报告](K4_OWN_BATCH_DRAFT_REPORT.md)。

使用参数见[服务说明](K4_BATCH_RUNTIME_USAGE.md)。在已准备的 resident 环境中，以唯一名称封存并串行排队；service/lifecycle/trace 的 component-root 指向本次通过的组件目录。两边仍会引用已测 batch draft 组件，不重新加载模型。

```bash
python3 -m unittest discover -s tests -q
python3 scripts/k4batch_stage.py unique-target unique-target.json k4batch_target_component.py --graphs
python3 scripts/k4batch_stage.py unique-service unique-service.json k4batch_target_service.py --graphs --component-root /workspace/e1/k4batch-20261023-unique-target
python3 scripts/k4batch_stage.py unique-trace unique-trace.json k4batch_target_trace.py --graphs --component-root /workspace/e1/k4batch-20261023-unique-target
python3 scripts/k4batch_stage.py unique-life unique-life.json k4batch_target_lifecycle.py --graphs --component-root /workspace/e1/k4batch-20261023-unique-target
python3 scripts/k4batch_target_export.py results/raw/k4-owned-target results/k4/2026-10-24-owned-target.json
```

导出器核对全部封存 manifest、实际执行源码/内核指纹、所有配对重复与 token 数、无计时内 capture/JIT、数值与生命周期检查和资源证据。默认值、冷请求与未完成范围必须随结果一起保留。
