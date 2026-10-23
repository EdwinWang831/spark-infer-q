# 双请求联合 Target 执行报告

2026-10-23。**DeepInfer 原生 Scheduler 驱动的两个 Qwen 请求，已能共享一次 48 层 GPU 前向。** 在两组中英文短请求、四轮交替对照中，decode 总吞吐由 **23.6363 提升至 38.6320 token/s（+63.4436%）**。这是同步、target-only greedy 的受控结果；尚未接入双请求 DFlash、异步 HTTP，也未切换默认服务路径。

[事前计划](JOINT_TARGET_EXECUTION_PLAN.md) · [逐次计时和校验证据](../results/joint-target/2026-10-23-joint-engine.json)

## 实现与归属

`JointTarget` 持有两份独立 GDN/QSA 状态，将两请求的行按 slot 顺序拼接，一次完成 48 层前向，保留五处 DFlash 特征和 E33 筛选输出头。每请求支持 1–8 个实际验证行，按 1/2/4/8 行 bucket 填充，总计最多 16 个物理行。提交按各请求 offset 与真实接受前缀处理，填充行不会被提交。所有 keeps 先完整预检，再写任一请求；重复提交、过早 retirement 和在途 slot 复用均被拒绝，commit event 完成后才能归还 slot。

`JointSchedulerBridge` 使用已有 DeepInfer C++ Scheduler 的 plan，不另写调度算法。同一 decode plan 中两个请求各自的 pending anchor 一起交给 `JointTargetBackend`。Prefill 仍逐请求使用原 8 行分块路径。`joint_decode=False` 让同一 plan 逐请求前向，构成此次匹配对照。

模型层、量化投影、NVFP4 专家核和状态提交算术继续借用外部 TensorFold；输出筛选继续使用本仓库 E33。此次贡献是本仓库维护的联合执行、状态所有权、提交预检与 fence、原生调度适配及完整校验证据，未新增 CUDA 专家核，也不将上游机制称为原创。

## 首轮失败与正确性

首轮既有原型 probe 报 `AssertionError: joint state differs`。它先算两个 solo 参考、再延迟提交；同宽度参考共用 scratch，首请求提交读到了第二请求的临时投影与 PLE 数据。此次独立复现首请求 `conv` / `ple_tail` 差异、第二请求无差异；为两请求使用独立 solo buffers，并在重用前提交后，联合执行的状态检查通过。原失败及源码快照已保留，没有删除失败记录或放宽比较。

| 检查 | 已完成的证据 |
| --- | --- |
| eager 与 Graph 各一组低层对照 | 每组 56 行完整 logits、56 行 DFlash 特征、16 次有效状态比较 |
| 宽度与提交 | 七种宽度组合，加 reset/reuse；不同接受前缀、零保留、padding、输入反序与不同 GDN parity |
| 生命周期与事务 | 非法第二请求不得提前提交第一请求；重复 commit、未完成 fence、stale ticket、在途复用均拒绝 |
| Graph | 80 张图预热；所选八次联合前向均 replay、零 fallback；未宣称全部 80 种组合均独立数值验证 |
| 原生生成 | 424 行完整 logits、248 次状态比较；split/joint 的输出、最终状态和逐轮 plan 一致 |
| 取消与复用 | waiting / prepared_prompt / prepared_decode 三阶段；幸存请求、取消前缀、同 ID 新 generation 与过期 handle 均正确 |
| 自然 EOS | 两个真实请求各 3 token 后 stop，状态与 solo 一致 |
| 长度上限 | 两请求 `max_tokens=1` 各输出 1 token 后 length，设备位置均为 prompt + output − 1，最后输出保留 pending |
| 资源 | logical blocks、slot leases、原生队列和 Gloo group 归还；同一常驻模型未重载 |

Graph probe 的准备耗时 **29.444 秒**，不计入以下暖态吞吐。80 张图涵盖单 slot 的四种宽度及 parity，以及双 slot 的宽度/parity 组合。计时中没有新增捕获，也没有 Graph fallback。提交预检保证非法输入不会先改另一请求；它不提供任意 GPU 故障后的状态回滚。

## 正式对照

同一台 DGX Spark、同一完整 target 权重、E33 筛选头、80 张 Graph 和原生 Scheduler，`NEXTN=0`，无草稿。两组中英文请求对，每组两个请求；第一组各生成 32 token，第二组各生成 24 token。每对四轮 AB/BA 交替，共 **16 次双请求运行 / 32 个请求**。每个 mode 累计 **448 completion token、432 计时 decode token**。

| 指标 | split：逐请求前向 | joint：两请求联合前向 |
| --- | ---: | ---: |
| decode 总吞吐（token/s） | 23.6363 | 38.6320 |
| decode 总耗时（秒） | 18.2770 | 11.1824 |
| 完整请求总耗时（秒） | 22.5865 | 15.5221 |
| 完整请求 completion 总吞吐（token/s） | 19.8349 | 28.8620 |
| TTFT p50（毫秒） | 412.533 | 426.532 |
| TTFT p95（毫秒） | 613.984 | 590.027 |
| TPOT p50（毫秒） | 84.845 | 52.163 |
| TPOT p95（毫秒） | 89.402 | 54.679 |

吞吐按 **总 token / 总墙钟** 汇总，不平均各次 token/s。四轮 decode 增幅分别 **61.4684%、67.8994%、61.0335%、63.4311%**。TTFT 中位数略增，不能声称所有延迟都改善。这里只有两组独立请求对，重复运行不能代替广泛任务覆盖，也不足以给出可靠的泛化区间。

计时使用已分词 token API：完整请求从提交到完成，包含 prefill、decode、调度与回收；decode 从两请求均得到首 token 后的共同边界开始。模型准备、engine 建立、Graph 捕获、参考校验在计时外，tokenization、文本解码和 HTTP 网络不在内。两组输出和逐轮 native plan 一致，区别仅在 decode 前向分组。

**这不是双请求动态 DFlash 或服务压测结果。** 不能与单请求 DFlash 的约 48.5 token/s 横比，也没有建立超过原生 TensorFold 的结论。

## GPU 工作量的独立证据

另行 CUPTI 诊断同样处理两个 anchor，不混入正式测速。只统计 `jointtarget.verify` 范围内 `cat=kernel` 的真实 GPU 事件，排除父级 annotation。

| 工作量 | split | joint |
| --- | ---: | ---: |
| 实际前向 / 实际行 | 2 / 2 | 1 / 2 |
| GPU kernel 调用 | 3582 | 1915 |
| NVFP4 专家核调用 | 192 | 96 |
| qmmf 量化投影调用 | 384 | 192 |

因此确实减少了重复前向及 kernel 调用。这里没有 DRAM 字节硬件计数，也不能把调用减半解释为所有算术或访存减半。

## API 与复现

依赖已准备的 TensorFold 源码/扩展、完整权重、E33 筛选头及 DeepInfer 构建，不是脱离外部依赖的安装包。下例在已有研究 session 内运行；`NativeScheduler` 使用 `max_drafts=0`：

```python
from e3_wide_target import WideTarget
from spark_infer_q.screened_head_fused import FusedScreenedHead
from spark_infer_q.kaifra_joint_target import JointTarget
from spark_infer_q.kaifra_joint_runtime import joint_engine
from spark_infer_q.native_scheduler import NativeScheduler

target = JointTarget(WideTarget(session.weights, capacity=2048),
                     FusedScreenedHead(session.weights.head))
try:
    target.warmup()  # once, before admitting requests
    with NativeScheduler(deepinfer_source, deepinfer_library, max_drafts=0) as native:
        with joint_engine(native, target, eos_ids=session.eos,
                          capacity=256, tokenizer=session.tokenizer) as engine:
            engine.submit('a', [{'role': 'user', 'content': 'Explain a cache.'}], max_tokens=32)
            engine.submit('b', [{'role': 'user', 'content': '解释一下队列。'}], max_tokens=32)
            for event in engine.drain():
                consume(event)  # caller supplies output handling
finally:
    target.close()  # after engine drains/releases leases
```

一次只允许一个 in-flight target ticket；外层仍同步驱动。`engine.cancel(handle)` 在同步 step 边界生效，本次未验证异步在途 HTTP 取消。底层支持两请求不同验证宽度的状态提交，运行时此次只调度每请求一行 anchor；双请求 DFlash 的草稿、预算、验收与回滚仍待接入。

六个独立封存 stage：`probe-v1`、`graph-v1`、`native-v1`、`trace-v1`、`lifecycle-v1`、`audit-v1`；全部经同一常驻串行队列执行并 PASS。导出器检查 stage/source/wrapper SHA、同一模型对象、逐次 plan/输出、计时汇总、GPU 核计数和最终资源。执行时 `load_count=1`，最终 READY，allocated 回到 **77,594,734,592 bytes**；没有重启、重载模型或调用 `empty_cache`。该数值是收尾基线，不是联合执行峰值显存。

```bash
python3 scripts/joint_target_export.py \
  --raw results/raw/joint-target/final/worker \
  --out results/joint-target/2026-10-23-joint-engine.json
python3 -m unittest discover -s tests -q
```

原始归档含 523 个文件，包括首轮失败、源码快照、六阶段完整结果与 trace，位于 Git 忽略的 raw 目录。归档 SHA256：`1b4808566d61a79f93201548e0419f5357a48135bd5eb31d7e38711509afda0f`。公开 JSON 包含源/扩展指纹、逐次标量、plan 和输出 hash，移除了私有路径与对象 ID；复算完整证明需要本地原始归档。

本地共运行 **219 项测试，207 通过、12 项因可选依赖缺失跳过**。新增测试覆盖提交预检与 fence、按 state 映射窗口、一次 native plan 的联合调用，以及总 token/总墙钟统计、缺失/重复运行拒绝、plan/输出不匹配拒绝和 trace 事件过滤。公开 JSON 已通过私有路径、对象 ID 和常见凭据模式扫描；raw 归档保持 Git 忽略。

下一阶段应将这条联合 Target 接入多请求动态 DFlash，按相同输出、预算与计时边界测试完整收益，再推进异步服务和更广任务覆盖。现有单请求路径保持原选择。
