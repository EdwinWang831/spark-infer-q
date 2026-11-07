# A1 / P1D：RTX 5090 动态宽度实验

**本轮没有得到优于固定宽度 16 的可靠加速证据，继续保留固定 16 作为基线。** 校准选出的候选在 24 道新测试题上，生成 wall 的点估计增加 **1.21%**；按任务重采样的耗时减少比例 95% 区间为 **[−2.94%, +0.09%]**。区间跨零，不将候选晋升为默认配置，也不把点估计解释为已证明普遍变慢。

本报告归档用户所称的「A1 验证宽度」实验；原实验阶段为 P1D，冻结标识仍是 `rtx5090-adaptive-v1`。这是 **Qwen3-4B 官方 DFlash 的草拟前动态 block width / 跳过草拟实验**：决定同时影响草拟长度和后续验证宽度，不是草拟完成后单独截断验证预算，也不是 Spark 上的 AB1 预测器实验。

## 固定环境与比较边界

| 项目 | 冻结配置 |
| --- | --- |
| 硬件 | 单张 RTX 5090，GPU 1；未使用四卡并行 |
| Target | `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` |
| Draft | `z-lab/Qwen3-4B-DFlash-b16@b74e3a329c4d963783143b1e970d95b002be72bd` |
| DFlash 上游 | `07ebd93db9f472af339b644bb70221ad8428328a` |
| 软件 | Torch 2.13.0、Transformers 5.15.0、nvidia-ml-py 13.615.71 |
| 推理 | BF16、target SDPA、greedy、thinking=false、最多 256 个输出 token |
| 校准 | 12 道旧校准题 × 9 配置 × 2 次计时 = 216 次 |
| 独立测试 | 24 道新题 × 4 配置 × 3 次计时 = 288 次 |

全部 **504 次保留计时**均来自完整生成；没有因竞争而丢弃的计时或重试。模型常驻并复用，冻结记录的 `load_count=1`；worker 的 READY 是该次实验结束时的历史状态。

主指标为每请求多次计时的中位数之和。生成 wall 包含分配/缓存初始化、prefill、草拟、验证、提交、控制器、终止和同步；输入已分词并位于 GPU。它排除分词、HTTP/传输、排队、模型加载、评分及等待竞争进程的时间，不能当作完整服务延迟。另报 decode tok/s，其分子排除每请求的首 token，分母为 decode 时间。

## 策略与冻结选择

所有自适应决策都在**本轮草拟前**发生，只读取过去已完成草拟轮的 committed progress；没有读取本轮草稿、验证结果、未来 EOS、题目 ID 或答案标签。这里的 progress 包含该轮实际提交的推进量，不等同于纯草稿接受 token 数。初始动作都是 16。

| 方法 | 动作规则 |
| --- | --- |
| `official16` | 未修改的官方生成器，固定宽度 16 |
| `lagged` | 上一草拟轮 progress ≤4 则用 8，否则用 16 |
| `ewma` | 历史 progress 的 EWMA（alpha=0.5）≤4 则用 8，否则用 16 |
| `candidate` | 校准从 6 个规则中选出的 `short0_skip2` |

候选网格为短宽度阈值 `{0,2,4}` × 跳过耐心 `{1,2}`。连续指定次数的草拟轮都仅推进 1 token 时，跳过一次草拟（width=1）；跳过不更新接受历史/EWMA，并强制下一轮恢复 16。阈值 0 禁用 width=8，因此选中的 `short0_skip2` 平时用 16，只在连续两次低推进后跳过一次。

选择仅按校准集上每请求中位 wall 之和最小确定，平局按名称排序，不按质量或测试结果重选。校准中该候选的耗时减少点估计约 1.14%，区间跨零。`lagged` 和 `ewma` 是本地简单对照，**不是 DSpark 或 vLLM 的复现**。

测试集由 GSM8K、HumanEval、SQuAD 各 8 题组成；使用固定公共源快照和哈希选择，排除先前全部 36 个题目 ID 及来源组（包括整个 SQuAD 段落）。任务清单和执行源码在推理前冻结；分析源码在校准结束前冻结；候选选择在新题推理前冻结。具体顺序和哈希见 [seal](../results/rtx5090/adaptive-v1/seal.json)、[analysis-seal](../results/rtx5090/adaptive-v1/analysis-seal.json)、[selection](../results/rtx5090/adaptive-v1/selection.json) 与 [audit](../results/rtx5090/adaptive-v1/audit.json)。

## 新题结果

下表每项为 24 个请求分别取三次计时中位数后汇总；正的耗时变化表示更慢。输出长度不同，不能将这些数字称为固定输出量的加速比。

| 方法 | 总生成 wall | 相对固定 16 的耗时变化 | decode tok/s | 输出 token 总数 |
| --- | ---: | ---: | ---: | ---: |
| 官方固定 16 | 9.5319 s | 参考 | 238.19 | 2182 |
| lagged | 10.5751 s | +10.94% | 215.79 | 2204 |
| EWMA | 10.3162 s | +8.23% | 221.43 | 2204 |
| 候选 `short0_skip2` | 9.6471 s | +1.21% | 234.96 | 2180 |

统计量定义为 `1 − 候选总 wall / 固定 16 总 wall`，正值表示节时。以请求为单位、按数据集分层进行 4000 次成对 bootstrap，随机种子为 20261106：

| 方法 | 耗时减少比例 | 95% bootstrap 区间 |
| --- | ---: | ---: |
| lagged | −10.94% | [−15.26%, −5.89%] |
| EWMA | −8.23% | [−13.33%, −3.79%] |
| 候选 `short0_skip2` | −1.21% | [−2.94%, +0.09%] |

候选在 **23/24** 个请求上与固定 16 输出 token 完全相同。该事后子集的耗时减少比例为 −0.36%，区间 [−0.97%, +0.21%]；主结果仍保留全部 24 题。另一题 `gsm8k/test/935` 的输出从 232 变为 230 token，耗时增加约 84.8 ms。子集分析不替代主结果，也不作为重新选题依据。

四种方法的逐题主评分相同：GSM8K **7/8**、HumanEval **5/8**、SQuAD EM **7/8**、F1 **0.975**；均无测试输出上限命中。数学严格格式分为固定 16 的 2/8、其余三种的 3/8；主数值分沿用此前冻结的尖括号归一化规则。样本太小，不能证明质量非劣。

四种方法均仅 **21/24** 与独立 HF target 输出完全一致。已知 BF16 计算形状差异会影响输出，不能宣称 bitwise losslessness。

## 轨迹诊断与已有正确性证据

每个请求取一条稳定轨迹计数，不将三次重复相乘：

| 方法 | 草拟调用 | 跳过草拟 | target 解码/验证轮数 |
| --- | ---: | ---: | ---: |
| 固定 16 | 382 | 0 | 382 |
| 候选 | 380 | 9 | 389 |

候选只少了 2 次完整草拟，却多了 7 次 target 轮次。这与省下的工作被额外轮次抵消的解释一致，但不是逐轮反事实归因。控制器 `choose/observe` 的 CPU 子计时共 0.8891 ms，约占候选总 wall 的 0.0092%；该子计时不覆盖所有适配器语句，完整 wall 则包含全部适配开销。

历史 GPU 运行记录包含以下检查：

- 固定 1/8/16 适配器与官方同宽输出、推进量一致；始终跳过与独立 HF target 一致；连续跳过后恢复 8/16 的缓存前沿和重复输出检查通过。
- 全部 36 个请求的固定 16 适配器与官方生成器 token/推进量精确一致；自适应配置的非计时运行检查 target cache、draft cache 与待消费特征的前沿。
- 每配置重复运行的 token、动作和推进量一致；冻结分析器逐轮从因果历史重算并核对动作。
- NVML 校准采样 2354 次、测试采样 2718 次，最大间隔约 0.111/0.103 秒，无记录到的竞争事件；仍不能排除采样间的短活动或共享 CPU 干扰。

结论限于单模型、单卡、24 道公共测试题的预实验。历史 progress 阈值和跳过思路已有 [E18](E18_PREDRAFT_COMPUTE_REPORT.md) 等前期探索，本结果本身没有建立新颖性。它保留了有用的基线和负结果，不能推论所有动态宽度方法无效。已观察的 24 道测试题也不能再被后续调参当作未见测试集。

## 证据与 CPU 复算

[冻结证据目录](../results/rtx5090/adaptive-v1/)包含任务清单、完整计时、token 与动作轨迹、质量结果、选择、汇总、源码快照、任务回执和审计，共 30 个原始文件。本次发布保持这些文件逐字节不变；`executed-sources/` 是当时执行的源码，不替换为后续版本。

- [校准计时](../results/rtx5090/adaptive-v1/calibration-run.json)与[选择](../results/rtx5090/adaptive-v1/selection.json)
- [测试计时](../results/rtx5090/adaptive-v1/test-run.json)、[质量](../results/rtx5090/adaptive-v1/test-quality.json)与[汇总](../results/rtx5090/adaptive-v1/summary.json)
- [冻结控制器/生成器适配](../results/rtx5090/adaptive-v1/executed-sources/rtx5090_adaptive_policy.py)与[分析器](../results/rtx5090/adaptive-v1/executed-sources/rtx5090_adaptive_analyze.py)

在仓库根目录，使用 Python 3.10+ 标准库即可核对：

```bash
python3 scripts/rtx5090_adaptive_audit.py
PYTHONPATH=results/rtx5090/adaptive-v1/executed-sources \
  python3 -B -m unittest discover \
  -s results/rtx5090/adaptive-v1/executed-sources \
  -p test_rtx5090_adaptive.py -v
```

审计脚本验证 `audit.json` 中的 28 个文件哈希、源码封存、mirror 记录的 audit 哈希及冻结顺序，并用冻结分析器在临时目录重新生成选择和测试汇总。除 `created_unix` 外，全部字段（包括逐题指标、bootstrap 区间和来源哈希）须完全一致。分析器同时核对任务/配置覆盖、重复计时标记及因果动作回放；另外运行 3 个冻结控制器状态测试。

这是对既有 GPU 结果的 CPU 复算，使用冻结质量文件，不重新运行 GPU 或评分沙箱。源码快照中的 GPU runner 仍依赖历史常驻 session 和指定 DFlash 环境；任务回执保留原服务器路径用于溯源，不是任意机器可直接执行的启动命令。
