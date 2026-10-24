# K4：长 Prefill 期间的 Decode 推进与快速 Prompt 接入

本轮修复了已出首 token 的请求被连续 prompt 轮次挡住的问题，并把已提交的 E40 自有 prefill 模块接入两请求异步服务。**五轮固定到达顺序下，短请求最大输出间隔从约 3 秒降到 0.28～0.31 秒；完整请求吞吐 11.54→25.60 tok/s。** 这是 prompt/调度与完整服务改善，不是纯 decode kernel 提速，也没有完成四请求服务。

公开数据：[2026-10-24-prefill-progress.json](../results/k4/2026-10-24-prefill-progress.json)。原 [Target 迁移报告](K4_OWN_TARGET_REPORT.md)保留原来的未通过性能结果，不用本轮替换或拼接其样本。

## 1. 已验证的调度原因与修复

固定 DeepInfer `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` 的 NORMAL Scheduler 每次只排 prompt 或 decode；prompt step 会将 `decode_step` 归零，未完成的 prompt 回到 waiting 队首。原双请求桥接把两种 prefill polling interval 都设为 1，因此连续 prompt chunks 会反复优先于已经 ready 的 decode。只开启 chunked-prefill 不会自动交错两种工作。

新增可选 `interleave_prefill`：双 slot、单 rank 下将 polling interval 设为 2；上一计划是 prompt、且确有尚未终止的 ready decode 时，把下一次 polling phase 置为 1。实际请求仍由原生 Scheduler 选择、准入和记账；一次 decode 后到达 phase2，恢复 prompt 机会。没有 ready decode 时继续 prompt，不制造无意义的 decode 空转。没有改 MTP 的 schedule/process 计数、预算算法或 GPU 源码。

该控制只允许 two-slot/single-rank，避免破坏跨 rank 同型投票。准备中的 plan 仍禁止重复 schedule；所有状态发布和回收仍经过原有 CUDA fence。原生 CPU 跟踪实际跑了 NEXTN0/7：相同输出下，旧版在短请求首 token 后连续跑三个长 prompt chunk，新版下一轮先推进 ready decode。单元测试还覆盖无 ready、已取消、未完成 prompt 和在途 plan 的分支。

## 2. 接入 E40 的边界

从本仓库另一工作线的已提交 `b1dfe0b` 原样引入 `prefill.py`、`prefill_tensorfold.py`、相应测试与 MIT 归属声明；没有修改主工作区或其冻结实验。自有 executor 管理分块、单流工作区、结果所有权与失败隔离；provider 管理层顺序/GDN prompt 状态，仍借用 TensorFold 的加载、权重、缓冲和低层算子。来源见 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)。不把适配实现称为全部原创 kernel 或完全去依赖。

K4 的 `batch_engine` 新增 caller-owned `prefill_executor`、`prompt_token_budget` 和 `interleave_prefill`。执行器从当前 worker CUDA stream 创建；prefill 生成末行 logits 和完整 DFlash features，交接到既有 compact Target、owned batch DFlash、E17 与原 E29 动态预算。prefill 失败会隔离该 executor；不能重试部分修改过的请求状态。关闭先完成 owner/fence，再释放工作区。

测试的快速方案使用 128 行 executor 和 native prompt budget128。相较 512 行整块，这给 decode 留出更细的调度边界；它不是任意输入上的硬毫秒时限。对照的快速方案同时改变 executor 和 prompt quantum，不能把所有增量归因于单独一个 E40 算子。

## 3. 事前固定的三种方案与到达方式

各方案共用同一 compact Target、模型/量化/BF16 KV、owned batch DFlash、Q4/W8、E17 NumPy 模型和原 E29 控制。没有训练/更换预测器，没有固定草稿模式。

| 方案 | Prompt 计算 | Native prompt budget | Decode/prefill 推进 |
| --- | --- | ---: | --- |
| baseline | 原 rows8 | 32 | 原 prompt 优先 |
| interleave | 相同 rows8 | 32 | ready decode 与 prompt 轮次交错 |
| fast_prefill | E40 executor，chunk128 | 128 | 相同交错策略 |

短输入 cap32 与 cap128 使用两个不同提示同时到达；`staggered_prefill` 先让中文短请求出首 token，客户端观察到它之后才提交长输入，不使用猜测的固定 sleep。短/长请求 cap32/24。该设计稳定复现原先“已开始输出却暂停数秒”的问题，避免把首 token 顺序的随机差异当作 kernel 差异。

每组每方案五轮，轮换方案顺序。先做 9 个直接数值诊断，再预热所有服务形状；45 个正式 cohort 内没有新增 Target capture 或 exact-head JIT，也没有晚形状重试。每次输出都与 baseline gold greedy tokens 检查，所有样本保留。

## 4. 完整请求吞吐与等待

完整请求从第一个请求开始到 cohort 全部结束，包含 prefill、首 token、草稿/验证、HTTP 与最终回收；总实际 completion tokens 除以总墙钟。

| 两请求负载 | baseline | 只交错 | 快速 prefill + 交错 |
| --- | ---: | ---: | ---: |
| 短输入 cap32 | 33.40 | 32.33 | 37.76 |
| 短输入 cap128 | 35.28 | 35.95 | 38.09 |
| 短请求已开始，长输入随后到达 | 11.54 | 11.42 | 25.60 |

单位 token/s。快速方案相对 baseline 的配对 cohort bootstrap 95% 区间：短组 +13.05%（+10.58%～+15.42%），cap128 +7.99%（+3.18%～+13.71%），staggered +121.89%（+113.61%～+129.30%）。每组只有五个 cohort；不把 token 当作独立请求样本。

三种方案下，staggered 短请求的五个最大 chunk gap：

| 方案 | 五轮最大间隔，ms |
| --- | --- |
| baseline | 3130.4 / 3130.1 / 2965.0 / 3185.2 / 3158.1 |
| interleave | 362.1 / 374.9 / 394.0 / 374.2 / 393.9 |
| fast_prefill | 283.5 / 303.0 / 308.5 / 298.3 / 287.8 |

只交错的整组吞吐没有改善，长请求 TTFT p95 还从约 3113ms 增至 3766ms：原来的工作没有减少，只改变了顺序。短请求仍需跨多个 prompt chunk 分享 GPU，平均 TPOT 也不能用最大停顿的改善替代。快速方案将平均 prompt 墙钟 3187.6→674.9ms，长请求 TTFT p95 降至 805.4ms，才同时缩短整组完成时间。

SSE cohort 另从最早首输出计时，扣除每请求恰好一个首 token，仍可能包含 companion prefill。短组 37.50→39.89，cap128 36.70→39.25，staggered 11.69→26.08 tok/s。这些数值不能与单流约48 tok/s、旧报告不同首 token 边界或流速率之和直接比较。

快速方案在本轮短组/长输出两流的 TTFT、request TPOT p95 均低于各自 baseline；staggered 长请求 TPOT p95 21.82→22.45ms，约 +2.9%。每方案每组仅十个请求，尚不足以建立长期尾延迟界限；cap128 完整请求增幅也低于 +10% 阶段目标。默认入口仍保持原配置，改进通过显式开关启用。

## 5. 草稿耗时归因

正式样本之外另跑三轮 direct 与三轮 SSE，冻结相同短负载。按完整有序 context 长度/anchor 签名匹配，有 12 个 backbone 与 12 个 selector 签名可比较。匹配签名的均值：backbone wall 9.52→11.35ms，thread CPU 9.51→11.12ms，GPU event span 9.69→11.52ms；selector wall 3.13→3.06ms，基本不变。

这把新增耗时定位到 backbone 的提交/执行区间，而非 E17 或 selector。thread CPU 同时增加，不能单凭该数据说全是 GIL 阻塞；GPU span 也包含 eager 主机提交空隙，不是逐 kernel 活跃时间之和。一次工作量、签名和 CPU 放置的独立探针用于进一步区分，结果与正式服务样本分开保存。

这台 Spark 的允许 CPU 分为 `cpuinfo_max_freq=2808000` 和 `3900000` 两组。后者为 CPU5～9/15～19。另做三轮 direct/default SSE/高频组 SSE/低频组 SSE 的交错探针，实际记录每次起止 CPU；保持模型/输入/输出检查与事件边界相同。每个 cohort 的平均累计 backbone wall：direct183.83ms、default SSE222.51ms、高频组169.43ms、低频组881.31ms；对应 thread CPU183.80/217.40/166.18/878.28ms。selector 分别57.53/59.86/56.54/97.04ms。相同签名的结果及未匹配调用均保留。

这支持 CPU 放置是草稿差额的一个实际因素；不能据此把整个残余差额归因于单个原因。新增 `worker_cpu('performance')` 只在调用 owner 线程上选择原允许 mask 内最高最大频率的一组，结束或异常时恢复原 mask，不更改其他线程、频率或 governor。默认不设置 affinity。无法读取拓扑/设置 mask 则明确失败，没有猜测 CPU 编号作为通用默认。

再独立测五轮完整 SSE，对照两边都启用相同快速 prefill/交错，仅更换 worker CPU 放置：

| 两请求负载 | 默认放置完整请求 | 高频组完整请求 | 默认放置 SSE cohort | 高频组 SSE cohort |
| --- | ---: | ---: | ---: | ---: |
| 短输入 cap32 | 39.31 | 40.61 | 42.86 | 44.00 |
| 短输入 cap128 | 36.45 | 39.12 | 37.14 | 39.73 |
| 短流随后到达长输入 | 25.73 | 27.23 | 26.32 | 27.77 |

单位 token/s。这是新一轮同期对照，**不与前一张表相乘或拼接成单一提升率**。完整请求增幅及配对95%区间：短组 +3.32%（−2.04%～+8.88%，未建立稳定提升），cap128 +7.31%（+6.05%～+8.57%），staggered +5.81%（+3.74%～+7.45%）。cap128草稿平均累计墙钟1096.10→799.14ms，Target verify3628.24→3577.93ms；收益主要落在草稿阶段，未改草稿算术或 E17 权重。

这30个正式cohort期间也没有新capture/JIT，另有924 logits行、924 feature行、244状态及16个prompt交接检查通过。短/长输出两流的主要TTFT/TPOT p95没有超过本轮默认放置的 +5% 初始约束；staggered长请求TTFT p95803.70→809.70ms。每组仍只有十个请求，保留为可选配置，不承诺更广到达率、不同机器或长期尾延迟。

## 6. 数值、生命周期与资源

三种方案的直接诊断检查通过：1,386 decode logits 行、1,386 feature 行、366 次有效状态、44 个 prompt 交接。快速 prefill 的末行完整 logits、全部 prompt features 与 rows8 参考匹配，decode 继续检查 logits/features/状态；所有45正式 SSE cohort 输出通过 gold 检查。旧 batch draft 相对原始逐请求草稿的数值差异仍是已披露限制；本轮没有移除它。

快速方案加performance CPU放置的14项真实 GPU 生命周期通过：NEXTN0/7、EOS/cap、pending position、不同取消阶段、在途 survivor/同ID复用、关闭超时与native guard、故障隔离与新owner恢复。CPU/Torch实际跑21项state/prefill/phase/事务/affinity测试，全部通过。本地264项，249通过、15个可选环境跳过。

正式服务最多31张Target Graph，无fallback，保持32张上限。GPU工作始终通过共享resident队列串行；同一PID319/load_count1，没有重载/停止模型、没有empty_cache。最终资源审计及封存指纹见公开JSON；后续共享进程显存不能归因给本轮。

最终审计391：无活跃批次对象，Gloo/NEXTN环境恢复，主线程仍允许CPU0～19，二次GC后allocated稳定为77,963,833,344 bytes，worker回到READY。原始包 `results/raw/k4-prefill-progress/evidence.tar.gz` SHA256：`0a4678aaccb9b28d7c898400e7c5d52e699591a6da8ef9adc3a6aaad8e912c67`，包含所有正式/失败封存、内核库与完整记录；补充affinity之前的包另存，未覆盖。

初始原生CPU探针378失败是观测器读了target-only不生成的prompt metrics，改为记录实际native Plan.rows后381通过。380上传中断留下未封完目录，未入resident执行；保留记录，不覆盖冻结源。它们不作为正确性或性能成功样本。

387在写完CPU放置的测量后，因跳过附加探针的空scope取模而失败；修正附加循环，完整重跑五轮388，387不与388合并。390首次上传也中断，改用新封存版本后才执行。提交器现先核对完整传输SHA，在临时目录核对manifest，再原子发布；相同定义可恢复排队，已使用的不同定义被拒绝，不把半传输目录当作可执行快照。

复现使用已准备环境与常驻模型，所有版本/作业名必须唯一：

```bash
python3 -m unittest discover -s tests -q
python3 scripts/k4batch_stage.py unique-native unique-native.json k4batch_progress_native.py
python3 scripts/k4batch_stage.py unique-service unique-service.json k4batch_progress_validate.py --graphs
python3 scripts/k4batch_stage.py unique-draft unique-draft.json k4batch_progress_draft.py --graphs
python3 scripts/k4batch_stage.py unique-life unique-life.json k4batch_progress_lifecycle.py --graphs
python3 scripts/k4batch_progress_export.py results/raw/k4-prefill-progress results/k4/2026-10-24-prefill-progress.json
```

导出器核对全部封存manifest与实际执行模块指纹、每组每方案所有重复及token计数、数值/生命周期/环境检查，拒绝正式时间内的新capture/JIT。GPU主体kernel、批次成本重校准、更广输入/到达率及四请求完整链路仍待后续工作。
