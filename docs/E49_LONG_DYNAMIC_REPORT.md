# E49：四256K容量检查与长输出动态DFlash

2026-10-25。用户取消固定草稿对照，本轮只比较target-only与DFlash＋动态预算；四并发目标仍按每请求约256K理解。

**当前布局未通过四路容量门槛。** 在常驻模型上测得完整INT8状态的实际分配增量，外推四份状态＋公共工作区至少90.14GiB，已超过本轮沿用的88GiB分配限额，尚未计入私有decode Graph、screened head辅助张量和四份草稿缓存。没有通过触发OOM来验证，也没有宣称已执行四个长请求。这是当前配置的限制，不能据此断言硬件永远无法支持四路。

[事前方案与修订](E49_LONG_DYNAMIC_PLAN.md) · [脱敏证据](../results/e49/2026-10-25-long-dynamic.json)

## 长输出对照

四个正式请求均达到512-token输出上限。两格式内部，target-only与动态DFlash的全部输出tokens和完整已提交KV/scale/QSA/GDN/PLE指纹一致，各自检索3/3；正式计时内无Graph/head fallback或新增kernel登记。

| KV格式 | Target-only decode | 动态DFlash decode | Decode变化 | 完整请求wall | Wall变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 16.36 tok/s | **30.81 tok/s** | +88.37% | 253.491→240.133秒 | -5.27% |
| INT8 | 13.29 tok/s | **29.18 tok/s** | +119.58% | 259.103→239.308秒 | -7.64% |

| KV格式 | TTFT（对照→动态） | 首token后decode耗时（对照→动态） | 验证轮数（对照→动态） | 动态草稿已接受/已观察 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 222.242→223.540秒 | 31.241→16.585秒 | 511→180 | 331/478，69.25% |
| INT8 | 220.642→221.788秒 | 38.451→17.511秒 | 511→191 | 320/483，66.25% |

接受率分母是实际已观察的草稿标签；第一次拒绝后的候选不继续形成标签。若以所有送入验证的草稿行作分母，BF16为331/695（47.63%），INT8为320/737（43.42%）。两种口径不可混用，也不能把输出tokens/秒当作接受率。

| 动态路径 | 草稿/预测段wall合计 | 验证段wall合计 | 提交段wall合计 | 峰值allocated |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 2.824秒 | 10.486秒 | 0.211秒 | 81.16GiB |
| INT8 | 2.942秒 | 11.400秒 | 0.228秒 | 78.35GiB |

这些分段来自宿主计时边界，含相应读回/同步；不是纯GPU event剖析，异步提交可能在后续边界付费。与完整decode窗口的差额也包含引擎、事件和保护/进度开销，不能全部归为Python。

动态选择的实际验证宽度：BF16为`{"1": 1, "2": 10, "3": 18, "4": 60, "5": 41, "6": 8, "7": 29, "8": 13}`，INT8为`{"1": 1, "2": 8, "3": 21, "4": 70, "5": 38, "6": 7, "7": 32, "8": 14}`（键为含anchor的target行数，值为轮数）。没有固定草稿对照。

**每格式只有一组长输入配对，无多轮置信区间。** Decode改善明确存在于本次记录；完整wall受约221～224秒prefill/首输出支配，收益只有约5%～8%。不将其当作通用速度、原生Mia排名或HTTP吞吐结论。

输入为261120 tokens的中英合成ledger和英文工程教程请求，使用真实chat template；输出上限512，正常EOS生效，没有强行屏蔽停止。请求要求先列三处检索键，再详细解释推理服务。它是一个用于容量/工程测速的合成样本，不代表自然长文、多跳推理、代码正确率或全面语言质量。

每种KV格式各一组同期配对，先target-only后adaptive。两臂共用同一格式的优化target、16个Graph桶和prefill工作区；固定采用chunk1024。模型/算子/Graph/输出头预热在计时外；submit到最终drain及同步在完整wall内，状态哈希在计时后计算。decode定义为首个非空输出事件到最后非空事件的时间，分子为首事件之后输出token数，包含全部草稿、预测、验证、提交和引擎事件开销。没有中途诊断暂停，内存保护及进度落盘开销保留在计时内。

BF16与INT8的target-only输出前326个tokens一致，从第327个token开始分歧，各自仍找回三个检索键。措辞不同本身不是答案正确率下降的证明，也不能宣称两格式生成等价。两格式后续输出轨迹不同，因此横比其tok/s不能全部归因于KV格式；主结论来自每种格式内部的配对。

草稿为已有W8 MLP/Q4 head/BF16 attention与selector，使用绝对位置2040行后缀；target完整历史不裁剪。E17 NumPy预测器和E29短/长成本表冻结，没有针对256K重训或重新校准，因此测得的是现有动态策略的表现，不能称为长上下文最优预算。

## 四请求内存下界

| 项目 | 实测/推算bytes | GiB |
| --- | ---: | ---: |
| 实验前常驻CUDA allocated | 77393408000 | 72.08 |
| 常驻对象＋共享eager/prefill工作区 | 78121510400 | 72.76 |
| 一个完整INT8请求状态的分配增量 | 4665401856 | 4.35 |
| 其中KV张量存储 | 3422760960 | 3.19 |
| 其中QSA index/pool存储 | 1006694400 | 0.94 |
| 四状态＋公共区分配下界 | **96783117824** | **90.14** |
| 本轮保留的分配限额 | 94489280512 | 88.00 |

实际只构造一个完整容量状态，用该布局的分配增量外推另外三份。4.35GiB还包含GDN/PLE等固定状态和分配对齐，故比E47的4.13GiB context-growing KV/index增长项略大。至少缺2.14GiB，实际执行还需要上述未计入项。

去掉状态分配对齐，单状态去重CUDA storage本身为4664352780 bytes；公共区＋四份storage仍有96778921520 bytes（约90.13GiB），也超过限额。停止结论不依赖把分配对齐逐份累加。

原始容量报告的host-available外推没有扣除可复用的CUDA reserved slack，是保守估计，不能作为四状态实测RAM；停止结论依据allocated字节下界超过限额。没有提高限额、卸载模型或empty_cache。

另一工作区的K4四路路径限于已测capacity1536/physical2048；本轮没有改写它，不能将其短上下文吞吐移作四256K结果。物理分页可改善按需分配；四份独立历史全部填满时仍需要容纳这些实际数据。

## 保留失败与审计

首次study-v1在第一份BF16动态长请求结束时，因`formal head JIT changed`停止。预热覆盖了候选数量1..128，却只覆盖1行；subset登记按`(verify_rows, candidate_count)`区分，动态验证引入其他行数。失败版本的已完成target-only控制仅作诊断保留，不拼入正式配对。

study-v2在所有计时前预热rows1..8 × candidate counts1..128，并在任何后续时延/fallback门槛失败前先保存完整last-run记录。库与模型算术未修改，正式计时仍要求kernel哈希不变、Graph/head fallback为0。修复和原始失败分别封存。

| 阶段 | 作业 |
| --- | --- |
| 实际布局/四状态下界检查 | `520-e49-capacity-v1.json` |
| 容量阶段独立回收审计 | `522-e49-capacity-audit-v1.json` |
| 原始预热覆盖失败 | `524-e49-study-v1.json` |
| 失败独立回收审计 | `526-e49-study-audit-v1.json` |
| 修订后的完整对照 | `528-e49-study-v2.json` |
| 最终独立回收审计 | `530-e49-study-audit-v2.json` |

容量作业返回`STOP_E49_FOUR_CAPACITY`；失败study-v1保留`FAIL`，其余两次独立审计及最终study-v2/审计均通过。复用同一worker PID46235、load_count1，原算子/no_grad、进程组、环境和实验模块均恢复；最终allocated回到77393408000 bytes，worker READY。没有重载权重、清空CUDA缓存或提高限额。本轮稳定基线回收不抹去E43历史32MiB差额。

本地完整380项测试：361通过、19项可选依赖跳过。新增4项证据拒绝测试覆盖时钟/工作量、非法fixed arm、接受率/成本、正式JIT与假长输入；真实GPU证据来自上述封存作业。

## 复核与后续

原始忽略归档`results/raw/e49/2026-10-25-final/`包含537个文件，压缩包1864658 bytes，无权重。SHA256：`12dc4f1e792b6c551cb690e3e007418dccf952a6f10fbef1e713b206a44f7253`。离线导出器验证来源链、原始失败、回收、当前库/外部源码和二进制、事件/计划/工作量、接受率和计时算术，结果`PASS_E49_EVIDENCE`。

```bash
python3 scripts/e49_export.py \
  --raw results/raw/e49/2026-10-25-final/worker \
  --out results/e49/2026-10-25-long-dynamic.json \
  --capacity-version capacity-v1 --capacity-job 520-e49-capacity-v1.json \
  --capacity-audit-version capacity-audit-v1 --capacity-audit-job 522-e49-capacity-audit-v1.json \
  --study-version study-v2 --study-job 528-e49-study-v2.json \
  --audit-version study-audit-v2 --audit-job 530-e49-study-audit-v2.json
```

本轮不晋升INT8精度或运行时默认设置。E47原始raw-logit L2门槛仍失败；同格式两种执行路径相同，不能推导INT8与BF16内部数值等效。尚无四256K、长HTTP服务、多请求长上下文DFlash、1M质量或随机采样结论。后续四路工作必须先减少实际内存需求或另行评估资源配置，再验证每请求草稿/预算/提交隔离和公平调度。
