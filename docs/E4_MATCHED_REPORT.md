# E4：同机原生 MTP 与 DFlash2 成本预算实测

日期：2026-10-20（America/Los_Angeles）。[预先固定的方法](E4_MATCHED_PLAN.md)、[主实验完整数据](../results/e4/2026-10-20-matched.json)。

**结论：DFlash2 动态预算在这批新输入上比自身固定六/七草稿约快 2%，但没有建立相对原生 TensorFold MTP 的优势。当前更大的工程瓶颈是冷 prompt 处理；decode 中 target 验证约占 80%，整个 drafter 约占 18%。** 原始原生计时存在明显首轮效应，并且双方预热不对称，不能将原始汇总中约 3% 的 decode 差距当作公平排名。逐用例预热补充对照的原生 MTP 为 55.99 token/s，候选原始动态为 47.37；补充为事后顺序 session，且跨引擎输出仍有差异，不能解释成纯算法差距。

## 范围与身份

这是同台 DGX Spark / GB10 / SM121 的短上下文、text-only、单请求 greedy 实验。底座仍是 TensorFold target 执行、我们适配的 DFlash2 和原生 DeepInfer MTPVarlen；没有接入完整 DeepInfer 引擎或多请求 scheduler。

| 项目 | 固定值 |
| --- | --- |
| MiaAI-Lab 配方 | `656879fd752749bbd11a6e1a065a326c00940911` |
| TensorFold base | `db281878ddb836fd0df510d8771ecb7e0fe47d26`，配方补丁安装态 |
| 实际原生 ELF SHA256 | `cb2eb4e21e2031cad267ad646c80170df894f76b426ebddcba0895628682185d` |
| SM121 AOT manifest SHA256 | `f87c8e05dd76a68c843cc9f3c823217eeee93f59f49732233b106ab271459189`，320 个条目 |
| 容器 image SHA256 | `49a4adb03c4acdff4e4c54b6e8b491b01203cc9e00070ef0e47cea27ce249694` |
| target checkpoint | NVIDIA FP4，revision `b9ce18a6887e18c089fdea8509a33e63429a3300` |
| KV / 输入 / 采样 | BF16 / 纯文本 / temperature=0、top-p=1、top-k=1，显式 thinking 开关 |
| 容量 | 原生 context=1017，含六草稿及 anchor 的内部 buffer=1024；候选 capacity=1024，prompt+limit<=1017 |
| 原生 MTP | 最大六草稿，累计概率阈值 0.4，product streams=5 |
| 草稿输出词表 | 原生嵌入 79,591 IDs；候选完整 248,320 IDs |
| 缓存 | 原生 prompt-cache=0、repeat-cache=0；每个计时请求 cached prompt tokens=0 |

六个新的英文提示各自 thinking 开/关，共十二个自然 EOS 设置；另有两个 256-token capped 设置。**共八个不同提示文本、十四种设置，不是十四个独立问题。** 自然任务包括整数、JSON、两个纯 Python 函数、散文和文字算术；自然输出上限为 direct 256 / thinking 512。没有中文、工具调用、多轮或质量数据集覆盖。

四个加载 session 按原生 A → 候选 B → 候选 B → 原生 A 运行。每模式每设置四次；候选 serial / fixed6 / fixed7 / dynamic7 用循环 Latin square，原生 serial / MTP 交替。合计 **336 次计时**。动态成本及接受率先验沿用 E3 独立校准，未用本批输出调参；没有叠加局部学习策略、EOS 裁剪或草稿常量缓存。

## 原始 ABBA 测量

下表保留全部原始样本，含首轮初始化成本。每个设置先对四次 **耗时** 取中位数，再用实际 decode token 总数除以中位耗时之和；不平均 token/s。decode 排除首个输出，冷 TTFT 包含 prompt 和首个输出，完整请求时间由同一 loopback SSE 客户端测量。加载和 oracle 不计入。

十二个自然设置均 EOS：

| 路径 | decode token/s | 十二设置中位请求耗时之和（秒） |
| --- | ---: | ---: |
| 原生 serial | 23.20 | 61.10 |
| 原生 MTP | 49.40 | 29.73 |
| 我们 serial | 23.43 | 87.15 |
| 我们固定六草稿 | 47.67 | 58.45 |
| 我们固定七草稿 | 47.72 | 58.53 |
| 我们动态零至七草稿 | 48.83 | 57.74 |

两个 capped 设置均生成 256 token，合计 510 decode token：

| 路径 | decode token/s | 两设置中位请求耗时之和（秒） |
| --- | ---: | ---: |
| 原生 serial | 19.97 | 25.96 |
| 原生 MTP | 47.34 | 11.18 |
| 我们 serial | 23.42 | 26.33 |
| 我们固定六草稿 | 43.36 | 16.32 |
| 我们固定七草稿 | 42.98 | 16.37 |
| 我们动态零至七草稿 | 43.97 | 16.09 |

全部十四设置原始汇总：原生 MTP 48.82、候选 fixed6 46.39、fixed7 46.30、dynamic7 **47.37 token/s**。候选动态比固定六/七分别高 **2.12% / 2.31%**；自然设置分别高 2.43% / 2.32%。候选每个重复编号的 decode 总耗时均优于两种固定长度，重复级耗时比 fixed6/dynamic 为 1.0200、1.0225、1.0226、1.0115。样本规模和任务分布有限，不能称统计显著或通用收益，也不能将不同 E3 实验的提升相乘。

跨路径实际输出有差异：自然设置原生 1359、候选 1323 decode token，全部设置为 1869 / 1833。跨引擎数字是路径观测，不是控制了输出轨迹的纯算法加速比。

## 首轮效应与预热控制

原始计时中，原生重复 0/2 通常比 1/3 慢。相同输出、相同 MTP round/acceptance 下，LRU capped 代码的原生 MTP decode 四次约为 **5.45 / 3.43 / 5.51 / 3.37 秒**；散文 thinking 为 10.62 / 7.00 / 11.60 / 7.07 秒。候选动态各设置最大/最小 decode 耗时比不超过约 1.12。

方法审计发现：原生每 session 仅用通用 ready 提示预热；候选每设置在计时前做了完整 serial 和三种非 serial oracle。**原始两边的预热并不对称。** ABBA 和交替顺序不能消除此差别。原生源码 `cuda_engine.zig` 的 GraphKey 包含 sequence、buffer version、行数、parity 和 context；`graphed` 在 key 缺失时先 eager 执行再捕获，证明执行路径确实可能包含初始化成本，但没有逐次 capture 追踪，不能将全部波动精确归因于 CUDA Graph。

为量化影响，保持主实验源哈希、原生二进制和所有服务参数不变，另外重载一次原生服务，对每个设置先完整运行 serial 与 MTP，再各计时四次。补充为 **28 次完整预热 + 112 次计时**，每次仍关闭 prompt/repeat cache。这是发现问题后的原生单路径顺序控制，复用已见输入，不是原始 ABBA、独立测试集或新优化结果。原始 336 次不删除、不替换。

[补充完整数据](../results/e4/2026-10-20-native-warmup-control.json)已核验：112 次计时及 28 次预热的输出摘要、计数、正文、推理文本、停止原因和答案检查均与原始原生记录一致，140 次请求 cached prompt tokens 均为零。

| 设置 | 原始原生 MTP（token/s） | 逐用例预热原生 MTP（token/s） | 候选动态原始值（token/s） |
| --- | ---: | ---: | ---: |
| 自然十二设置 | 49.40 | **57.00** | 48.83 |
| capped 两设置 | 47.34 | **53.45** | 43.97 |
| 全部十四设置 | 48.82 | **55.99** | 47.37 |

补充原生 serial 的自然/capped/全部汇总为 24.73 / 24.91 / 24.78 token/s。MTP 各设置最大/最小 decode 耗时比降至不超过 1.129，serial 不超过 1.049；LRU capped 代码 MTP 中位 decode 3.517 秒，四次最大/最小比 1.017。补充自然设置完整请求中位耗时之和为 26.05 秒，冷 TTFT 中位数之和为 2.21 秒，各设置 TTFT 中位数再取中位数为 177 ms。

预热差别确实会实质改变原生基线，**不能使用原始约 3% 的汇总差距宣称接近原生性能**。补充只有一个顺序 session，跨引擎还有六个不同输出设置，不能将表格差距变成精确算法加速比；但现有证据没有支持候选优于原生。事后筛选八个输出摘要相同的设置也只能作诊断，不能拿它替代完整任务集制造领先结论。

## 正确性证据

- 十四设置的 prompt IDs 在两条路径完全一致，并已对照 HF tokenizer；原生 MTP 与其 serial 的输出摘要、计数和正文/推理文本一致。
- 候选全部 **224 次计时**的完整输出 IDs 与本路径 serial oracle 一致。另有 **84 次独立非 serial oracle**检查通过，覆盖已提交 logits 和最终有效 GDN/QSA 状态。计时运行没有逐次做完整状态对照，不能把独立 oracle 描述成每次计时都验证状态。
- 原生接口只暴露输出计数、正文/推理文本和 12 位 token SHA，没有返回完整输出 IDs、logits 或状态。SHA 为逗号分隔十进制 ID 的 SHA256 前十二位；候选据此复算。跨路径 **8/14** 设置摘要一致，六个不同；不是原生完整 IDs 的直接读取，也不构成跨引擎全量数值一致性证明。
- 十个可客观判定的自然设置，六模式各四次，共 **240 次答案检查全部通过**。其中四个函数设置形成 **96 次受限函数执行**，覆盖空列表、负数、重复及部分浮点输入。并非完整代码质量评测。
- 散文只检查非空，没有语义质量分；两个 capped 输出只检查非空前缀，截断代码不称为正确完整程序。所有模式的检查结果在四次重复间一致。

此前的 target prefill/serial 数值差异与官方 drafter BF16 fidelity 尚未解决；本轮沿用候选串行 prompt 算术，没有重新执行官方 FP32 数学对照或证明独立整模正确性。greedy 的本路径保真不能替代这些问题。

## 时间花在哪里

自然设置的各用例冷 TTFT 中位数再取中位数：原生 MTP **182 ms**，候选动态 **2412 ms**。十二设置 TTFT 中位数之和为 **2.20 / 30.65 秒**，完整请求为 **29.73 / 57.74 秒**。这些是原始 session 的冷 prompt 观测，受上述预热差异影响；不能将该比值称为纯 prefill 内核加速比。候选仍逐 token 处理 prompt，已足以确定它是端到端的主要工程瓶颈。

候选动态全部计时的 decode 阶段耗时占比：

| 阶段 | 占比 |
| --- | ---: |
| drafter（含其投影） | 18.02% |
| target 验证（含特征捕获） | 80.25% |
| accepted-prefix 提交 | 1.29% |
| 其余 | 0.44% |

“其余”包括预算、采样、拼接、流式及 Python 等开销，没有单独测得纯控制器成本。不是全部 drafter 时间都属于输出 head。动态共 1909 步，实际验证行数（含 anchor）4/5/7/8 分别出现 119/677/320/793 次；未选零/一草稿。fixed6/fixed7 分别为 1900/1852 步。不能只凭步数推断耗时。

原生默认已用 79,591-ID 草稿词表，候选仍投影完整 248,320 词表。草稿网络、词表、特征捕获、数值路径、prefill 和图执行都不同，不能把整条路径差异全部归因于动态预算或 Python/Zig。

## 运行障碍与资源

初始原生 context=1536 预检失败：内部 1543 行需要未提供的 BF16 dense NCH=4 AOT 变体，报 `MissingTritonVariant`，清理又报 `CUDA_ERROR_ILLEGAL_ADDRESS`。默认 262144 context 可加载；正式新提示执行前改用 context=1017 / buffer=1024，覆盖所有本批请求，未修改或重建外部内核。裸 `--help` 复用了 Python 帮助文字，实际测量仍为经过 SHA 校验的 ELF 原生程序。

预检期间另有一次 host 无权限结束 root worker 的操作失败，重复 worker 在正式 study 前均已清理。正式 supervisor 每阶段断言无残余 worker，再启动下一路径。主实验四个 session 均完成，耗时约 61.16 分钟，无模型并驻测量或 cgroup OOM 事件。

原生 ready 为 88.05 / 104.02 秒；候选为 370.11 / 398.11 秒，其中 target load 为 361.20 / 391.77 秒。候选两 session 的 PyTorch peak allocated/reserved 为 **72.21 / 72.26 GiB**，它不等于统一内存整机峰值。阶段末 MemAvailable 约原生 28.8 GiB、候选 28.7 / 27.3 GiB，亦非全程最小值。候选最长已提交位置 435，原生按 prompt+输出计数为 456；1024 buffer 的分配不能推出长上下文容量。GPU 阶段端点为约 2450–2528 MHz、66–70°C，没有手动更改时钟/功率，也没有全程遥测。补充控制约 11.94 分钟，ready 98.05 秒；完成后 worker 数为零，cgroup memory events 各项为零。实验容器随后停止，最终 exited、OOMKilled=false。

## 工程选择与下一阶段

1. **优先解决 prefill 数值边界并接入成批 prompt 执行。** 用相同 prompt 的 teacher-forced logits/路由/状态定位差异，选定数值政策，再对照冷 TTFT 和完整请求；不能仅为 token 一致而退回逐 token prompt，也不能未经质量检查直接启用不同算术路径。
2. **优化验证执行与特征捕获。** 这是 decode 最大成本；评估静态 buffer、图执行和移入成熟原生执行路径，同时保持拒绝后继续生成的 logits/状态检查。不能只围绕草稿策略开展所有工程。
3. **独立评估 DFlash2 草稿 head 量化、缩词表和组合。** 当前复用 target 的 BF16 输出矩阵；需另建草稿投影，保持 target 权重和验证逻辑完整。独立副本可能增加总驻留，投影读量变少不等于整机省内存。先测 head 时间，再比较提案变化、各位置接受率、decode、完整请求及总内存；包含中文、数学/LaTeX、代码/JSON 的新冻结输入。量化位宽和词表大小由这些对照选择。缩词表后必须先把局部 top-k 索引映射回原始 token ID，再送入 candidate selector；检查特殊 token 覆盖，并重新运行输出/logits/状态 oracle。

本轮约 2% 的候选内部收益是正面但有限的结果，支持继续研究成本预算；尚未建立跨任务通用自适应优势。完整引擎集成、多请求、长上下文、FP8 KV、vision、随机采样以及 vLLM/SGLang 同机对照仍未完成。

更具体的后续研究假设是：**草稿精度、词表大小和验证预算的联合选择，能否优于分别固定它们？** 压缩会同时改变 drafter 时间、候选 top-k 和各位置接受率，原成本曲线/先验不能直接沿用。先在独立校准输入上测每种配置，再冻结到包含中文/数学/代码的新测试集；比较完整 Pareto 取舍及端到端收益。仅做量化或缩词表已有先例，本项目尚未做该联合问题的新颖性检索，也没有取得联合优化结果。

复现入口：`scripts/e4_study.py`、`e4_client.py`、`e4_ours.py`、`e4_tasks.py`、`e4_process.py`、`e4_export.py`；补充控制为 `e4_warmup_control.py` / `e4_warmup_export.py`。执行文件哈希随结果保存，依赖外部 E3 target、TensorFold 安装、权重和冻结校准；仓库不包含权重、第三方源、二进制或原始私有日志。
