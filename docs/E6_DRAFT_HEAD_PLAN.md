# E6：DFlash2 草稿输出头成本与压缩

2026-10-21 固定设计，产生本轮新提示 GPU 输出前保存。只更换草稿投影，不改 target、embedding、五层 DFlash2 backbone、selector 或原始 DeepInfer MTPVarlen。与 E5 target-only 原生 Scheduler 桥接分开，使用 E3/E4 已验证的单请求 speculative 执行循环，避免混合归因。

四配置：共享完整 248,320-row BF16 target head；原生固定 `draft_vocab.txt` ID 集合并入所有 EOS 的 BF16 独立副本；完整词表 group-32 affine Q4；同一子词表 Q4。子词表不从新任务输出学习，不试验多个大小后挑最佳。缩词表 top-k 索引必须映射回全局 token ID，再调用原 selector。Q4 复用固定 TensorFold 实现与外部 CUDA 源，BF16 激活不变；先做独立解包/FP32 数学检查、尾列和非标准 stream 检查。权重/源码/二进制不进仓库。

先拆测真实上下文上的 backbone、head+top-k、selector/host 三部分，并比较完整 proposer。所有配置都有完整逐用例预热；投影小基准按配置轮换。BF16-full 适配器与原 proposer 的 hidden、top-k/提案严格回归。Q4 相对 BF16 的数值/提案变化按实测保存，不要求相同。

两条预先固定校准提示独立于八条新测试文本。目标行数 1..8 的 verify+commit 曲线共同测量；每个 head 单独测 draft 成本、校准各深度接受率，随后冻结。每 head 同时比较 fixed7 与原生动态 0..7 草稿；四次重复，head 采用四阶循环位置平衡，每个 head 内两种预算按重复/任务交替先后，共 8×4×2×4=256 次计时。先做每模式独立 logits/state oracle；每次计时检查最终输出和有效状态，排除这些检查、prompt 与初始 state clone 后计算完整 decode 耗时。另在前两条测试上做各 head fixed7 两次冷 prompt+decode 的 executor 内总耗时（16 次），与 API/wire TTFT 区分。

新任务覆盖中文解释、中文 JSON、数学 LaTeX、两个可执行纯 Python 函数、英文算术、英文说明、thinking 算术。自然 EOS 与达到上限分别报告，不挑选成功/快任务。代码与整数/JSON 有限定答案检查；散文/LaTeX 不设语义质量得分，截断代码不称完整正确程序。完整输出、已提交 logits、GDN/QSA 状态必须对照同路径 serial；若出现 target row 数值不一致，保留并停止相应性能结论，不能算作草稿头收益。

全部 head 配置同驻留便于轮换；报告每配置额外 payload、实际共同驻留 allocated/reserved 与峰值，不能将投影压缩比例称整机省内存：原 target BF16 head 必须保留。target head 全量 SHA256 前后不变。速度按每任务耗时中位数之和加权实际 decode token，不平均 token/s。单台 Spark、BF16 KV、单请求 greedy、短上下文、不改时钟/功率；保留已有 prefill 与 drafter BF16 fidelity 边界，不宣称全引擎/随机采样/通用质量或算法新颖性。

实施补充（产生本轮新任务输出前）：按使用者要求，压缩接口与这台机器的研究驱动分离。库默认 dense/BF16，Q4 提供者显式注入，检查设备与形状；不支持时默认报错，只允许使用者显式启用带提示、可观察实际精度的 BF16 回退。不自动将 Qwen 固定词表推广给其他 tokenizer。候选重合、提案一致与接受率用于量化 fidelity 诊断；最终逐 token 等价只针对当前 NVFP4 target 的同路径 greedy 基线，并非整模 BF16 等价。Q8/混合精度属于 Q4 诊断后的独立候选，本轮不混入四配置统计。接口和复现边界见 [DRAFT_HEAD_USAGE.md](DRAFT_HEAD_USAGE.md)。
