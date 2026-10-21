# E4：同机 TensorFold 与 DFlash2 成本预算对照

在任何新提示 GPU 输出产生前固定输入和方法。源码/权重身份沿用 E3 与固定的 MiaAI-Lab recipe `656879fd752749bbd11a6e1a065a326c00940911`，实际可执行文件和环境另记 SHA256，不根据镜像标签猜测后端。

使用同一 DGX Spark、NVIDIA FP4 checkpoint `b9ce18a6887e18c089fdea8509a33e63429a3300`、BF16 KV、text-only、单请求、temperature=0、显式 thinking 开关、相同输出上限和原生上下文位置规则。不改 GPU 时钟/功率，只记录实际状态。权重加载分时进行，不与其他模型并驻。

`scripts/e4_tasks.py` 固定六个新提示分别 thinking 开/关，共十二个自然结束用例，覆盖整数、JSON、两个可执行纯 Python 函数、散文和文字算术。另有两个 256-token capped 长输出用例；只有实际达到上限的样本才记作定长吞吐，提前 EOS 单列。自然任务达到上限按截断保存，不伪称完整回答。旧 E3 提示只做回归，不加入新的泛化汇总。

基线为实际 TensorFold Zig 原生 MTP（最大六草稿、累计置信度阈值 0.4）及其关闭草稿的 serial。候选为现有 Python target 的 serial、DFlash2 固定六/七草稿和原生 DeepInfer 动态零到七草稿。候选不叠加未测的学习策略/缓存/EOS 优化，成本曲线及条件先验冻结使用上轮独立校准结果，不用新测试调整控制器。

每模式四次计时，候选采用四阶循环 Latin square，基线两个模式交替。两个加载 session 分别使用重复编号 0/1、2/3，按 TensorFold A → 候选 B → 候选 B → TensorFold A 次序形成 ABBA，减轻单向热漂移；每 session 都重载模型。预热输入与测试不同。每次请求均冷 prompt：基线关闭 prompt/repeat cache；候选从新状态逐 token 处理 prompt，保留已知 prefill 数值边界，不冒称已经优化的 prefill。

分别报告模型内 decode、冷 prompt/首输出、请求完整时间、采样/控制与草稿/验证/提交成本、状态内存及实际输出。统一计时尽可能使用相同本机 loopback SSE 客户端；若路径或统计范围无法对齐，则明确分开，不能合成算法加速比。网络、加载、校准和 oracle 不混入 decode。

候选非 serial 输出、已提交 logits 和最终有效状态对照其 serial。基线原生 MTP 对照其 serial。跨 Zig/Python 输出不预设 bit-exact：比较 prompt IDs、输出 IDs、长度、停止原因和任务检查；有差异时同时保留其影响，不能单凭 token/s 宣称更强。代码检查使用受限纯函数 AST、隔离进程与超时；长代码截断只做前缀观测，不称正确程序。

先做 BF16/text-only 的上述核心对照；FP8/vision、多请求和长上下文为后续配置，不用未执行项目填充结果。若出现真实运行障碍，保留失败及修复记录，不以公开 README 数字替代本机结果。原始日志、第三方源、权重和二进制留私有区，仅公开实验代码、受控输出、配置与结果。

前置加载修正（新提示输出产生前）：原生 Zig 的 1536 context 加六草稿及 anchor 后，attention scratch 为 1543 行，需要未预编译的 dense NCH=4；已提供的 BF16 dense 变体为 NCH=2、稀疏变体为 NCH=5。默认 262144 context 加载成功。核心对照改为原生 context=1017、内部 capacity=1024，候选也用 capacity=1024，并检查 prompt+reply<=1017，使双方实际缓存几何一致且覆盖这批短上下文请求。不补造缺失内核，不将默认上下文的预检计入正式性能。此对照不能代表原生默认长上下文配置。

原生 API 暴露完整 prompt IDs、输出 token 数和 12 位 token SHA，但没有实现请求中的 return_token_ids。保持基线二进制不变，记录输出全文、推理文本、计数和摘要；跨路径可将候选完整 IDs 按上游逗号分隔 SHA 算法对照。摘要相同不称为直接读取原生完整 IDs，也不称全量 logits/state 检查；未能重建的原生 IDs 保持未获取。

补充基线配置归属：固定原生 `cuda_weights.zig` 嵌入默认 79,591 个草稿词表 ID；候选 DFlash2 使用完整 248,320 词表的 head。保留这一默认优化，不把整条路径的差异全部归因于预算或语言运行时。草稿网络、head 规模、特征捕获和图执行均属于实现路径差异，只有候选内部固定六/七与动态对照能更直接分离预算影响。

用户确认的后续方向：分别评估草稿 head 量化、缩减词表及二者组合；本轮不启用。当前 drafter 直接复用 target 的 BF16 输出矩阵，因此后续需另建仅供草稿使用的投影，不能原地改动 target。额外量化/子集副本可能增加总驻留，不能把投影读取量减少直接称为整机内存节省。收益以接受率、完整 decode/请求时间和总驻留共同判断，并用新的中文、数学/LaTeX、代码/JSON 输入检查覆盖；本轮英文提示不能替代这些验证。

事后方法审计（原始 336 次完成后）：原生首轮显著慢于后轮，而候选在各设置计时前运行了完整 oracle，存在预热不对称。保留全部原始证据，另行重载一次原生服务，每设置先完整 serial/MTP 预热，再各四次交替计时，共 28 预热+112 计时；使用原二进制、参数、输入和缓存设置，单独存档，复用已见输入，不称原计划 ABBA 或新测试集。详见 [报告](E4_MATCHED_REPORT.md)与[补充数据](../results/e4/2026-10-20-native-warmup-control.json)。
