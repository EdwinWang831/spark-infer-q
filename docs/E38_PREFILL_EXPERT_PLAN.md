# E38：prefill 专家多行权重复用

2026-10-24，事前冻结。承接 E36 算术和 E37 定位；不修改旧 overlay、原模型、默认服务或 K4 工作区。所有 GPU 工作经常驻 worker 队列串行执行，不重载模型。

先对 E37 同四条封存输入（code512、english64、mixed512、JSON128）逐层采集每专家行数、16/32 行分块尾部和 gate/up、down 的独立 CUDA-event 诊断。只观察真实路由，不将观测计时称为正式收益；输出 hash 必须与 E36 一致。

候选只研究相邻两组 16 行共享一次 packed NVFP4 权重加载、FP4 解码和 scale 转换。原实现已聚合同专家行，不能将 grouping 当作新增。32 行 plan 中不超过 16 行的 item 保留原单组实现，超过 16 行才走双组累加器；两种 item 的内核分别启动，过滤互斥。每行仍按原 K-block 顺序 MMA → FP32 scale FMA → per-expert scale → 原 SwiGLU 或 F32 down。输入 BF16、weight pack、最终 slot/reduction 次序均不变。不得以修改 scale、FP4 激活、BF16 down 或 split-K 近似代替本候选。

若真实路由显示 32 行复用覆盖太低，则记录事实并停止该候选。否则独立编译一个 SM121 扩展，冻结生成源、公共头与 binary 指纹。组件对照覆盖真实 prefill 输入的层0/23/47与四条输入；每份比较完整 act/y/router/pick/wts，含 guard、stride、不同 caller stream 和 Graph 重放。另用随机路由负载覆盖 1/15/16/17/31/32/33/64/128/512 行。组件正式计时六轮 AB/BA，原版/候选完整 MoE（含路由和计划、共享专家）均计时；128MiB 缓存扰动在 timer 外，另报热复用。保持真实共享专家算术不变。

只在组件精确通过、扰动条件总完整 MoE wall 至少减少5%、至少5/6轮更快且任何输入不慢超过2%时进入完整 prompt。校准为三个真实层的四份输入，不以组件结果宣称完整 prompt、decode 或 HTTP 加速；不无限扩大微小噪声试验。

完整 prompt 复用 E37 四条输入顺序，每例六轮 AB/BA，48次 wall。State初始化、stage、48层、全部五层特征、commit、末token与同步在 timer 内；buffer/JIT/预热与完整 hash/state 对照在外。fresh E36 作为唯一对照，不沿用失败的 scratch pool。每模式7296 prompt tokens，以总token/总wall计率。逐例完整末行logits、1216行拼接特征、有效state及每例8步原始decode全部精确，再对每次计时检查hash/最终state。采纳门槛：总prompt wall至少减少3%，至少5/6轮更快，任一输入不慢超过2%；否则不晋升。

测量、构建和收尾均封存为独立新stage/job。收尾单独审计源恢复、同target/draft对象、load_count=1和暂存显存释放。记录失败而不覆盖已执行源；公开结果去除token IDs、对象地址、私有路径和原始模型数据。无随机采样、4K+、服务收益或默认切换承诺。
