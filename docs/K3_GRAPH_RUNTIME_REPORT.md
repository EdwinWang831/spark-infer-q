# K3：动态预算运行时接入 Target CUDA Graph

2026-10-23，`PASS_K3_GRAPH_RUNTIME`。在 K2 的 E17 本轮动态预算上增加显式 `--execution graph`，固定1/3/7草稿模式不恢复。运行时现在可复用已预热的 Target CUDA Graph，管理请求独占workspace、取消/结束/reset和eager回退。[方案](K3_GRAPH_RUNTIME_PLAN.md)、[用法](K3_GRAPH_RUNTIME_USAGE.md)、[脱敏证据](../results/k3/2026-10-23-graph-runtime.json)。默认执行仍为eager，本轮完成K3的Graph接线部分，保留host完成同步，不宣称已完成全异步或同步精简。

## 接线与保留的行为

Graph的数学运算与五处DFlash特征提取沿用[E29](E29_TARGET_VERIFICATION_REPORT.md)。没有替换target内核、权重或全词表输出头；没有接E33筛选头、W8草稿或NumPy预测器。E17 Python线性头、特征、冻结成本及BudgetProbe保持K2配置，每轮仍先出草稿/预测，再动态选择1～8行。

Graph只替换decode compute：stage和commit仍用原TensorFold路径，prompt保持E8 eager。按窗口1～8行和GDN两种奇偶捕获16张图。服务启动预热后才READY，普通请求不capture；捕获时attention launch上界固定为现有1024物理容量。缺失bucket、不一致奇偶、state version变化时明确eager回退并计数；外来或未lease state拒绝。GPU计算仍同步单请求，不是DeepSeek原生Worker或跨请求batch。

workspace只允许一个请求lease，完成/取消先完成fence再释放，下一请求在同地址reset。图可以复用，上一请求的context和原生generation不可复用。engine公开实际capture/replay/fallback与预热计数，每次step更新；最后先关闭engine再关闭Graph owner。调用方须独占底层eager target/buffers，不能并发借用。

## 正确性与生命周期

- 本地207项：195通过、12因依赖缺失跳过。新增workspace单owner、foreign/released state拒绝、reset复用、活跃请求禁止capture/close、关闭后拒绝执行等契约；K1/K2回归通过。CLI的none+graph在加载前拒绝，固定模式仍不可用。
- Graph底层probe：**40个宽度/奇偶/接受前缀组合**，每次提交后继续生成；另测试未捕获width2的eager回退。合计 **244行完整logits、244行完整DFlash特征、81次有效状态比较精确一致**。主Graph始终16次capture/80次replay，回退目标仅2张图且没有临时capture。
- 真实DeepInfer Scheduler：**18组、838行完整committed logits、838行完整committed DFlash特征、203次有效state比较**全部精确一致。包括lagged/confidence/adaptive各三个输入，oracle-only全接受/首中末拒绝/EOS，adaptive长度1，以及waiting/prepared prompt/prepared decode取消后同workspace复用。oracle草稿只用于状态诊断。
- E17模式在三个输入上逐轮验证宽度与独立旧循环完全一致。此suite固定16张图、实际重放123次、零自然回退；不把单纯具备CUDAGraph对象当作完成接入。
- 在线另做30次完整输出/最终state核对（6预热+24计时），12对eager/graph完整验证宽度序列精确相同。真实loopback SSE和首输出后的显式取消运行在Graph路径，终止/DONE及lease回收通过。取消测试采用受控step边界，未覆盖任意异步竞态。
- 全部作业保持同一worker/load_count=1，不重载模型、不覆写冻结研究源。最终READY，Gloo group释放、NEXTN环境/缓存恢复，gc后allocated回到77,594,734,592 bytes；不调用empty_cache。

## 相同预算轨迹下的小样本计时

中文缓存解释、Python函数、JSON三个提示，输出上限32/48/48，两个执行模式各四次，共24次计时。各模式相同输出，每种 **412个decode token**。模式按重复交替AB/BA，完整请求/TTFT列是12个请求之和。

| Target执行 | decode token/s | 完整请求耗时之和 | TTFT之和 |
| --- | ---: | ---: | ---: |
| eager | 44.264 | 12.539 s | 3.231 s |
| graph | 44.778 | 12.350 s | 3.149 s |

Graph的decode点估计 **+1.160%**，完整请求耗时变化 **-1.501%**。这是同一个K2动态运行时、同一E17概率和预算、同样验证宽度下的执行差异。只有三个受控输入/四次重复，不建立统计显著性、广泛收益或默认晋升；不能与E29/E33不同任务的TPS横比或相乘。

计时从submit前到终止step后，包含tokenize、prompt、初始草稿及实际decode；decode从首输出事件到终止。模型/provider/native context构造、预热和oracle检查在计时外。初始草稿仍进入TTFT。Graph capture也在计时外，准备成本单列；没有把阶段前移、减少验证行数或E33头收益当作Graph加速。

## 准备成本与剩余边界

probe记录Graph capture准备段 **5.616s**，在线job为 **5.834s**。`graph_warmup_ms`段开始于static state/一token eager初始化/seed复制之后，不能当作完整冷启动时间；服务实际在全部准备完成后READY。probe的allocated前后快照增加 **253.55MiB**，包含静态state和捕获输出等，不是峰值或通用Graph独占开销。

本阶段保留Target.run/commit完成fence、CPU greedy读取和草稿特征传回等同步。减少`.tolist()`/`.cpu()`、GPU预算/采样、异步流水和联合GPU batch仍待后续工作。逻辑请求容量256，物理Target1024；未做长上下文、多请求同时replay、随机采样或新模型泛化验证。

研究E29已经验证Graph本身，此次新增的是运行时owner/lease/reset、动态计划的replay、取消/复用和真实HTTP边界。E33输出头研究的收益单独保留，未在K3混入。

## 复现

```bash
python3 scripts/k3_export.py results/raw/k3 results/k3/2026-10-23-graph-runtime.json
python3 -m unittest discover -s tests -q
```

四个封存作业分别probe/native GPU/online/resident audit，全部PASS；原始归档SHA256 `37372fe971f2fe670a36517159a631713e72bd469aefa4489775f40568046a9c`。公开导出校验最终源码SHA、native/预算/特征/state、固定capture数、逐轮预算一致及同worker/READY。probe封存在运行时live capability更新前，Graph本身SHA未变化，最终GPU/在线版本重复验证。原始记录位于Git忽略目录，公开证据不含提示词、权重或私有路径。独立CLI没有为验证重新加载模型，真实HTTP以resident session完成。
