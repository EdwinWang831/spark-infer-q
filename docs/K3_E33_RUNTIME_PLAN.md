# E33 完整优化配置迁入原生动态运行时

2026-10-23，基于 K3 Graph（PR #16）。本次一次性保留 E33 的辅助筛选头＋原BF16精算、Target Graph、E28 W8草稿MLP/Q4草稿头、E17 NumPy线性预测及E29 W8费用。固定草稿模式不恢复。独立 `--profile e33 --execution graph --mode adaptive` 入口；冻结artifact/kernel身份，不训练、不改原权重、不覆盖既有resident源。

## 接口与计时

- 筛选路径返回精确greedy预测与DFlash特征；不能把hidden/候选logits冒充完整logits。诊断时额外执行完整原头，检查子集值、argmax和serial committed logits；timed路径不执行诊断完整头。
- 新配置在decode prepare时生成当前草稿、预测并选择验证宽度。原生NEXTN仅预留候选容量，通过公开ZOS metadata hook最终确定q；schedule/process计数逐轮配对。初始草稿在首token输出后生成，完整计入decode，与E33原循环一致。旧baseline配置保留原位置。
- 按E33当前单请求目标函数choose_rows选择宽度，保留全部7个累计概率与冻结paid_ms/short-long费用；逐步核对context/rates/width/emitted/accepted/observed，拒绝非匹配预算的速度归因。

## 验收与测速

1. 本地回归；真实C++ CPU deferred-draft契约：prefill不出草稿、当前草稿/最终q/输出一致、取消/长度与计数回收。真实GPU单独检查迁入头1～8行、cap/全零并列/NaN回退，原权重保持不变。
2. 常驻模型三条短提示的整合GPU检查，含完整serial输出/logits/features/live state、EOS/cap、取消三阶段及复用；准备16张backbone图和8种筛选宽度，请求中不capture；SSE/取消使用完整配置。
3. E33同一 `e28_test_tasks.json` 八个工程root、原output_cap/自然EOS、每组四次AB/BA，共64次计时。同期原E33 screened研究路径与新native runtime两组均使用上述完全相同优化配置/2048物理容量；同一首token后的decode墙钟（包含首草稿/预算/验证/筛选/同步/提交/原生调度/事件），prefill/capture/JIT/检查单列在计时外。
4. 每条先独立serial oracle和两组完整正确性预热；timed运行后仍核对完整输出/最终state/全部决策轨迹，无新增capture或subset JIT。复算total tokens/total seconds、8-root bootstrap与逐重复结果，并报告全请求/TTFT准备边界，不能把历史48.51直接作同期对照。

所有失败保留；修复后新封存重跑。worker使用串行queue、保持load_count=1，收尾检查group/env/lease与allocated回归READY。不宣称最新TensorFold排名、多请求batch、随机采样、通用性能或全异步；本次结果是已迁入全部上述优化的新运行时实测值。
