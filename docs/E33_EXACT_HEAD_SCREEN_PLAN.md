# E33：真实 hidden 上的低位宽筛选与原 BF16 输出头精算

2026-10-23 UTC。执行原有 `EXACT_HEAD_SCREEN_PLAN.md` 的首个 GPU 可行性实验，不将旧合成结果当成真实候选数。冻结 E28 W8 草稿、E17 NumPy 线性预算、E29 W8费用和完整原target Graph。复用同一resident worker/权重，不改旧seal、全局dispatch或默认执行器。

仅为筛选建立一个 **逐输出通道对称INT8 RTN、FP32 scale** 的辅助头，BF16 hidden不量化；不是用INT8分数决定最终token，也不是替换target权重。与旧提议group32 affine格式不同，本轮事前固定单一格式，不扫描配方。近似核固定与原BF16头相同的BM16/BN64/BK64、4warp/3stage、SK1，以BF16整数权重operand和FP32累加，最后乘逐行scale。辅助payload约641MB；原1.27GB BF16头保留，准备耗时和实际内存另记。

误差界覆盖 `||W−sQ||₂ ||h||₂`、原与近似路径的FP32累加、近似最终scale乘法和原BF16输出舍入。两个逐词范数与hidden范数用FP64计算并向上膨胀/取nextafter；区间端点向外舍入。采用保守的 `gamma_(2K+1)(u=2^-23)`（每条路径）和BF16相对舍入界2^-8。静态权重/scale与动态hidden必须有限，非零幅值至少2^-48且最大不超过2^16，保证当前K2560下中间项不溢出且无非零FP32 subnormal；不满足则回退完整原BF16头。这是基于声明算术假设的保守界，实测不替代GPU形式化证明；一例越界即停止精确性和性能晋升。

保留所有 `upper >= max(lower)` 的词，取所有验证行候选的有序并集。并集最多2048词，否则回退完整头。子集核仅gather这些原BF16权重行，保留原BK64 dot链、SK1、输出BF16和最小ID并列规则；与全头对应logits逐位核对。非有限数、无候选、输入范围不符、范数/界无效均回退，不猜测结果。

先用四条E30已用校准root、各前8轮，32份真实head输入：完整词表逐元素检查误差/界比值、原argmax是否保留、候选子集logits逐位一致、最终argmax一致。覆盖1～8行的合成hidden、强制overflow/NaN/tie fallback等独立功能检查；真实前缀继续走冻结原Graph，候选只在隔离回放运行。

近似投影、动态检查、区间和mask采用按行数缓存的私有CUDA Graph；候选压缩及动态BF16子集仍在Graph外，必要同步计入总成本。先测Q8投影本身作为完整流程的成本下界。如果它本身已不快于全BF16头，直接停止，不展开长测。否则成对交替测 **完整已实现流程**（近似分数、动态检查、界、候选压缩/同步、BF16子集或回退、argmax）与原完整头+argmax。每份真实输入各6次配对，CUDA同步的host wall计时包含动态筛选开销；权重准备、Graph capture、正确性oracle、磁盘写入不计入。另报Q8-only及BF16-only CUDA event诊断，不能拿它们冒充完整流程收益。

事前组件门槛：全部检查通过；真实完整筛选头合计耗时至少减少10%，按4个root bootstrap速度收益95%下界为正、无root超过2%退化，才另封存正常完整decode A/B。不能将组件收益换算成实测decode TPS；不通过即停止本格式，不扫描另一个量化或kernel参数配方。仅适用于greedy，随机采样继续用完整头。
