# E52：小窗口 target 专家流水筛查

2026-10-26。继续E51之后的target执行效率研究。保持BF16激活、现有NVFP4权重、分块MMA/scale-FMA次序、E17/E29动态预算、W8/Q4草稿与Graph不变。复用同一worker46235/load_count1及已有E43流水二进制，不下载/重载权重、清缓存、提高进程限额或并行跑GPU任务。

E43只测了64～1024行prefill；此轮仅问它在1～8行真正decode plan上的表现。没有重新扫描warp/转换/几何，也不把prefill的失败改标成功。E43历史32MiB未归属分配问题保留；本轮使用当前77,393,408,000-byte稳定基线独立审计。

- 组件边界：宽度1～8、连续/stride、F32/BF16 down、共享/不同/混合专家路由、共享槽与首尾哨兵、Graph和额外caller stream，逐位一致。
- 真实输入：四个封存中英/代码工程prompt，每条前两轮动态验证，第0/23/47层完整MoE，24例。每例router/picks/wts/act/y全部精确，诊断完整logits/五处features/提交有效状态匹配原Graph。采样不更改原全局dispatch。
- 性能：完整MoE两臂各自CUDA Graph，warm每event5次、128MiB扰动每event1次，每条件六次AB/BA；捕获、扰动、检查均在event外。扰动不保证实际权重完全逐出cache，不代表整模型缓存状态。
- 晋升门槛：warm和扰动两组完整MoE中位数之和均至少减少5%，按prompt聚类bootstrap下界为正，任何root不退化超过2%。不通过则保留负结果，不做完整decode速度声明。通过才验证替换全部层的完整原生动态请求。

报告必须区分数值通过、组件收益和完整请求收益；不得用既有E43指令减少、寄存器降低或E51单组异常速度替代本轮证据。
