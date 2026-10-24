# E38 组件计时口径冻结

在组件作业提交之前明确：完整 MoE 以 CUDA-event 包围完整 CUDA Graph replay 计时，包含 router/select/plan、专家和共享分支及写回；称为设备区间成本，不将其称为 CPU wall。原计划“完整 MoE wall”应按此设备区间定义读取。warm每区间3次取每次成本，128MiB扰动每区间1次；扰动在timer外。四条输入×三层×两种缓存条件×六轮×两模式，共288区间、576次计时MoE调用。Graph capture、JIT、预热、正确性和hash检查均在外。每份输入每模式六次的中位数相加判定5%组件门槛，同时检查逐轮和逐例；若通过，正式prompt仍使用计划中的主机wall完整边界，不外推组件结果。

人工路由内核预检为40项：十种行数×连续/stride×F32/BF16 down，均完整比较所有act/y及guard、Graph两次重放。33行stride/F32额外使用独立caller stream。此预检不经过真实router，刻意使每个选定专家覆盖同一行数以测16/32边界；完整真实MoE对照另覆盖全部router/pick/wts/act/y。短输入策略与原版完全相同，只在人工内核预检中强制32行计划，以检验所有尾部。
