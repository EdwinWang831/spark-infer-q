# E36 后续诊断：第 0 层数值分歧的因子定位

在 `242-e36-diagnose-v2.json` 已完成后追加，**不是事前独立测试集**。五条受控前缀全部从第0层出现残差差异；两层同输入投影/递归局部参考不足以解释完整路径，因此在同一输入上做第0层诊断，不扩大性能主张。

只执行layer0，逐项采集GDN前HC读出、投影P、GDN输出、输出投影branch、MoE输入、router/picks/weights、专家slot和已写回残差；用rows8作为同路径数值控制。专用prefill下穷举三个因子的八种组合：A把forward中的MXFP8投影改为decode路径，B把prefill recurrence改为decode chain（每次独立scratch、正确parity/conv推进），C把MoE slots改为F32（同时切换shared-expert路径，明确是联动策略）。不声称这三种改动互相独立地对应三个算子。

先确认全原生因子的层0残差与上一阶段相同；再记录每个组合的逐组件误差、精确相等、路由集合一致率。数值消融用于定位必要/共同因素，不是性能筛选，未覆盖其余47层，也不验证完整prefill→decode交接或任务质量。所有作用域为实例私有clone，不改变原forward或模型权重，不重载resident。
