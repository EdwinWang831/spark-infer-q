# E35：共享专家与路由专家的独立分支重叠

固定E33的W8草稿、Q4草稿头、NumPy线性预算、E29费用、筛选头及原target算术。源码检查确认：现有grouped专家已复用同专家各行权重，不能把增加grouping当新优化；本候选不声称减少权重字节，而是测试MoE两条独立分支能否共享GPU执行时间。

原MoE顺序执行router/select、routed gate/up、shared gate/up、routed down、shared down。候选以一个持有的辅助stream执行shared gate/up和down，主stream保留router/select和routed gate/up/down，输入就绪后fork，结果合并前join。复制回原slot，保留BF16/FP32输出、路由/权重、逐slot最终合并次序。不得修改全局dispatch或其他工作区K4源。

先以原checkpoint的第0/23/47层做1～8行、连续/带stride输入、原路由、Graph重放及独立调用stream检查。随后四条E33旧校准root各取前2轮、上述3层，共24份真实输入：比较完整MoE串行/双stream Graph，热复用与128MiB缓存扰动各6次配对；检查所有activation、slot输出、路由ID/权重及8次诊断前向的完整logits、特征和提交live state。候选至少减少5%缓存扰动MoE时间，root bootstrap95%下界为正，任一root不退化超过2%，才进入正常生成；检查失败则停止。

额外在四条root的首轮第0层对两种Graph做独立CUPTI诊断，保留真正kernel事件/stream时间线，确认是否发生重叠，剖析时间不混入速度汇总。事件计时含Graph内依赖，不含capture、检查、扰动本身；缓存扰动不视为DRAM驱逐证明。组件收益不外推decode。

若过门槛，另一8条旧工程root各4次完整decode，保持预算/接受轨迹一致，计入所有copy/fence/循环。至少+1%、root95%下界为正、四轮均正才晋升；否则保留负结果。复用同一常驻模型、不下载、不重载，最终核对READY/源seal/内存，公开脱敏证据。
