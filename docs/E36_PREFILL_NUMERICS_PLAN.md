# E36：专用 prefill 的同输入数值诊断

2026-10-23，执行前冻结。范围为常驻同一 NVFP4 target，不改 K4 工作区、服务默认、checkpoint 或全局算子，不重载模型。承接尚未执行的 `PREFILL_REFERENCE_PLAN.md`，本轮只做其中的逐层定位与局部参考，**不冒称完整 FP32 模型参考或任务质量结果**。

## 输入与对照

固定四个 teacher-forced 提示前缀：英文解释64 token、中文解释64 token、英文代码512 token、中英资料512 token；不足长度时按冻结文本重复后取前缀。保留 [100,101,102,103] 为历史回归。文本与 token SHA、长度、来源在执行前 preflight 封存。它们是受控合成前缀，不是独立质量评测。

三条路径：serial rows1、现有 decode 算术 rows8、原生 `prefill=True` 整段。64 token 与历史4 token 核验 rows8/serial 的每层残差和末行 logits；512 token 以 rows8 为既有数值控制，不声称这两条长输入已另跑逐token oracle。每层从 pending 分支完成写回的副本取残差，另记每层专家集合；不得把未写回的 `b.h` 当层输出。

## 同输入局部参考

优先 layer 0/6 的 GDN 投影与 recurrence。取 rows8 轨迹的相同 BF16 attention 输入，对两种 MXFP8 投影执行路径作局部比较：按存储 e4m3/e8m0 解码的权重，TF32关闭，PyTorch FP32矩阵乘，不加中间BF16舍入。参考输出保留FP32，同时另记最终BF16舍入后的相等情况。格式解码共享同一权重实现，须披露。

递归隔离使用同一个真实投影 P、相同 conv 与初始 recurrent state。用共同 front 产生 q/k/v/g/beta，再比较 decode recurrence 与 prefill recurrence 对逐步 FP32 参考的输出与终态；额外 FP64 算术仅作为局部参考稳定性控制。共同 front 自身的误差不在这个参考的仲裁范围。若只能对比 gated BF16 输出，明确披露末端舍入，不把它误作未舍入 recurrence y。

逐层误差定位与同输入算子比较分别报告；禁止用两条已经分歧的输入直接判定某算子精度。包括 relative L2、max abs、逐行 top1/路由一致性（适用时）、首个不相等层。不得由单层误差较小推断整模更准确。

## 验收与停止

所有执行源/输入和外部依赖封存，经共享 resident 串行队列。每轮检查同一模型对象、load_count=1、未修改原源码，最终READY与资源回收；保留失败及异常。先做环境/输入 preflight，再封存新版本运行诊断。显存按需保留两层输入和逐层残差，避免整模FP32解量化；内存不足缩小采集批次，不重载共享模型。

本轮只有诊断 PASS，不设吞吐晋升。即使原生 prefill 局部误差更小，也必须后续验证独立整模参考、prefill→decode状态交接、DFlash特征与真实完整请求；未通过前不切换默认。没有新的4K支持、随机采样、联合DFlash或TensorFold速度排名。不得把旧短输入109.23 tok/s和他人4K自报值换算为本轮预期加速倍数。
