# E30：在冻结 W8＋线性预算＋Graph 基线上重新剖析 target

先核对历史：E10已扫描专家warp/寄存器预取深度（独立复核约+0.08%，未晋升）；E11已扫描BF16块形状/warp/stage及MXFP8归并方式（完整decode约+0.62%，未晋升）；E29已接入完整target前向CUDA Graph（相同轨迹+1.28%）。不重复这些扫描，不把旧Nsight比例当当前路径的比例。

复用同一常驻worker，保持原始target、E28五层MLP的W8包算法和二进制、E17 NumPy线性权重、E29冻结W8费用及完整target Graph不变。不开其他训练或量化分支，不重载模型，不修改既有封存源码、全局算子dispatch或系统profiler权限。

选择E29校准清单中的英文问答、英文代码、英文长输入和中文长输入四条，仅作热点诊断，不称独立泛化测试。每条观察前8个动态decode轮次，任务原始cap仍用于预算特征，计时token按实际emitted统计。预热全部16张target图；预热、初态复制、串行oracle和正确性检查在decode计时外。每条normal/profile/normal，normal与profile共用同样的图和决策；所有token、每个提交logit/特征和最终live state核对串行target。

优先尝试PyTorch CUPTI profiler在已运行进程内捕获CPU/CUDA活动，原始trace只保留在忽略目录。profile的扰动必须单列，不能当正常decode吞吐；如CUDA Graph未产生可用的GPU事件，则明确记录能力缺失，改用单独的Graph外部CUDA event分段，而不重启模型来配合旧式Nsight启动流程。没有硬件DRAM/SM计数器时不宣称已证明带宽瓶颈。

根据当前GPU热点、实际调用形状及既有工作选择一个候选。若候选只是旧参数组合或可兑现空间很小，就报告限制，不自动扩展多个优化方向。任何实际改动先保持数值和状态正确，使用相同前缀及正常Graph计时验证；进入完整decode对照前另行封存候选、输入和预算。

剖析完成后选择单一候选：BF16完整词表输出头在当前target验证kernel累计时间中约10.5%，旧E11改变过块/warp/stage，但没有测试NK→KN权重预排布。新候选仅转置存储和Triton读取方向，保留实际BF16值、BK64、SK1、BM16、BN64、4warp、3stage和最终输出精度，不改变target量化。先用四条诊断root的前8轮真实head输入核对全部logits位级一致，再以6轮交替、每轮10次CUDA-event区间测量。packing/clone/check不计入微计时，额外完整BF16副本及准备成本另报。事前门槛：全部位级一致，head耗时至少降低10%，且按root重采样增益下界为正，才进入完整decode；不满足则结束本轮候选，不追加参数扫描。
