# E37：prefill 时间分布与 GDN scratch 复用

2026-10-24，事前冻结。承接E36，不修改其执行源或另一个K4工作区；同一resident模型、相同输入、A/B/C算术、完整五层特征、State初始化、commit和末token准备。仅研究prompt本体，不宣称完整DFlash服务/HTTP/decode收益。

输入从已封存246作业读取，核对作业SHA256 `542d558988acb4c6a91c8b6dc8a73dbf567004036cf3ec973dadb16c0de19853`。依次code512、english64、mixed512、new_json128，覆盖大→小→大→中缓冲复用。全部末行logits+完整特征hash必须与E36逐例一致，fresh/pool全部有效state逐位一致；每例再从独立state运行8步原decode，逐步核对logits/features/state。

## 单变量对照

fresh为原E36构造器，每GDN层新建scratch。pool仅替换私有forward的prefill chain：同一单stream、单owner线程复用一个预备512行scratch，按实际rows取预备view。36层已在forward内提交state，后续prefill commit不读旧scratch；不改state、权重、算术或读取顺序。不支持并发stream、prefix cut或超预备容量，明确拒绝。

两模式工作缓冲与pool在计时外准备；每例每模式一次预热，再6轮AB/BA交替，共48次prompt wall；包含State初始化、stage、前向、全部特征拼接、commit、预测和同步。输出hash/最终state检查在timer外。吞吐用总token/总wall，另报逐例和逐轮变化。候选资源复用门槛：全部正确性通过、总wall至少减少2%、至少5/6轮更快且每例不慢超过2%，否则保持E36默认，不为微小噪声扩大长测。

## 时间分布

在另外两次512-token诊断中，用同一fresh基线采集：实例私有嵌套函数标记、CUDA event和主机enqueue区间；嵌套子区间从父段排除，不双计投影与GDN/attention。分组State初始化、host stage、embedding、HC、forward._mm投影、GDN其余工作、attention其余工作、MoE、PLE、特征、head、commit、末token与control。

GPU event区间包含该段的设备排队与主机提交间隙，不等于纯kernel时间。另独立CUPTI trace只读e37.prompt范围内真正cat=kernel事件，按kernel家族汇总累计时长与活动区间并集；不统计annotation父级，不把累计时长或活动率叫带宽/SM利用率。profiler/event测量与正式wall分开，披露扰动，不用剖析墙钟替代速度。

只有时间分布明确后，选择主要重核或主机开销作为后续方向，不预设GDN/Graph/scratch是最大瓶颈，不承诺新增倍数。所有源、输入、结果、trace封存，队列串行，同对象/load_count=1、源恢复和独立收尾内存检查；允许共享worker在本作业结束后继续其他任务。
