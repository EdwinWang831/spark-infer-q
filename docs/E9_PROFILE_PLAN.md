# E9：当前完整Q4动态decode时间分布

沿用E8 rows8默认、完整词表Q4、BF16 draft backbone/selector/KV及冻结E6预算先验，保留原有同步/断言/CPU返回，不同时优化。原E8源和数据不改。

选择已有三条输入：中文JSON `zh_json`、thinking算术 `thinking_math`、capped256长代码 `length_code`。每条先完整serial参考及四种E8路径的独立logits/state对照，再两次未采样控制、一次完整decode采样、两次未采样控制。共15次请求（12控制＋3采样），另15次完整预热/验证。输出IDs回归E7，所有请求结束后单独查最终有效状态。

Nsight Systems 2026.3.1预检：默认CUDA硬件追踪未捕获到kernel事件，切换`cuda-sw`后捕获成功。使用`--trace=cuda-sw,nvtx --sample=none --cpuctxsw=none`，通过CUDAProfilerStart/Stop只采集decode，排除prompt、加载、独立正确性与采集器启动/停止。原始nsys/sqlite仅留gitignored artifacts，不提交。

粗分段：预算选择、完整draft、target验证、接受决策/预测CPU返回、状态提交、context拼接、预算反馈、流式输出与末尾同步。细分draft backbone/head-topk/selector（含assert和tolist），target stage/embed/48层计算/五层特征捕获/head-finish/同步。NVTX只在采样请求激活，非采样target和selector直接调用原E8实现。

同时统计CPU CUDA API时间、同步调用、launch调用、内存传输和GPU kernel区间并集，保留每条输入与宽度分布、采样/邻近控制耗时比。嵌套区间要去重；CPU调用与GPU执行可能重叠，不能相加；同步API时间不是独立的可消除成本。GPU kernel空档包含主机调度/驱动/运行库等，不能一律称为Python。当前容器perf权限不支持CPU栈采样，不改系统权限；不能精确归因的部分明确保留。

首次采样是诊断，不作无采样性能提升或原生TensorFold排名宣称。报告instrumentation和动态宽度可能被采样扰动的限制，依据实际瓶颈提出下一项受控优化。
