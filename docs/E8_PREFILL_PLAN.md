# E8：沿用decode算术的分块prompt验证

2026-10-21。E7完成归档后单独建立实验目录，不改E7冻结执行源、模型权重或现有默认路径。

首轮使用`WideTarget`现有`prefill=False`、BF16 KV、capacity1024，比较每块1/2/4/8行。输入为E7十条prompt及第一条prompt的1..17 token前缀，共27条；每个2/4/8宽度对照逐token执行，在每个块边界核对完整逐行logits、五层draft特征、有效GDN/QSA/PLE状态，再核对八步greedy target-only延续的logits、特征和状态。预定81个对照，任何差异保留并阻止本轮性能结论。

正确性全部通过后，各任务/宽度完整预热一次，再各计时四次，循环改变宽度顺序，共160次。同步主机计时包含状态初始化、逐块feature capture和拼接，排除tokenization、加载、oracle和延续。按每任务中位数求和、按实际prompt tokens加权；保留逐次数据与波动。首次完整预热不计正式成绩。

首轮是prompt本体实验，不报告wire TTFT、不据此宣称相对E7原生端到端反超。通过后须把分块路径接入完整Q4动态生成，检查输出、已提交logits/状态，并重新配对测量请求TTFT、完整请求和decode。保持E6量化/预算先验固定；INT8、CUDA Graph、独立prefill算术留给后续受控对照。
