# E13：常驻模型上的草稿缓存与接受率信号

状态：实现并验证，不改变默认路径，不重新加载模型权重。

两项改变分开测量：A 为 DFlash 历史投影缓存，B 为候选分数的接受率校准。B 的采集与路由观察不得混入 A 的正式计时；本轮不把新接受率模型与新专家成本模型同时上线。

## A：历史投影缓存

`CachedContextDraft` 共享原 Target embedding/head 和 DFlash 权重，保留完整 Q4 输出头及 BF16 算术。只缓存 immutable committed features 的 FC+norm 和五层 K（norm/RoPE 后）/V。当前 block 的 attention、卷积与输出每轮重算。缓存预分配25 MiB，每个请求显式 `begin(request)`，仅追加 accepted Target features；回退须重建，不能在请求间或 rejected 分支间复用。

先在常驻模型上，通过一个合成长 prompt 的真实 Target features，检查 contexts64/256/768，及+1/+4尾部追加。比较原路径全量计算的 FC、各层K/V、最终hidden和草稿路径。BF16 GEMM形状改变可能改变舍入，逐项报告 exact / relative L2；预注册诊断阈值为组件0.5%、hidden1%，不能把阈值当模型质量保证。检查同长度新请求清空、rewind拒绝。分别计时 cold-build、纯reuse、append4、original，各8次平衡顺序；warm reuse不是完整请求收益。

通过后，对新的固定任务进行 original/cached 完整 bounded generation A/B，缓存首轮构建和逐轮追加计入 decode。Prompt状态复制、oracle和观测不计入 decode；不声称SSE完整请求/TTFT。两模式使用相同原生 MTPVarlen 和冻结E6先验，保留各自runtime反馈，预算变化须报告。至少4次平衡重复，输出和每次结束的live state对照serial，另做逐committed-logit oracle。若仅微基准改善、整体无收益，如实记录并保持opt-in。

## B：接受率/成本的廉价信号

从原DFlash已经计算的候选 logits 与selector边分数提取：候选top1-top2差、沿选中路径的边top1-top2差、仅16候选中的归一化得分和所选候选rank。它们都是proxy，**不等同于真实全词表概率或校准接受率**。只在完整草稿之后可用，不能用于同一步pre-draft跳过决策。

行为为原dynamic预算，在另外的观测执行中核对信号提取不改变原草稿。Target原路径verify计时与真实路由另次观察分开；路由只作本轮成本标签，下一步才能使用。首次拒绝以后的草稿没有接受标签；EOS/输出cap以后的条目也不作为失败。

校准提示与测试提示分开，冻结后再运行；先比较depth-only平滑先验与depth+训练集median分桶的简单规则，报告held-out Brier/log-loss、实际观察深度、逐prompt结果和不确定性。不能在相同测试集上反复选信号后当独立验证；本轮预选selector-edge margin为主要信号。信号提取成本另记，Brier改善不等于吞吐改善。只有收益空间和开销支持时，下一轮再做单独在线预算A/B。

模型在任务之间保持驻留。实验模块只复用session对象，并用源SHA固定每次任务内容；本轮不得修改session原draft/target的dispatch或释放其权重。

## 正式生成对照前的形状修正

首轮 micro 在 context65 的 hidden相对差异2.013%，未过1%门槛。失败保留；没有放宽阈值。后续合成输入诊断确认 FC 投影对矩阵行数敏感，甚至64→65会改变旧前缀的BF16舍入，因此不能声称原torch路径具有逐行bit稳定性。

冻结修正：context<256走原路径；context≥256启用缓存，追加不足4行时pad到4行后投影。增加contexts384/512检查，组件/hidden门槛保持0.5%/1%。修正后的micro通过后，才排入原定12任务、4重复的完整bounded生成A/B；微基准属于筛选，不当独立性能结论。缓存连同pad缓冲约25.12 MiB。短上下文回退会限制加速覆盖率，应分别报告短/长任务。

## 根据用户长上下文要求追加的独立验证

初始12任务不足以代表千token输入。保持权重驻留，独立创建capacity2048的Target缓冲/状态和共享原DFlash权重的wrapper，只把该wrapper的max_context提高到2048，不修改原session。先检查contexts1024/1536/1984的缓存，以及2000行前缀后8行teacher-forced logits/features与serial一致、commit1/4/8的live state一致。通过后追加1024/1536/1920-token实际prompt的96-token bounded生成：每模式4次平衡重复，沿用同一oracle标准，包含cold build与append。原chat模板开头、最终完整问题和assistant prefix保留，裁剪重复背景的中段；这些是合成上下文长度实验，不是长文本任务质量评测。

2K内所有key仍在原DFlash 2048窗口内。4K及以上需要显式滑窗与绝对位置语义、Target容量和状态验证，不由此轮外推。

成本信号的二级离线筛选只使用验证宽度、context、上一实际step的union及**第一个候选**的edge margin。第一个候选在k>0时必有标签，但其分数在verify前已知；不能用observed深度数量或拒绝以后的标签作输入。固定ridge惩罚0.001，train-only归一化。单次观察的wall时延波动很大时，该回归仅保留诊断，不据此主张成本预测或在线收益。
