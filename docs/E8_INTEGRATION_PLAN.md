# E8完整生成集成：预先冻结的方法

在已有81组完整prompt正确性和160次prompt本体计时后，把分块处理提取为`prompt_state`，接入新的完整Q4 SSE执行器`e8_ours.py`，默认8行。旧E7源、历史数据和其他服务不改。

保持相同target、Q4完整词表草稿头、BF16 backbone/selector/KV、E6先验与成本、E7动态decode/反馈机制。十条已见E7输入、thinking/输出上限不变；不混入INT8、Graph或新预算调参。

单次独立加载内，每条任务先运行完整serial rows1参考，再做serial rows8、fixed7 rows8、dynamic7 rows1与dynamic7 rows8四个完整生成对照；检查完整IDs、每个已提交decode logit及最终有效状态，并回归E7完整IDs/prompt IDs。共50次完整预热（其中40个对照），含长输出控制，不能据此宣称新输入泛化。

随后dynamic7 rows1/rows8各四次，按任务和重复交替AB/BA，共80次正式计时。共用原SSE客户端。每次完整输出ID与gold对照；请求结束后独立调用状态检查，不把GPU状态检查混入wire计时。每次都重置请求状态，不做prompt cache。失败留存并停止成绩结论。

报告wire TTFT、完整请求、内部prompt和实际token加权decode，保留逐任务、四次重复与调度宽度分布。单加载只证明同一加载内配对结果，不能替代跨加载稳定性、严格原生同机重测、并发吞吐或长上下文验证。只有完整验证和计时通过才推荐新执行器默认8行。E7原生数据仅作历史参照。
