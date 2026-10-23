# E24：固定配方，单独扩大数据覆盖

2026-10-23 UTC。E24 等待 E23 正常结束后执行，使用同一个常驻 Target/DFlash，原模型与执行器默认方案不替换。E23 当前的45,081训练位置不混入新的E24测试标签。

## 数据与归属

预先准备 **504 条、504 个 root**：360 train、72 calibration、72 test；英文336、中文168。所有 OASST root 与全部 E17/E23 root 排除，问题做标准化精确和 trigram .82 近重复排除；同 root 不跨 split。

| 场景 | 总提示 | 来源 |
| --- | ---: | --- |
| 英文/中文问答 | 168 | OASST1 各84条人工 reviewed、nondeleted、nonsynthetic root |
| Python 代码 | 84 | MBPP 官方 train task_id 601～974 中抽样；只给问题和公开测试，不给代码答案 |
| 英文数学 | 84 | GSM8K 官方 train 的问题，不给答案/推导 |
| 多轮对话 | 84 | OASST1 原始 user→assistant→user 三条 reviewed、nonsynthetic 消息；保持真实角色，最后一条作为待回答问题 |
| 长上下文 | 84 | 自编采购表，768/1024/1536 token，英中各42条；查询开头记录，按已有首部/尾部裁剪规则保留问题与查询记录 |

外部来源固定 revision 和原始 SHA，保留许可与归属：

- [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)：Apache-2.0，revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`。多轮消息源 SHA256 `286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c`。原来的 prompts 文件只有 root，不能据其声称采到真实多轮，所以独立取回 ready messages。
- [MBPP, Austin et al., 2021](https://github.com/google-research/google-research/tree/master/mbpp)：数据 **CC-BY-4.0**，不要与仓库 Apache-2.0 代码许可混淆。数据 revision `57183fd52c2d857aeef16d1fad15efe55aa61220`；许可依据[数据卡](https://huggingface.co/datasets/google-research-datasets/mbpp/blob/4bb6404fdc6cacfda99d4ac4205087b89d32030c/README.md)及其存档 SHA。
- [GSM8K, Cobbe et al., OpenAI](https://github.com/openai/grade-school-math)：MIT，revision `3101c7d5072418e28b9008a6636bde82a006892c`。

这里划分的是预算预测器训练/校准/测试，不是给主模型做监督微调，也不是 MBPP/GSM8K 官方基准成绩。采购表是合成格式，同一格式跨分区，只能检验此类输入内的覆盖；不证明陌生模板或自然长文档泛化。OASST1 可能在主模型预训练中出现，排除本项目 root 不等于排除预训练污染。

## 执行前检查与采集

071先用常驻 tokenizer 检查全部504条实际角色输入、裁剪后长度、输入＋384输出＋8预留≤2048，记录每条实际 token 序列 SHA；不执行 Target。072从 train 选中英问答、代码、数学、真实多轮、1536输入共六个场景，分别串行/加速采16-token轨迹，比较所有 token、完整已提交隐藏特征、logits、live state、校准结果和训练行，并核对真实多轮 prompt SHA。全部通过才采学习数据。

采集核心复用原 E21 已验证模块，E24 用独立的角色适配器将真实 messages 传给原 tokenizer，禁止将聊天历史伪装成单轮字符串。原 E17/E19/E21 源保持字节不变。每条记录的 prompt SHA 再与071核对。

学习432条分九片，每片48；仍最多64均匀前缀位置，保持137维特征、七条件标签及各宽度 `emitted`。calibration仍保留初始串行快照、各宽度成本检查。新成本只作为记录，不在本轮重拟合成本曲线，避免混入第三个因素。test保持独立，不进入学习入口。

## 固定配方与门槛

E24在新标签产生前封存 **E23 calibration 选中的训练配方**，即当前 `decision_regret`；不依据 E23 新 test 输赢来回头调整。结构、seed、优化预算、后端和旧成本不变，仅增加新 train，并使用旧＋新 calibration 做原样 Platt 校准和候选比较。train-only q仍按相同算法在新的训练集合计算。

候选为新数据重训结果与冻结的原 E23 模型，在同一份合并calibration上计入各自实测CPU费用。若新数据结果没有超过冻结对照，正常保留负结果，test不采。选中新结果才封存两模型、打开72条E24 test（另三片，每片24），一次性评估与冻结模型/原线性模型的分组增益及英中门槛；禁止按test重新选型、加超参数或回灌错例。

E24只检验固定配方下增加此批数据的效果；不能据此宣称数据量、来源多样性、长度或多轮中的某一个因素单独起作用。本轮输出离线评估，若通过才具备另行完整在线A/B的资格，没有自动部署或离线数值当在线TPS。

有限队列为071 preflight → 072 producer validation → 073～081 learning → 082 fit → 条件执行083～085 test → 086 evaluation。失败停止，不覆盖已有结果，不自动重载/重试。E23/E24各有独立源码、输入、候选模型及清单SHA封存。[E23方案](E23_OBJECTIVE_COMPARISON_PLAN.md)。
