# E14：在线置信度验证预算

冻结于首次 GPU 运行之前。复用常驻权重，不更换 Target 内核，不加入专家成本模型。

三组均启用 E13 上下文缓存、同一 Q4 head 和 BF16 selector：

- `native_cached`：现有 MTPVarlen 决策在 draft 前，E6 先验加过去步骤反馈。仅将最大深度限制为5，与另外两组一致。
- `depth_budget`：每轮先生成草稿，再用 E13 train-only 固定深度条件接受率选择验证宽度。
- `margin_budget`：与上一组相同，但改用当前草稿的 selector edge margin 查 train-only 分桶概率。

两种 post-draft 组都调用原生 MTPVarlen，当前与 previous rates 设为同一份已知当前草稿估计；不把它描述为原异步引擎。冻结同一 E6 verify+commit 曲线，所有动作（含 width1）均加入同一已付草稿成本，防止把收缩到1行误当作免草稿。无在线成本反馈、无标签拟合；native reference 保留原 runtime feedback。两种 post-draft 组使用相同精简 selector 输出；margin 额外提取/同步/传输计入正式 decode。native reference 保留原 metadata 输出，因此 native 对比包含决策结构和 metadata 差异，置信度的独立贡献只看 margin 对 depth。

仅深度1–5有校准，最多6行（anchor+5）；cap尾部进一步截断。当前 Target 路由和拒绝后的结果不能成为预算输入。逐步记录预测累计概率、宽度、接受/观察数、阶段耗时。冻结校准源和 SHA，不使用 E13 test 标签。

先做独立 smoke 的 selector 路径核对与 committed-logit/live-state oracle。随后9个新固定提示（6短、3个不同主题的合成重复长背景，1024/1536/1792输入token），每组4重复，顺序轮转并在偶数轮反转。每请求最多生成128 token，遇EOS提前停。每任务每模式另做完整逐committed-logit对照；正式计时仅检查结束输出与live state。所有校验在正式 decode计时之外。

主要结果：margin 对 depth 的任务median总耗时比、weighted decode TPS、逐任务与逐轮结果、任务bootstrap区间。native reference 对照另报。检查验证轮数、验证/提交行、接受标签、信号成本；信号采集与selector本体在同一计时段，不将其冒充独立GPU kernel成本。无显著收益或退化同样保留；不在同一测试集反复调参后宣布独立验证。该实验是bounded decode，不是TTFT/完整请求/随机采样/多请求吞吐证据。

## 正式运行中的结构诊断

首个任务完成后发现两种post-draft预算选择相同。枚举五深度二分桶的32种组合，在冻结成本上均选6行。为区分“信号无效”和“冻结成本使动作不敏感”，排入独立 `e14_cost_probe.py`：实际调用同一原生MTPVarlen核对32角点；用不含E14测试输出的新合成校准背景，在384/1024/1792长度逐宽度1–6各测4次verify+commit（不含state clone），并测cached草稿cold/reuse。每种width先与逐token logits/live state核对。校准是teacher-forced固定草稿前缀，不能称为完整请求吞吐或在线收益。不会替换正式E14的冻结曲线，不在相同测试集重调后重测。

## 复现入口

已有准备环境、同一外部二进制和常驻worker是前提。将 `scripts/e14_study.py`、`scripts/e14_tasks.py`、`scripts/e14_cost_probe.py` 和 `spark_infer_q/confidence_budget.py` 复制到对应准备目录，并将 [训练表](../results/e14/2026-10-21-training-calibration.json) 命名为 `e14-calibration.json`。按 [worker协议](RESIDENT_RESEARCH_WORKER.md) 提交确切源码SHA的 `e14_study`，先arguments `{"smoke":true}`，通过后用空arguments执行正式对照；`e14_cost_probe` 使用空arguments。不得重新加载模型或修改运行中的已哈希源。

原始正式artifact用 `python3 scripts/e14_analyze.py INPUT OUTPUT` 验证并导出。训练表仅复制E13公开结果的train-only `model` 中1–5深度，保留原文件SHA；不读取held-out标签重新拟合。

实跑沿用常驻环境中 `ee7a725` 的 `spark_infer_q/draft_context.py`，与当前工作树只差EOF空行；实际源SHA随结果保留。需要严格复现源字节时，从该提交提取helper，不覆盖正在运行的文件。其余本轮新模块以公开结果内SHA为准。
