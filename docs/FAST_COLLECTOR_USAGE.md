# 加速训练标签采集器

新入口为 `scripts/e20_collect.py:run(session, arguments)`。它运行在已有常驻worker中，不自行加载Target或DFlash。E17/E19封存的采集源码、数据及选型顺序不替换；后续训练数据清单可以使用这个入口。

## 方法

原采集器先逐token运行Target，得到完整greedy参考轨迹，再在每条任务最多64个均匀位置生成DFlash特征及七深度标签。新版本在第一阶段用固定8行的投机验证推进轨迹：只提交接受前缀，将相应Target隐藏特征写入预分配的连续缓冲区。纠正/bonus token保持pending，不提前提交EOS或拒绝的草稿。第二阶段的位置选择、DFlash完整七草稿、137维F1特征和`emitted_by_width`标签保持原契约；两个独立的采集草稿缓存按请求重置，不复用轨迹生成的缓存作为标签输入。

校准任务的前八次转换仍逐token执行，保留原来的逐位置状态和logits快照。第一位置的1～8行成本预热及三次正式计时、committed-logit/live-state核对仍保留。

## 启用条件

必须先完成 `e20_validate` 常驻作业：8个训练来源任务、三个配对重复，比较完整采集时间。覆盖中英文、混合语言、代码、JSON提取、1K/1.5K上下文及输出cap尾部；另有first-token EOS、cap=1/2/9的实际GPU边界检查。整条已提交隐藏特征、输出token、完整committed logits、最终有效状态、采样位置、逐位置F1输入/七深度标签必须精确相同。正式计时保留生产采集自身的检查，额外的跨模式oracle比较在计时外。

验证状态为 `PASS_E20_FAST_COLLECTOR` 且有正的完整采集加速后才允许新入口运行。入口核对验证报告SHA、所有受验证源码SHA、输入清单SHA，并拒绝覆盖已有结果。验证后会额外调用一次实际采集入口，生成两条16-token smoke记录，检查结果持久化和来源绑定。验证失败时保留失败证据；不能仅因token相同就放宽隐藏特征要求。

## 输入及作业参数

在worker根目录准备一个独立JSON清单，顶层为`{"tasks": [...]}`。每条任务包含：

- `id`, `group`, `family`, `language`（`en/zh/mixed`）和 `split`（`train/calibration`）。
- `prompt`, `thinking`, `output_cap`（1～1024），可选`target_prompt_tokens`沿用现有首尾裁剪语义。
- 同root/语义组必须在同一split，id不可重复。实际tokenizer检查输入＋输出cap＋8行预留不超过2048。

入口目前只接收学习数据，明确拒绝test split。后续独立测试仍由各实验的封存选型门槛控制。

常驻作业参数示例（SHA均须从实际文件计算，不能使用示例占位值执行）：

```json
{
  "manifest": "next-learning-manifest.json",
  "manifest_sha256": "<exact input SHA256>",
  "validation_sha256": "<exact e20-collector-validation.json SHA256>",
  "run_name": "next-learning",
  "shard": 0,
  "shard_size": 51
}
```

按常驻队列协议提交`module=e20_collect`及其实际源码SHA，一次仅执行一个作业。输出为独立的 `e20-fast-<run_name>-<shard>.json` 和 `-rows.jsonl`。已有结果存在时直接拒绝，失败不会自动重试或重载模型。

记录中的`trajectory_ms`包含分词、prompt处理、轨迹生成及必要特征保留；`collection_ms`还包含采样草稿/特征、标签和校准成本复核。`trajectory_target_calls`报告实际Target调用数。新记录不把投机轨迹时间命名为`serial_ms`。行级`x/y/emitted`与原训练契约兼容，但数据集元信息属于独立生产者，后续训练驱动需要显式验证其状态/SHA，不能伪装成旧E17分片。

这是有界greedy单请求采集加速，不是模型训练加速、模型服务decode TPS、多请求batch或4K+验证。数据采集收益不可直接换算成用户推理收益。
