# E46：直接 256K 实跑

2026-10-25。按用户要求跳过逐级8K/32K里程碑，**单请求256K执行门槛通过**。
实际262,080-token输入、24-token输出（EOS），logical262144、physical262160，输出上限64；
没有扩大checkpoint的262144位置配置。尚未测双请求256K或多并发1M。

[方案](E46_256K_PLAN.md) · [完整脱敏数据](../results/e46/2026-10-25-direct-256k.json)

## 实测结果

| 项目 | 观察值 |
| --- | ---: |
| 原生请求输入 | 262,080 tokens |
| Prefill＋首输出事件 | 230.399秒，约1,137.50 input tok/s |
| Native TTFT | 230.384秒 |
| 输出 | 24 tokens，EOS |
| 诊断后恢复的decode | 1.149秒，23个首token后tokens，20.018 tok/s |
| 观察到的草稿接受 | 18/22，81.82%；6轮验证 |
| 三个检索键 | 3/3命中 |
| 有界原始草稿特征 | 104,448,000 bytes，99.61MiB |
| 研究过程峰值CUDA allocated | 87,413,067,264 bytes，81.41GiB |
| Prefill期间最低系统可用内存 | 17,644,822,528 bytes，16.43GiB |

这是一次容量/正确性研究请求，在首token后插入7.890秒诊断暂停；表中decode计时从恢复
驱动到完成，排除该暂停。它不是未中断的HTTP/完整服务测速，也没有配对原生引擎排名。
只有24-token短回答、6轮验证，不将20.018 tok/s当作通用稳定吞吐，也不与E45四任务汇总横比。

输入为可复现的中英文合成ledger，使用真实chat template，保留长文前部及最后256 tokens。
三条registry事实位于开头、中段及末尾，模型输出：

```text
MAPLE-7319
RIVER-4826
CEDAR-9053
```

这是该单一合成检索样本通过，不代表自然长文、全部256K任务或1M质量已验证。

## 执行与状态证据

同一份常驻权重，DeepInfer原生单槽调度，255块1024行＋最后960行；每块保存位置、耗时和内存。
target仍用BF16完整KV/QSA索引、GDN与PLE；prefill为E44 Mia tile/chunk1024，
decode为Graph/E33/W8/E17/E29及E45绝对位置2040行草稿后缀。长程成本/置信预测保持冻结。

QSA在超过128K后使用原有tiled radix selector。实际前缀位置261889/261985/262080的三行，
读取各65K左右的真实block scores，以CPU稳定排序、低block ID优先处理平分，再展开key列表。
全部与GPU选择结果精确一致，实际选择2049/2049/2048个keys。这里只独立复核selector，
没有宣称全部indexer浮点计算具有另一套独立全模型oracle。

在同一256K前缀，width1–8 × 两种GDN parity，共16桶、72行完整词表logits：
Graph与原生eager完整logits、五层特征、完整有效GDN/PLE状态和新提交KV/index/pool尾部一致，
包含完整/部分提交。之后单独算target-only greedy continuation，再恢复同一已提交前缀，
继续真实动态请求：24个输出tokens、最终状态与cache尾部完全一致。
所有诊断及decode之前/之后，完整已提交长前缀KV/index/pool的SHA256一致。
实际Graph共22次replay（16个数值桶＋6轮动态验证），无新capture、Graph或head fallback。

没有将整段256K prefill再用逐token oracle重跑；这轮证据是已通过E45门槛的prefill路径，
加实际长程selector、Graph、同前缀continuation和完整前缀不变检查。

## 内存与并发的含义

**这轮只有1并发。16.43GiB是系统可用内存，不是一个256K请求的KV总大小。**
target张量本身约69.85GiB；本轮请求前常驻CUDA allocated为72.08GiB，含其他常驻对象。
峰值比该基线多9.33GiB，包含请求KV/状态、Graph/head、prefill及诊断工作区。
按实际BF16布局计算，每256K请求的KV＋QSA原始/池化index增长项约6.94GiB；
第二个请求共享权重，但仍需额外KV/GDN/草稿状态，不能复制整台进程占用，也不能只用
系统可用内存除以6.94来宣称并发数。

现有resident进程CUDA分配限额仍为88GiB，单请求Graph workspace仍是独占lease。
双请求容量及其执行/Graph复用需要另行验证；本轮不通过抬高限额、卸载权重或清理共享cache凑容量。
下一优先级转向KV压缩及多请求状态/工作区复用，而不是继续停在更短的上下文门槛。
分页负责物理分配/复用，量化降低每token容量；1M还需要超出当前位置配置的质量证据。

## 改动、使用与收尾

扩展显式target/prefill物理容量入口至262160，默认请求路径仍不启用长后缀。
大容量Graph预热改用已有`State.snapshot/restore`，只保存GDN/PLE等小状态；
预热始终保留位置0那一行KV，覆盖的位置均>=1，不再为seed复制约7GiB的完整KV。
<=8192容量仍保留原预热路径，新增CPU回归明确禁止大容量clone，并检查16种桶和已提交KV前缀。

已有resident/native环境中的主要参数为：

```python
target = optimized_target(session, capacity=262160)
target.warmup()
provider, profile = optimized_provider(
    session, history_window_rows=2040, max_request_context=262160,
)
# TensorFoldPrefillProvider(..., capacity=262160, expert_backend=experts)
# PrefillExecutor(..., chunk_rows=1024)
# speculative_engine(..., capacity=262144, max_requests=1, token_budget=1024)
# prompt<=262080, max_tokens=64 for the measured workload.
```

使用依赖同[E45接口](E45_LONG_CONTEXT_USAGE.md)，仍为greedy/text-only研究环境。
E46没有另外重测长HTTP、取消或双请求，因此不把E45的4K生命周期证据扩大为256K服务证据。

一直复用worker46235/load_count1。job480执行、job482独立收尾审计通过：
原算子/no_grad、native group/env及实验模块恢复，
**allocated77,393,408,000→77,393,408,000 bytes**，没有重载权重或empty_cache。
E43更早未解释的32MiB差额仍单独保留，不由本轮基线审计覆盖。

原始归档243文件、1,071,575bytes，SHA256
`b297329171c350f9744e15917fb44a875600728341c108ca91f73f7ebe034431`，
位于忽略目录`results/raw/e46/2026-10-25-final/`，不含权重。

```bash
python3 scripts/e46_export.py \
  --raw results/raw/e46/2026-10-25-final/worker \
  --out results/e46/2026-10-25-direct-256k.json \
  --study-version study-v1 --study-job 480-e46-study-v1.json \
  --audit-version audit-v1 --audit-job 482-e46-audit-v1.json
python3 -m unittest discover -s tests -q
```

本地357项测试：338通过、19项可选依赖跳过。新回归检查大容量预热不clone完整KV，
并拒绝较短输入冒充256K、缺失selector/Graph桶、状态/prefix差异、错误计时和夸大检索命中。
