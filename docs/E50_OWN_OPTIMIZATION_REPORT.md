# E50：自主长上下文优化与八路容量路线

2026-10-25。暂停跨引擎对比，只改进已有动态DFlash链路。**单并发、261120-token输入、512-token输出，同期decode从32.14升至34.49 tok/s（+7.34%）**，完整输出、KV/GDN/PLE状态和动态预算轨迹一致。完整请求wall减少1.58%。这是一个工程输入的一组完整请求配对，不是通用吞吐或八并发结果。

[事前方案及范围扩展](E50_OWN_OPTIMIZATION_PLAN.md) · [脱敏证据](../results/e50/2026-10-25-own-optimization.json)

## 已完成的改动

`optimized_provider(..., query_graphs=True)`显式选择`GraphRollingDraft`：把五层固定query前段和尾段整理为十个Graph岛，复用K4的岛式设计；保留原有上下文投影、attention、Q4 selector、E17 CPU特征和E29预算算术。每次将绝对位置写入稳定输入，避免把首次capture时的Python窗口偏移冻结。十个岛分别持有其输出/残差，跨proposal复用。

`output_cache=True`显式开启原生发布缓存：`process_outputs`后读取原生`last_accepted_token`和`last_output_token`，核对本步结果、累计长度及末token，再追加到每代请求的输出缓存；结束时仍读取原生完整输出做独立核对。不会直接把未验证草稿当作输出，也不修改原生MTP计数。目标是避免每次流式事件都将完整prompt历史转换成Python列表。

两个选项均默认关闭；target的BF16 KV、Graph/E33、W8草稿、E17预测器、动态预算和模型权重保持原配置。没有固定草稿对照、重训、权重下载、重载或跨引擎运行。

## 完整原生请求对照

| 指标 | 原有动态链路 | 草稿query Graph＋原生输出缓存 |
| --- | ---: | ---: |
| 首token后decode | 32.135 tok/s | **34.493 tok/s** |
| 首token后耗时，511 tokens | 15.902秒 | 14.815秒 |
| TTFT | 223.853秒 | 221.142秒 |
| 完整请求wall | 239.762秒 | 235.964秒 |
| 验证轮数 | 180 | 180 |
| 已接受/已观察草稿 | 331/478 | 331/478 |
| 流式发布中的完整历史读取 | 434次 | **0次** |
| 流式发布中的缓存读取 | 0次 | 434次 |
| 终态原生完整输出核对 | 原路径 | 保留1次 |

读取计数覆盖prefill期间的无输出轮询及decode发布，不是输出事件数量。两组都生成满512 tokens，检索3/3；每步宽度、累计接受概率、接受量、head候选并集，以及完整已提交KV/scale/QSA/GDN/PLE指纹一致。target实际Graph replay各180次，16个bucket无新增capture；head/kernel哈希不变，无fallback。优化组草稿十岛共按实际proposal重放，无计时内capture。

模型、Graph、输出头、进程组和算子准备在计时外；submit到drain及最后同步计入完整wall。decode为首个非空输出之后至最后非空输出的窗口，包含草稿、预测、验证、提交、原生调度、流式事件及保护/进度开销。完整状态哈希在计时后计算。没有中途诊断暂停。

原生C++ Scheduler与确定性host backend另验证12个cache开/关场景：混合长度、early EOS、late admission、waiting取消、prepared prompt取消、prepared decode取消，输出、块回收和同ID新代复用通过。完整模型对两个选项组合另通过六组取消/复用，以及预热中的正常停止/输出上限。

**只有一组完整长请求配对，无统计置信区间。** TTFT与完整wall的变化不能全部归因于某个组件，也不把本轮32.14与E49历史30.81拼成累计提速；主比较使用本轮同期基线。

## 组件证据与真正的主要成本

同一真实256K前缀，仅保存小GDN/PLE快照和2040行特征，复用不可变完整KV，不复制整份长缓存。四轮AB/BA、每组512-token的seeded decode中，单独加入query Graph得到38.09→38.61 tok/s（+1.38%），每轮均为正。该循环不含原生Scheduler/事件成本，不能当成完整引擎速率。

真正原生`SequenceData`绑定的CPU探针，五组各100次：261120-token prompt＋512-token输出，`get_token_ids()`后切片的中位成本3.712ms/次，2048-token prompt为0.0284ms；输出缓存tuple读取约0.00045ms。这个微基准不等于整请求增益，完整链路结果见上表。

另在真实前缀做normal/profile/normal的16轮动态验证，轨迹和状态一致。下表来自CUPTI真实`cat=kernel`事件；CUDA annotation父级不重复计入。

| Target验证内核家族 | 累计GPU核时间占比 |
| --- | ---: |
| 路由专家 | 37.55% |
| 量化矩阵投影 | 26.55% |
| BF16矩阵运算 | 12.63% |
| QSA评分/选择 | 7.16% |
| attention主核 | 1.44% |
| 其他 | 14.66% |

验证范围16轮、30000个真实kernel事件，累计910.606ms；kernel区间并集910.262ms、含拷贝/清零活动并集910.539ms。区间总wall961.204ms。区间差额不是可直接删除的Python时间，更不是SM利用率；profiler耗时也不用于正式TPS。当前重核仍是专家和投影，后续要优先研究执行效率及跨请求合批。

## 八份256K状态：实测布局与未完成项

以下为**每次只分配一个空状态**的实际CUDA分配增量，再加E49实测公共区外推八份；没有生成、没有同时分配八份，也没有验证INT4质量。

| KV格式 | 一个完整请求状态 | 八状态＋已有公共区下界 |
| --- | ---: | ---: |
| INT8 | 4.345 GiB | 107.516 GiB |
| FP8 | 4.251 GiB | **106.766 GiB** |
| INT4 | 2.845 GiB | 95.516 GiB |

完整状态含GDN/PLE、缩放和BF16 QSA索引，不能把它与K/V主体大小混用。FP8与INT8都是8bit，容量很接近；INT4的缩放与其他状态没有减半。公共区已有72.76GiB常驻对象/eager/prefill分配；表内仍未计齐八路Graph、head辅助及草稿特征/缓存。

当前88GiB是保留的进程分配限额，不是硬件容量。系统可见总内存约121.63GiB，不能简单减去CUDA allocated就当作可用空间。一次独立快照中，去重target CUDA权重69.85GiB、draft约2.01GiB，进程PSS约7.15GiB；还有驱动、系统和其他进程。reserved空闲块、驱动free与MemAvailable含义不同，不能重复相加。逻辑PLE表26.82GiB也不能直接当成已驻留RAM。快照会随缓存和工作负载变化，不是八路峰值测量。

此外，现有长上下文自适应桥仍限单请求，共享长Graph路径限两个target-only槽；另一条短上下文batch路径最多四路。资源够用之后，还需扩展slot/metadata、每代草稿缓存、Graph键、部分提交、取消恢复和真正的多请求验证。改一个并发常量不构成八路支持。

### 保留FP8的优先研究方向：有界原始索引

固定版本`attention.py::_pool_block`从当前位置所在block开始，读取本次新完成的四行块；历史检索的`_scores`读取完整`POOLED`。因此完整原始`IKC`历史可能不必一直保留。每请求原始索引805355520 bytes，约0.75GiB；完整池化索引继续保留。

最大prepare1024行、ratio4时，理论上需要新窗口加之前未完成块的最多3行，即至少1027行；2048行环可覆盖这个地址活跃范围。CPU模型跨644个窗口验证了整段长prefill、回绕、零保留、部分接受与重算，池化读取的输入逐项一致。假设2048行存储，八路理论节省约5.95GiB，且不降低FP8 K/V或池化索引精度。

**该环尚未实现，更没有GPU/服务验收。** CPU模型只证明所检查reader的地址活跃范围，不证明CUDA映射、同步、Graph地址、完整部分提交或跨请求隔离。实际方案还要考虑分配粒度；discarded原始历史的旧全前缀hash也不能继续当作状态等价契约。下一步先验证有界IKC仍保持pooled/selector/logits/features/有效尾部和完整输出精确，再核算实际节省及八路峰值。

FP8八并发256K保留为目标。INT4是容量备选，需要独立同格式正确性及长文质量门槛，不因空状态放得下就晋升；E47原始精度失败也不会在本轮被重标通过。

## 复现与收尾

worker46235/load_count1始终常驻；probe、完整study及两个layout阶段的独立审计均通过，原算子/no_grad、进程组、环境和实验模块恢复，最终CUDA allocated回到77393408000 bytes。没有empty_cache、提高限额或停服务；E43历史32MiB差额仍单独记录。

本地完整391项测试：372通过、19项可选依赖跳过。新增测试覆盖原生发布的中段/末尾/长度/终态篡改、默认行为、非法选项、事件计时/执行量、真实kernel去父级与重叠区间归并，以及CPU原始索引活跃范围模型。

显式API在已有E46/E49环境中使用：

```python
provider, profile = optimized_provider(
    session, history_window_rows=2040, max_request_context=262160,
    query_graphs=True,
)
provider.worker.warmup()  # request timers之前，owner空闲时
# speculative_engine(..., provider=provider, mode="adaptive", output_cache=True)
# engine完成并关闭后，关闭这份provider拥有的Graph；保留session权重。
provider.worker.close()
```

原始忽略归档`results/raw/e50/2026-10-25-final-v3/`共932文件，压缩包7763191 bytes，无权重。SHA256：`de7969814293f46e76660fc73fd13bb7800cb43d482c5cbc54204574875958c2`。包含原始CUPTI trace、全部stage源码/作业、Native绑定源码和E49引用。CPU活跃范围模型随本仓库提交，单独记录源码SHA。

```bash
python3 scripts/e50_export.py \
  --raw results/raw/e50/2026-10-25-final-v3/worker \
  --out results/e50/2026-10-25-own-optimization.json
```

导出器复核所有stage源码、作业/引用/审计链、外部算子与二进制、当前库、原生事件/预算/执行量、内存算术、CUPTI真实核归类与CPU索引模型，结果`PASS_E50_EVIDENCE`。早期归档传输副本解包EOF被保留，最终改用远端文件封存、文件传输及端到端SHA/长度校验，不将传输失败算作模型或GPU实验失败。
