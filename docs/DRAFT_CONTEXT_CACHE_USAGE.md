# 可选 DFlash 历史投影缓存

只验证了本仓库的 pinned BF16 DFlash / Q4 head、Torch2.13+CUDA13.2、SM121路径。没有改变Target权重、量化或路由；默认draft仍用原路径。

在已经驻留的 `HeadDraft` 上创建wrapper，不重新加载权重：

```python
from spark_infer_q.draft_context import CachedContextDraft

cached = CachedContextDraft.attach(existing_head_draft)
cached.begin(request_id)
proposals, metadata = cached.propose(committed_target_features, anchor)
```

调用约束：

- 每个新请求都调用 `begin`，即使长度或文本相同。缓存绑定的是单请求的、不可修改的已提交Target特征前缀；调用者只能追加接受的特征。相同长度内容替换、修改旧内容或回退后须先重建。接口不会每步比较整个前缀来发现调用者违约。
- 当前实现串行单请求使用。不要在多请求/多stream间共享可变cache；没有跨请求prefix共享或树分支语义。
- 小于256行使用原始计算；更长时缓存FC+norm、五层norm/RoPE后的K和V，追加不足4行时pad到4行再投影。该门槛来自本机形状诊断，不能当其他GPU的通用最佳值。
- 保留当前8行block的attention、卷积、MLP和head计算。缓存不是完整DFlash KV架构替换。
- `max_context` 在创建wrapper前设置，必须是已验证的容量。1024/2048容量的缓存含pad约25.12/50.12 MiB；Target状态和其他临时内存另计。block及prefix须共同落在验证窗口内。
- 只支持本实验的BF16激活。FP32诊断、其他量化backbone或kernel变化，需要重新核对数值与接受率。2K以上必须实现和验证DFlash滑窗/绝对位置以及Target容量，不能简单放大缓存。

缓存代数上复用的是相同历史投影，但不同矩阵行数可能让BF16库采用不同舍入路径。E13记录了一个失败初版及修正后的逐组件检查；不能仅因为greedy Target输出相同，就声称所有DFlash hidden在任意长度都bit相同。性能以包含首轮构建/逐轮追加的实际生成对照为准。

参见 [E13方案](E13_DRAFT_CACHE_CONFIDENCE_PLAN.md)。模型应由[常驻worker](RESIDENT_RESEARCH_WORKER.md)持有，实验结束后继续保留session。
