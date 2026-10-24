# K4：自有批量 DFlash 后端

本报告保留该阶段的 legacy Target 对照。后续[自有紧凑 Target 迁移](K4_OWN_TARGET_REPORT.md)已完成两请求研究实现及限定数值/生命周期检查，完整服务性能晋升仍失败；两阶段结果独立计量，不合并样本。

**自有批量草稿已接通两请求异步服务；默认不变，性能晋升尚未通过。** 本轮实现了自有五层 query-block 执行、请求隔离的 attention/卷积、增量上下文 K/V 缓存、候选选择和 1～32 行 W8 内核。四请求只完成草稿组件验证，完整服务仍限制两个请求。Target 仍使用此前的 TensorFold 后端；草稿的 Q4 输出投影也是显式注入的 TensorFold provider，不能宣称已完全去依赖。

公开结果：[2026-10-24-batch-draft.json](../results/k4/2026-10-24-batch-draft.json)，由 `scripts/k4batch_draft_export.py` 从封存结果重算。正式对照是最后一版的同期旧/新执行，不与此前约 48 tok/s 单请求、上游同 prompt 多流或早期 K4 结果拼接。

## 1. 本轮实现

- `spark_infer_q/draft_batch.py` 自行组织多个请求的五层 DFlash 计算；没有 TensorFold model/state/forward 导入。共享 query 投影、MLP、输出投影和 selector；attention 明确带请求维和有效 K/V mask，卷积在每个八行 block 的边界截断。
- 上下文 FC/K/V 投影保留逐请求 GEMM 形状，避免额外改变 BF16 reduced-precision 归约。它们尚未合批。长上下文按需分配各 generation 独立的 K/V；短上下文不为每请求预分配整份缓存。失败的 append 不发布新的有效长度，重试只覆盖未提交尾部。
- `scripts/k4batch_draft_w8.cu` 是自有 W8 的批次扩展：一个 warp 的权重读取供最多 32 行复用，保持逐行 FMA、warp reduction 和缩放顺序。真实 gate/up 权重在 1/7/8/16/24/32 行与原内核分块结果逐位一致。没有完整展开 INT8 权重。
- `BatchedRequestDraftProvider` 复用冻结 E17 NumPy 模型、辅助头/Q4、W8 权重和原动态预算。仅剩一个输出 token 时跳过草稿；按 generation 回收缓存。当前不重训预测器、不新增固定草稿运行模式。
- native `prepare` 一次调用 `propose_batch`，排除取消请求，再逐请求发布 proposal 和预算。共享草稿墙钟只按实际参与请求摊分一次；`draft_batch_ms/size` 单列，避免把整个批次时间重复收费。
- 单个 survivor 使用自有后端的三维 attention 快路，避免继续分配批次 padding/mask。DFlash 仍为 eager；Target/精确辅助头沿用已有 Graph，未把本轮称为新的 Graph 加速。

## 2. 数值差异没有被隐藏

直接合并上下文 FC 行时，第一层之前就出现误差。独立探针确认，同一输入/权重只改变矩阵行数，Torch BF16 reduced-precision GEMM 就会产生不同结果；切换 reduction 标志也不能复原所有原结果。最终实现因此保留上下文投影形状。

即便如此，批量 attention 的存储/计算布局及后续 BF16 运算仍使 draft hidden 不逐位相同，选定组件场景的 hidden 相对 L2 约 1.1%～2.1%。长上下文组件还使用重复的真实 Target feature 行构造合成前缀；部分场景的候选改变，不能当作自然长文本的质量评测。单请求增量缓存的 260/261/263/266/270 行探针中，proposal 和 signals 与原路径一致。

E17 权重和特征定义没有改，但输入数值会变。正式在线轨迹的同位置比较中，224 次出现概率变化，最大绝对差约 0.164；这是执行路径引起的输入变化，不能声称预测器输出仍完全一致。短组和长输出组的选定诊断中，接受总数和 Target forward 次数保持一致，物理行仅有少量差别。

**草稿数值等价门槛未通过。** 另行验证的最终 greedy tokens 与 Target logits/features/有效状态在本轮选定服务用例中一致：共 1,727 logits 行、1,424 已接受 feature 行、328 次状态检查。该结论不覆盖随机采样或未测输入；没有用草稿候选相同代替 Target 状态验收。

## 3. 同期五轮服务对照

模型、量化、BF16 KV、E17、原 E29 单请求预算、输入、输出 cap 和自然 EOS 保持相同。每组每模式五次，旧/新交错运行，保存全部样本；接受率和动作可以因上述草稿数值差异而变化，因此这是两条完整动态路径的比较，不是强制相同 proposal/预算轨迹的纯算子消融。

| 两请求负载 | 旧 direct decode | 新 direct decode | 旧 SSE cohort | 新 SSE cohort |
| --- | ---: | ---: | ---: | ---: |
| 短输入，cap 32 | 44.26 | 45.67 | 38.57 | 39.92 |
| 长短输入混合，cap 24/16 | 60.45 | 61.70 | 13.39 | 13.49 |
| 短输入，cap 128 | 38.09 | 40.00 | 36.44 | 37.84 |

单位 token/s，均为总 tokens 除以总墙钟。direct 从两个请求都得到首 token 后计时，包含首轮草稿、后续 native 回收。SSE 从该 cohort 最早的首 token 到全部请求终止计时，扣除每请求恰好一个首 token；它可能包含另一个请求尚未结束的 prefill，不能与旧报告“两个首 token 之后”的 SSE decode 数字直接比较。

长短组的 prefill 约三秒，先到请求和流的输出时机会显著影响 decode 区间。完整请求吞吐旧/新为 6.34/6.48 tok/s；短组 33.76/34.60，长输出组 34.75/36.20。原始 TTFT、每请求 TPOT 和 chunk gap 均在 JSON 中。

最终 SSE 短组约 +3.49%、长输出组 +3.83%。按 cohort 配对 bootstrap 的 95% 区间分别约 −4.77%～+10.95%、+1.00%～+7.85%；只有五个 cohort，尾延迟每模式每组只有十个请求。短组 TTFT p95 还从 502.99ms 增至 534.16ms，超出预设 +5% 初始约束。**未达到吞吐 +10% 目标，也未建立广泛稳定的尾延迟优势，默认不晋升。**

组件测量与服务结果分开：短混合两请求的草稿中位时间 21.86→13.84ms，约减少 36.7%；四请求混合 39.83→28.67ms，约减少 28.0%。组件区间排除 Target，仅反映草稿执行；不能把它们当作整机吞吐增益。

## 4. 为什么服务收益还有限

在选定 direct 正确性轨迹中，短组只有 7/18 个 Target forward 同时处理两个请求，长输出组只有 10/65。自然 EOS 和不同接受进度使大量轮次只剩一个 survivor，草稿合批只能加速其中一部分。短组 Target forward 仍为 18，长输出仍为 65；当前 Target 的 padding、逐请求 GDN/QSA 内部路径与主机阶段没有在本轮替换。

下一步按[自有后端计划](DEEPINFER_OWN_BACKEND_PLAN.md)推进 Target 紧凑行布局、跨请求 GDN/QSA 状态/执行，以及有界 Graph 和四请求完整链路。不能只把请求上限改成 4，或依赖预测器更换，就宣称获得上游那样的并发吞吐。

## 5. 生命周期、资源与结果追溯

最终版本本地 246 项测试，232 通过、14 因本机没有可选 Torch 环境跳过；新增 tensor 卷积边界和不等长 attention 测试也在远端 CPU/Torch 环境实际通过。最初的远端八项测试中，成本表兼容性测试因封存包遗漏 JSON fixture 而报 FileNotFoundError；补齐打包清单，在最终封存源码上重跑八项，全部通过。

真实 GPU 14 项既有生命周期用例在 batch draft 下通过，覆盖 EOS/cap、pending position、不同取消阶段、在途 survivor/同 ID 复用、关闭超时、native owner guard、故障回收和新 owner 恢复。另有 5 项草稿 GPU 检查：更换 companion 内容不改变另一请求 hidden、部分 append 故障后前缀不变及重试、同 ID 新 generation 与新后端相同、整批预校验、四缓存上限及释放后复用。Target 保留 80 张 Graph，无 fallback。

全部 GPU 工作通过共享 resident 队列串行运行，PID319、load_count=1。最终 347 审计无活跃批次对象，Gloo/NEXTN 环境恢复，二次 GC 后 allocated 稳定为 77,829,615,616 bytes；未调用 empty_cache、未重载/停止模型。其他实验在这些作业之间和之后也通过队列运行，后续全进程显存不能归因给本轮。

初始 330 验证脚本把 Tensor 当成 Tensor 列表再次 cat，失败记录保留；修正后继续使用同一份已构建内核。331/332 为合并上下文投影的早期探索，334 为保留形状后的版本。335/336/338/339 是 singleton 快路之前的完整服务/生命周期/隔离/审计，**不与最终结果合并**；最终正式作业为 340/341/342/343/347，344 为补齐 CPU fixture 之前的审计，因果 GEMM 探针为 333。

raw 包含所有上述作业、封存源代码/manifest、完整结果以及 W8 二进制。`results/raw/k4-batch-draft/evidence.tar.gz` SHA256：`a94741124ec4c805b9ce96677812b6c54a90890ba1c1ba1fab9d6e7a01b81859`。新 W8 source SHA256：`bed35d108f06aad2d16e7eaed7365285398b43abc9418484e40a348668b1e335`，binary SHA256：`2dc215c8af8e379ff90da8b28bdce17fb3c9f02322d6af36eb47cf568a7a5f71`。

## 6. 使用与复现

在已准备环境、持有已核对的 W8 batch library 时，可在原 K4 启动参数上追加：

```bash
--draft-execution batch \
--draft-w8-batch-library /path/to/spark_q_draft_w8_batch.so \
--draft-w8-batch-sha256 2dc215c8af8e379ff90da8b28bdce17fb3c9f02322d6af36eb47cf568a7a5f71
```

默认 `--draft-execution sequential`。batch 仅支持 adaptive，不与旧 `--batch-costs` 表混用；Python API 的新 W8 指纹也会使旧表校验失败。当前 E29 成本仅作对照，`draft_cost_calibrated=false`；没有把旧串行草稿成本当作已重新校准的批次成本。

复现组件会在独立封存目录构建内核；后续脚本通过 `--component-root` 显式引用该结果，不覆盖旧作业或重载 resident 模型：

```bash
python3 -m unittest discover -s tests -q
python3 scripts/k4batch_stage.py unique-component unique-component.json k4batch_draft_validate.py
python3 scripts/k4batch_stage.py unique-service unique-service.json k4batch_draft_service.py --component-root /workspace/e1/k4batch-20261023-unique-component
python3 scripts/k4batch_stage.py unique-life unique-life.json k4batch_draft_lifecycle.py --component-root /workspace/e1/k4batch-20261023-unique-component
python3 scripts/k4batch_stage.py unique-isolation unique-isolation.json k4batch_draft_isolation.py --component-root /workspace/e1/k4batch-20261023-unique-component
```

每次使用唯一版本/作业名。完整历史包的公开结果可用 `k4batch_draft_export.py` 重算；其固定作业表用于本轮证据，不自动接受新实验替换旧结果。
