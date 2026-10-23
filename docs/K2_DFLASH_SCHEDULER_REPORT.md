# K2：DFlash2 与现有 E17 动态预算接入 Scheduler

2026-10-23，`PASS_K2_INTEGRATED`。K1 API 现支持 Q4 DFlash2、固定 K=1/3/7、原生 lagged/E16，以及 **E17 线性预测器 + 本轮 post-draft 动态预算**。`k2_serve.py` 默认 `adaptive`，保留其它模式作显式对照。没有训练新预测器或重载常驻权重。[事前方案](K2_DFLASH_SCHEDULER_PLAN.md)、[用法](K2_DFLASH_SCHEDULER_USAGE.md)、[E17 接入证据](../results/k2/2026-10-23-postdraft-e17.json)、[早期 E16/原生滞后证据](../results/k2/2026-10-23-native-dflash.json)。

这次首先交付调度接入与有界正确性验收。小样本计时不能证明优于接入前的研究执行器，也不建立最新 TensorFold/Graph 执行路径排名。

## 现有预测器与预算是否保留

`adaptive` 使用冻结 E17 `unweighted_f1` Python 线性头，保留137维 F1特征、标准化、七个 heads 和校准，模型 SHA256 `ed7a4873f1d8e2d95ed54c8acf2c34ef3cf4772ea113b48ee09e2775f3391dfe`。该模型保持 experimental 身份；本轮选择它是为了保留已有组合，不是新晋升结论。E27 的线性模型元数据与它相同，冻结验证成本相同；本模式采用 E17 原冻结的 CPU predict 耗时。

每次当前草稿生成后，预测器输出七深度累计概率，已有 `BudgetProbe` 同时接收该概率作为 current/previous rates，选择下一次验证宽度。候选数受剩余输出上限限制，剩一 token 时跳过草稿/预测。context<768 用 short 成本，否则 long；每个候选成本都包含完整 draft/feature/predict 成本，含 width1。没有开启未晋升的 E27 历史成本修正，也没有切换 NumPy、W8 drafter 或整模 Graph。

预算算法沿用已编译的上游 MTPVarlen，二进制 SHA256 `722b6ea835fa3e7e349af7f22ac727045a65af487a35f6709f3838e066bd678e`。原生 Scheduler 负责请求、保守最大 NEXTN=7 预留、inflight 轮次和输出账本；公开 `update_metadata_for_sampling_zos` 更新 metadata 和 process_outputs 使用的最终 q。最大预留 token 数保守保留，单请求模式不把空余预留重新分配给其它请求。

独立执行旧循环，三个输入的完整验证宽度序列与新运行时逐轮完全一致：

- 算术：`[7]`。
- 中文：`[3, 3, 4, 2, 1, 3, 2, 2, 1, 2, 3, 2, 2, 3, 3, 3, 4, 3, 2]`。
- 代码：`[8, 8, 8, 5]`。

`confidence` 另保留 E16 模型 + 原生 N-2/N-1 滞后规则；其 SHA256 为 `c203b5d1a29f85603b630759b687430a4be3a927e4ee7ceec92c64d7d515f7cb`。这个模式不代表现有本轮 E17 组合，早期测出的39.789 token/s只属于该配置。

## 状态、停止与生命周期

每轮验证 pending token + 草稿前缀，`greedy_result` 决定输出与已提交行数。拒绝后特征不进入 context，纠正/bonus token 保持 pending。随后 `process_MTPInfo` → `process_outputs` 配对每次 schedule，包括 prompt chunks、取消与空计划。退出恢复 NEXTN 环境/Python缓存，按所有权清理 Gloo group。

- 本地204项：192通过，12因依赖缺失跳过；K1 HTTP、取消/断线与资源隔离回归通过。
- 真实 C++ Scheduler + CPU Toy Target：19组，131次状态检查。新增最终验证长度调整、全接受及首/中/末拒绝后的继续生成；同时保留固定 K、双请求、prompt分块、EOS、长度1、取消/代次复用检查。
- 整模 Qwen/Q4 GPU：28组，**1,362行完整 committed logits、354次有效状态检查精确一致**。三个输入分别覆盖六个自然模式；另有五组 oracle-only 草稿边界、双请求 fixed7，以及 adaptive 长度1/三种取消与复用。受控草稿只用于边界，不计入自然接受率或性能结论。
- 旧 E17 独立循环在三个输入上逐轮宽度一致，并与相同 serial 输出一致。短样本不构成长上下文验证；768阈值的成本桶切换有本地契约检查。
- 在线计时另做48次完整输出/最终状态核对（12预热+36计时）。真实 loopback SSE 与显式取消运行在 adaptive 模式，输出和 DONE/停止状态通过；取消使用受控 step 边界，不声称覆盖任意异步竞态。
- 全部终止检查无活动状态、原生队列清空及逻辑块归还。最终审计确认 group释放、环境恢复、worker READY，load_count=1；不重载权重、不调用 empty_cache。

## 同轮小样本测速

中文解释、Python函数、JSON三个提示，上限32/48/48，每种模式三次，四模式共36次。每种模式同样完整输出，合计 **309个 decode token**，九个请求各排除首 token。TPS按总 token/总时间计算；请求和TTFT列是九次之和。

| 模式 | decode token/s | 完整请求耗时之和 | TTFT之和 |
| --- | ---: | ---: | ---: |
| none | 23.230 | 15.566 s | 2.264 s |
| fixed7 | 38.701 | 10.331 s | 2.346 s |
| confidence | 40.731 | 9.945 s | 2.358 s |
| adaptive | 45.016 | 9.224 s | 2.359 s |

本轮 E17 adaptive 相对 fixed7 的 decode 点估计为 **+16.32%**，完整请求耗时变化 **-10.72%**；相对 E16原生滞后 confidence 的 decode 点估计为 **+10.52%**。只有三个受控输入与三次重复，无代表性负载或独立性能晋升门槛，不作一般收益承诺。无草稿差额不能归因于本次 Scheduler 接线。

计时包含实际草稿、特征、预测、原生调度、验证、提交及循环；模型/provider/native上下文构造、预热和serial核对在计时外。请求计时含submit/tokenize/prompt，decode从首输出事件到终止。首轮草稿在第一个 token 发布前预计算，进入TTFT；旧研究循环从首 token 后计入首草稿，阶段边界不同，不能直接横比历史decode TPS。这里同时报告完整请求耗时。GPU请求仍分别同步执行，不是联合batch。

## 复现与保留的失败

```bash
python3 scripts/k2_export.py results/raw/k2 results/k2/2026-10-23-postdraft-e17.json
python3 -m unittest discover -s tests -q
```

导出器校验最新 GPU/在线封存源码SHA、case数、旧循环宽度对齐、36次计时、48次状态核对、同worker/load_count与最后READY。旧结果按原封存保留，不能冒充当前源码测试。原始日志/失败traceback在Git忽略目录，公开文件保存脱敏摘要和SHA。

早期 CPU 双请求fixture误设为单槽，在GPU前失败并修正。早期在线harness恢复release时产生引用环，删除临时实例hook后重跑；旧报告使用修正后版本。新增 E17/最终长度接口的CPU、GPU、在线均重新封存验证。GPU参考与候选共用Target算术，不是独立整模数值证明；独立CLI未为验证再次加载大模型，真实HTTP以resident session验证。

跨请求GPU batch、异步流水、完整KV页映射、随机采样、长上下文和更广负载下新旧执行器性能比较仍待后续工作。K2不表示DeepInfer全部模块已适配。
