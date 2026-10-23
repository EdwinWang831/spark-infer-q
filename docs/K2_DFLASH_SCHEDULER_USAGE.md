# K2 原生调度推测解码

`kaifra_speculative.py` 在 K1 请求 API 上接入 Q4 DFlash2。请求计划和输出账本由真实 DeepInfer C++ Scheduler 管理，验证宽度可用原生滞后策略或现有 E17 本轮动态预算；Target 沿用 TensorFold 算术。运行时仍为同步、greedy、纯文本；K2 推测解码当前只接受单请求。

| 模式 | 原生调度行为 | 范围 |
| --- | --- | --- |
| none | K1 target-only，NEXTN=0 | 无草稿 |
| lagged | 原生 MTPVarlen + 按历史观测更新的条件接受率 | 单请求，显式单请求成本曲线 |
| adaptive（CLI 默认） | E17 线性预测器 + 现有本轮 post-draft BudgetProbe；原生 ZOS 接口最终确定验证长度 | 单请求，按上下文切换冻结 short/long 成本 |
| confidence | 原生 MTPVarlen + 已冻结 E16 七深度线性置信头 | 单请求，显式单请求成本曲线 |

固定草稿运行模式已移除；`adaptive` 为默认，另外可选 `lagged`、`confidence` 动态对照及 `none` Target-only 对照。七个草稿是候选上限，实际验证宽度由预算器决定。

## 复用常驻 session

```python
from spark_infer_q.native_scheduler import NativeScheduler
from spark_infer_q.kaifra_speculative import session_provider, speculative_engine

with NativeScheduler('/workspace/e1/DeepInfer',
                     '/workspace/e1/build-release/lib/deepinfer', max_drafts=7) as native:
    provider, curve, metadata = session_provider(session, 'adaptive')
    with speculative_engine(native, session.target, provider,
            eos_ids=session.eos, prior=session.calibration['conditional_prior'],
            mode='adaptive', capacity=256, tokenizer=session.tokenizer,
            curve=curve) as engine:
        handle = engine.submit('example', [{'role':'user', 'content':'你好'}], max_tokens=32)
        for event in engine.drain():
            print(event.token_ids, event.text, event.finish_reason)
```

`session_provider` 只复用已加载的 DFlash/Q4 权重，另建有界的 context 投影缓存；不训练、不下载。adaptive 默认读取已有 `artifacts/e17-selected-model.json`，保留 E17 experimental 身份及 unweighted_f1 候选，不意味着新收益晋升。需要可导入的冻结 `spark_q_e3_varlen` 扩展，SHA256 `722b6ea835fa3e7e349af7f22ac727045a65af487a35f6709f3838e066bd678e`。confidence 默认读取已有 `artifacts/e16-selected-model.json`，也可显式传入已通过门槛的七深度 artifact。`metadata` 返回实际 model SHA256。特征组、标准化、校准和线性权重原样保留，累计概率转成条件概率后交给原生发布函数。

动态预算使用成本曲线1～8行，每行均计入已付出的完整草稿费用，包含 width1。默认 lagged 使用冻结 E6 单请求曲线，confidence 使用所选 artifact 的 short 成本桶；调用者可明确选择 long。成本仍是估算，原生 feedback 会更新估时，不代表已测得上下文无关或跨请求 batch 成本。

adaptive 将本轮累计概率同时传入已有 BudgetProbe 的 current/previous 字段，候选宽度受剩余输出限制；context<768 使用 short，否则 long。原生 Scheduler 保守预留八行，公开 `update_metadata_for_sampling_zos` 更新最终 q，输出账本按此 q 提交；预留 token 数保守保留。它使用冻结成本，不启用尚未晋升的 E27 历史成本修正。

原生 MTPVarlen 按 N-2 估计总预算、按 N-1 分配宽度，因此 confidence 模式与旧研究执行器的本轮 post-draft 决策不等价。初始缺少历史时可能先选一行。历史预测器的约10%结果不能直接移植为本运行时的速度承诺。

## 位置与提交

每轮执行的是一个 pending token 加草稿前缀。`greedy_result` 决定输出和 Target 提交的行数，拒绝后的特征不进入 context；最终纠正/bonus token 仍为 pending。随后依次调用原生 `process_MTPInfo` 和 `process_outputs`，让原生轮次、草稿、输出及实际设备位置一致。原生 CPU metadata slot 不代表已接入 DeepSeek GPU MTPInfoPool 或 sampler。

逻辑 capacity 是 prompt+输出的准入边界。另要求 Target 容量至少 capacity+14，DFlash context 至少 capacity+8，以满足原生最大验证/下一轮草稿 lookahead。不能直接把 Target 的完整物理容量当作逻辑准入上限。

取消/代次/输出事件与 K1 相同，`text` 为累计快照。`engine.last_step_metrics` 只保留最近一次调度的主机分段计时、验证行数和输出数；需要历史时由调用者保存。它不是纯 GPU 计时，也不能将等待区间相加作为性能上界。

当前在完成 prompt 时就生成下一轮草稿，再发布第一个 token，因此首轮草稿计算进入 TTFT。比较 decode 时必须同时查看 TTFT 和完整请求耗时，避免把阶段前移当成整体加速。

## loopback SSE

```bash
E3_ROOT=/path/to/checkout/scripts bash /path/to/prepared/e3_run.sh \
  k2_serve.py --root /path/to/prepared --checkpoint /path/to/target \
  --mode adaptive --capacity 256 --port 18083
```

独立入口只加载一次 Target/DFlash，并复用 K1 的 `/health`、`/v1/chat/completions` 和 `/cancel`。每个 server 固定一种模式，一次只允许一个 HTTP generation。已有常驻 worker 时通过 session API 复用模型；本轮验证没有另起进程重载权重，也没有将此服务部署为公网服务。

## 验证

```bash
python3 -m unittest discover -s tests -p 'test_kaifra*.py' -v
```

封存源码目录和 manifest.json 交给既有 worker：`k2_native_validate.run` 做真实 C++ / CPU 契约检查，`k2_gpu_validate.run` 做整模输出/logits/state 检查，`k2_online_validate.run` 做小规模计时和真实 loopback SSE 检查。三者都只接收 `source_root`，不重载 session。受控 oracle 草稿只用于拒绝/提交边界，不进入速度或自然接受率结果。
