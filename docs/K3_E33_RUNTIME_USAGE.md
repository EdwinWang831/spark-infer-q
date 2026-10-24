# 完整 E33 优化配置

在准备好的K2/K3环境中启用整套优化：

```bash
E3_ROOT=/path/to/checkout/scripts bash /path/to/prepared/e3_run.sh \
  k2_serve.py --root /path/to/prepared --checkpoint /path/to/target \
  --profile e33 --execution graph --mode adaptive --capacity 1536 --port 18083
```

此profile固定使用E33融合辅助筛选＋原BF16精算、Target Graph、E28 W8草稿MLP/Q4草稿头、E17 NumPy线性预测及E29 W8费用。原BF16权重保留；新增辅助存储，不声称整机内存减少。模型/cost/W8源和二进制身份校验不匹配时明确拒绝，不能用其它artifact冒充E33配置。`--model`不适用于冻结profile。

只允许adaptive单请求greedy，固定草稿不恢复。Graph只是执行方式，实际验证宽度仍由当前草稿的预测概率和上下文成本决定。首轮草稿在第一个token后才生成，完整进入decode；这一点与旧baseline profile的首轮草稿进入TTFT不同。

物理Target/DFlash容量2048，推荐逻辑1536以保留lookahead；`--capacity`仍限制prompt+output准入。最大已测提示1024/output_cap384，未覆盖4K+、随机采样或联合GPU batch。

session API：

```python
from spark_infer_q.kaifra_optimized import optimized_provider, optimized_target
from spark_infer_q.kaifra_speculative import speculative_engine

with session.torch.inference_mode():
    provider, metadata = optimized_provider(session)
    execution = optimized_target(session)
try:
    execution.warmup()
    with speculative_engine(native, execution, provider, mode='adaptive',
            capacity=1536, eos_ids=session.eos,
            prior=session.calibration['conditional_prior'], tokenizer=session.tokenizer) as engine:
        engine.submit('example', [{'role': 'user', 'content': '你好'}], max_tokens=128)
        for event in engine.drain():
            print(event.token_ids, event.finish_reason)
finally:
    execution.close()
```

`native`为NEXTN=7 NativeScheduler，owner/lease规则同K3。筛选路径显式返回greedy预测，诊断时可重算完整原头；不会把hidden或候选logits当作全词表logits。Graph缺失bucket/state不兼容时回退完整eager头；筛选范围/候选cap/非有限值不满足时也回退原完整头。主机完成同步仍保留。

启动准备W8包、辅助头和图后才READY。新输入可能触发新的动态候选子集JIT shape；本轮steady-state测速在每任务正确性预热后执行，并断言计时内无新增capture/JIT，不能当作新请求冷启动延迟。HTTP一次一请求，未部署公网服务。

`--profile baseline`保留之前的K2/K3对照配置，默认仍为baseline；测试结果见[迁入报告](K3_E33_RUNTIME_REPORT.md)。
