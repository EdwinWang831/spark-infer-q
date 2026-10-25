# E43：单独移植专家指令流水，没有新增收益

2026-10-24启动，2026-10-25收尾。**不采用。** 只把 Mia 共享输入tile中的MMA/FMA
交错调度移入原16行专家内核，保持plan、4-warps、depth2、BF16激活、NVFP4权重、
K16 scale-FMA顺序与F32 down不变。它通过组件精确性检查，但完整MoE成本基本持平。

[事前方案](E43_PREFILL_PIPELINE_PLAN.md) · [公开结果](../results/e43/2026-10-25-prefill-pipeline.json)

44项边界检查覆盖11种行数至1024、连续/stride、F32/BF16 down、guard、Graph重放及
另一caller stream。English64/code512/code1024/mixed1024 × 层0/23/47的router logits、
pick、wts、act、y全部精确；观测后的完整控制prompt logits/features/state也一致。
这是选定MoE替换检查，不是候选替换全部48层的状态或性能验证。

| 12份完整MoE的中位数之和 | 原版 | 流水候选 | 耗时变化 |
| --- | ---: | ---: | ---: |
| 热复用 | 54.217 ms | 54.130 ms | −0.160% |
| 128 MiB缓存扰动 | 53.931 ms | 54.008 ms | +0.143% |

每份两种条件、六轮AB/BA，288个完整MoE CUDA-event区间。计时包括router/select/plan、
路由及共享专家、输出slots；预热/捕获/检查/扰动在外。扰动不保证缓存完全驱逐。
未达预定至少降低5%的门槛，不运行候选完整prompt/request性能测试，不择优重跑。

当前binary资源：原gate/up128寄存器/线程，候选124；down均96。资源降低没有转化为
可采用的速度收益。没有硬件stall计数器，不能据此断言所有指令流水都无效。
后续 [E44](E44_MIA_PREFILL_TILE_REPORT.md) 是完整共享输入tile适配，属于不同候选。

## 审计异常完整保留

本轮复用已有worker46235/load_count1；它与历史E42进程不同，没有由本轮重载模型。
434首版审计错误假定新进程已加载DeepInfer模块，因ModuleNotFoundError失败；组件阶段
没有启动native scheduler。440改为检查现有模块后，原函数/包装/模块清理检查继续通过，
但在内存门槛失败：**77,359,853,568→77,393,408,000 bytes，多33,554,432 bytes**。

期间有其他共享串行队列任务。448只读复核保留同一数值，返回
`E43_AUDIT_MEMORY_UNRESOLVED`：原算子、native group/environment及stage模块检查通过，
**早期总allocated恢复门槛仍未通过，32 MiB归属没有证实**。没有清空全局cache或删除
其他工作区对象来消除差额，也没有把此复核称为完整审计PASS。

公开证据明确标记 `E43_REJECTED_WITH_UNRESOLVED_MEMORY_AUDIT` 和
`memory_gate_passed=false`。E44以后来的稳定基线独立验证；它通过自己的门槛不覆盖
E43这条历史失败记录。没有因异常修改性能门槛或推广候选。

最终归档459文件、1,982,114bytes，SHA256
`848098ce0bb067affceff123aa1a8a13ef13efd0f84c3277e277c684b69f7d82`，
位于忽略目录 `results/raw/e43/2026-10-25-final-v3/`。含434/440失败记录、448复核、
执行源码/manifest、生成CUDA、control/candidate binary和资源信息；此前归档也保留。

```bash
python3 scripts/e43_export.py \
  --raw results/raw/e43/2026-10-25-final-v3/worker \
  --out results/e43/2026-10-25-prefill-pipeline.json \
  --build-version build-v1 --build-job 430-e43-build-v1.json \
  --component-version component-v1 --component-job 432-e43-component-v1.json \
  --audit-version inspect-v1 --audit-job 448-e43-inspect-v1.json
```
