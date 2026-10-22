# 提议：按实际路由核算 target 内核的专家占用速率

状态：**PROPOSED，尚未执行，没有新的 GPU 结果。** 实际排期时再分配 E 编号。所有提议的状态汇总在 [EXPERIMENTS.md](EXPERIMENTS.md#proposals-without-results-yet)。

后续更新：[E10](https://github.com/EdwinWang831/spark-infer-q/blob/codex/e10-target-kernels/docs/E10_TARGET_KERNEL_REPORT.md)/[E11](https://github.com/EdwinWang831/spark-infer-q/blob/codex/e11-target-projections/docs/E11_PROJECTION_REPORT.md)（均在各自实验分支，尚未合入 main）的内核调参整段收益只有 +0.08% / +0.62%；[E12](E12_EXPERT_VERIFICATION_REPORT.md) 已逐步记录 Top-10 路由，并修正了 `routed_bytes` 的含义：pack 占用**不是** DRAM 流量，也不是它的下界（缓存命中会少读，tile 重读会多读）。E10 尝试 Nsight Compute 时驱动返回 `ERR_NVGPUCTRPERM`。在拿到硬件计数器之前，本方案只能给出指示性的占用速率，不能证明带宽利用率。

## 动机

[E9](E9_DECODE_PROFILE_REPORT.md) 三份采样 decode 共 72 步，全部为宽度 6。`nvfp4_expert_kernel` 1.826 s、`qmmf_kernel` 1.147 s、`_b16mm` 1.007 s，合计占内核累计时长 78.6%。E9 的结论是先查这些核，但没有每步实际读取的字节数，无法判断它们离带宽上限还有多远。

粗估（**未验证**）：expert kernel 每步约 25.4 ms；单专家 native pack 为 2,764,812 B（[E2](E2_NATIVE_REPORT.md)）。若每层平均有 30–45 个不同专家，选中 pack 的占用约 4.0–6.0 GB，折合约 157–235 GB/s，即标称 273 GB/s 的 57–86%。反过来，若每个 pack 每步恰好从 DRAM 读一次，按标称峰值计算，每层平均并集不可能超过约 52。这个区间太宽，不足以决定内核优化是否值得做，需要实测专家并集。

## 采集

- 沿用 E9 的仪器化方式：只在采样请求中，每个验证 step 记录行数、每行每层的 Top-10 专家 ID（复制到 host），以及带 step 序号的 NVTX 区间。保留未采样控制并报告扰动；正常时间分布仍以 E8 的未采样数据为准。
- 内核字节模型：
  - `nvfp4_expert_kernel`：Σ_layer 并集 × 2,764,812 B。这是每个 pack 计一次的占用，不是 DRAM 流量。
  - `qmmf_kernel`：在 wrapper 层记录每次调用实际权重对象的字节数（含 scale），不用 model-budget 的类别总量推断。
  - `_b16mm`：在调用点记录操作数形状，区分 target（hyper-connection 投影、输出头）与 draft backbone。
- 可达峰值：同一 session 单独测 ≥4 GiB 大张量只读 reduction 的带宽，取多次中位数。结果同时给出相对标称 273 GB/s 和相对实测可达值两种分母。
- 输出：每 step、每个内核家族的占用字节、时间和占用速率，按宽度分桶；专家并集随验证行数增长的曲线，即每加一行平均新增多少专家。

## 离线多请求估计

一行的路由只取决于该行自己的 hidden，与同 batch 的其他行无关。这个 batch 不变性假设需要在真正的 batch forward 上核对。在此假设下，可以把两个独立请求的行合并求并集，在 Kaifra K4（双请求 batch）完成前，先估计双请求的专家共享率：联合并集 ÷ 各自并集之和。这只是估计，不能当作双请求实测。

## 事前决策规则（执行前冻结；下列阈值为初始建议）

占用速率不是 DRAM 利用率，以下规则只作指示；拿到硬件计数器后以计数器为准。

- expert kernel 的占用速率达到实测可达带宽的 75% 及以上 → 降低内核调优优先级，重点转向减少读取字节（无效验证行、专家并集、草稿 backbone 精度）。
- 低于 50% → 内核或布局优化有实际空间，选一个核做单变量 A/B，并保留输出、logits 和状态验证。
- 介于两者之间 → 两条路线以单变量 A/B 决定。

## 边界

不改内核和权重。仪器化存在扰动。pack 占用不是 DRAM 流量的上界或下界，缓存命中和重读在没有硬件计数器时无法区分，因此结论只能写成“占用速率”，不能称为带宽利用率。不涉及多请求实测、长上下文或随机采样。

## 参考实现

[`spark_infer_q/moe_cost.py`](../spark_infer_q/moe_cost.py) 提供 `union_sizes`、`prefix_unions`、`joint_union_total`、`routed_bytes` 和 `achieved_bandwidth`。它们是标准库 host-only 代码，有单元测试，未在 GPU 上运行。`routed_bytes(48*512)` 与 E2 model-budget 记录的 67,948,019,712 B 一致；`achieved_bandwidth` 用在占用字节上时，得到的是占用速率。
