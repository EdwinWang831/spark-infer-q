# E42：1,024 行 prefill 分块的真实增量

2026-10-24。**通过数值和成本门槛，开放 1,024 行显式选项，默认仍为 512 行。**
对照是 E41 已优化的大块 prefill，两臂都使用 Graph/E33/W8/E17/E29；不是旧 rows8。
长输入 prefill 合计耗时减少 **5.80%**；两个长输入的完整请求分别减少 **4.58% / 1.79%**。
这轮没有证实新增 decode 收益，也没有新内核、量化或训练变化。

[事前方案](E42_WIDER_PREFILL_PLAN.md) · [公开证据](../results/e42/2026-10-24-wider-prefill.json) ·
[模块用法](E40_PREFILL_EXECUTOR_USAGE.md)

## 对照配置与数值

同一常驻 DGX Spark、target/draft 对象及冻结外部 TensorFold 源。唯一实验变量是
PrefillExecutor 的 chunk_rows=512/1024。两臂 physical capacity2048、logical1792，
完整请求的 scheduler token_budget 都为1024；decode 使用既有 Graph/E33 BF16最终裁决、
W8草稿MLP/Q4草稿头、BF16 attention/selector、E17 NumPy及冻结E29 W8成本。
Greedy 输出上限64，首轮草稿延迟到首token后并计入decode。

更大块可能合并专家分组尾部并摊薄 stage/层执行的重复成本；关键矩阵路径与专家tile16
不变。本轮没有设备计数器或独立阶段剖析，**没有把潜在原因认定为已测归因**。

五条输入 English64、code513、code1024、mixed1025、Chinese1536 的完整末行logits、
全部4,162行五层草稿特征、有效GDN/QSA/PLE状态均逐位一致；覆盖[512,1]→[513]、
[512,512]→[1024]、[512,512,1]→[1024,1]和[512,512,512]→[1024,512]边界。
每条再运行width1–8，奇数完整提交、偶数半提交：40窗口、185行完整logits及状态通过，
16种width/parity桶都有实际Graph replay，无fallback。比较的是当前有效循环状态，
不同分块次数造成的物理ping-pong槽位差异不属于状态错误。

## Prefill 本体

每臂每例先预热一次，长输入各六轮AB/BA，短输入四轮，共32次正式prompt。
Provider/workspace准备在外；state创建、stage、48层、特征、末行head、commit及完成同步
在内；每次在计时外核对完整输出和有效state。下表为每次均值：

| 输入 | chunk512 | chunk1024 | 耗时减少 |
| --- | ---: | ---: | ---: |
| English64 | 159.69 ms | 162.09 ms | −1.50% |
| Code1024 | 853.66 ms | 788.11 ms | 7.68% |
| Mixed1536 | 1,256.28 ms | 1,199.38 ms | 4.53% |

两条长输入相同15,360 tokens/臂，总wall **12,659.63→11,924.93 ms，减少5.80%**，
六轮配对汇总全部更快。达到预定至少3%、至少4/6轮更快、每条长输入不慢超过2%、
短输入不慢超过3%的门槛。短输入没有收益，不能推导越大分块越好。

## 完整优化请求

每例每臂一次预热，随后四轮AB/BA；共8次预热和32次正式请求。
输出token、事件位置、完整已提交特征上下文、最终有效状态及decode的接受/宽度/概率/
head轨迹一致。Prompt内部的分块轨迹会随chunk_rows改变，单独保留而不要求逐段相等。
正式请求没有新增Graph capture/head JIT形状，Graph/head fallback均为0。

以下是每例四次均值，单位毫秒：

| 输入 | 输出tokens | TTFT 512→1024 | 完整请求512→1024 | 请求耗时减少 |
| --- | ---: | ---: | ---: | ---: |
| English23 | 64 | 111.36→113.03 | 1,196.55→1,205.46 | −0.74% |
| Chinese23 | 51（EOS） | 116.64→120.34 | 1,974.98→1,966.86 | 0.41% |
| Code1024 | 64 | 843.49→788.36 | 1,561.74→1,490.24 | 4.58% |
| Mixed1536 | 64 | 1,285.50→1,235.43 | 1,990.21→1,954.52 | 1.79% |

Code四轮总请求均更快；mixed四轮有一轮慢0.23%。相同972输出tokens/臂的汇总wall为
26,893.90→26,468.30ms，减少1.58%，不能作为任意流量的总体吞吐结论。
所有长文本均为受控重复并截断文本，不是任务质量或推广性评测。

计时从token API submit前开始，到drain后CUDA同步完成；profile/engine准备和正确性检查
在外，TTFT使用原生API首输出事件。首token后残差定义为“完整wall减原生TTFT”，含少量
submit/事件边界开销，不是纯GPU时间或外部客户端的token间隔。排除每次首token后，
相同956个decode tokens/臂耗时17,465.94→17,439.63ms，即 **54.74→54.82 tok/s（+0.15%）**。
Code残差略好、mixed略差，样本不足以证实decode增益，不做新的decode提速声明。

## 生命周期、清理与工程接入

三种取消与ID复用通过：waiting、prepared prompt、prepared decode；后续一token请求可用，
lease和backend states释放。本机中文短输入SSE返回24个reference-prefix tokens/length；
取消返回1token/cancelled。HTTP仅为功能测试，没有HTTP测速或公网部署。

420-e42-study-v1与422-e42-audit-v1均PASS。同一worker319、load_count1、同一常驻对象；
原外部函数代码/no_grad包装、native group/environment、stage模块释放通过独立审计。
Allocated **78,030,942,208→78,030,942,208 bytes**，未重载权重或清空缓存。

GPU测量时通过继承原provider、仅设max_chunk_rows=1024的实验子类开放更大块。
测量和审计完成后，生产provider只改一行准入上限512→1024，所有run/allocate算术函数
逐字节不变；PrefillExecutor的默认chunk_rows=512也不变。导出器精确检查该单行差异并
分别记录测量版/当前版源hash，拒绝其他库改动。公开结果中的study字段反映测量当时
production_bound_changed=false，顶层library_policy明确记录之后的上限开放。
没有把后续单行上限修改描述成另做过一次GPU性能实验。

显式设置 `PrefillExecutor(provider, chunk_rows=1024)` 并保证native scheduler的
`token_budget>=1024` 才能产生1024行prompt计划；预算会影响调度公平性，本轮只验证单请求。
不自动切换默认服务或其他工作区。

## 证据与边界

原始封存227文件、753,511bytes，SHA256
`312d8a73c8f50d747546f9ae9a2e1315c8ca2b27ad12c4b78e125ce04d536183`，
位于忽略目录 `results/raw/e42/2026-10-24-final/`，包含stage/manifest/job、冻结成本/预测器、
W8源及binary和外部源，不含模型权重。封存源码及审计链、数值/计时/工作量和失败门槛
由导出器重算核对，拒绝篡改。

```bash
python3 scripts/e42_export.py \
  --raw results/raw/e42/2026-10-24-final/worker \
  --out results/e42/2026-10-24-wider-prefill.json \
  --study-version study-v1 --study-job 420-e42-study-v1.json \
  --audit-version audit-v1 --audit-job 422-e42-audit-v1.json
python3 -m unittest discover -s tests -q
```

本地333项测试：314通过，19项可选依赖跳过；包含13项E42门槛/工作量/布尔类型/
失败诊断及上限修改证据检查。CPU fixtures仅验证门槛，不替代GPU证据。

最大测试输入1,536，物理容量2,048；未验证1,024行从任意非零前缀开始的增量调用、
4K+、vision、随机采样、多请求batch或最新原生TensorFold排名。Prefill仍eager，Graph
覆盖decode；加载器、状态布局和多个CUDA算子仍依赖外部TensorFold。
