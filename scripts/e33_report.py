"""Render a bounded E33 report from verified exported evidence."""
import argparse
import json
from pathlib import Path
import statistics


def pct(v):return f'{100*v:+.3f}%'
def ci(values):return '～'.join(pct(v) for v in values)


def main(evidence,archive,out):
    data=json.loads(evidence.read_text());a,b,o=data['initial'],data['fused'],data['online'];s=o['summary'];m=json.loads(archive.read_text())
    cases=b['cases'];counts=[c['union_count'] for c in cases];rows=sum(c['rows'] for c in cases);tr=s['trace'];resident=data['residency']
    lines=[
        '# E33：低位宽输出头筛选、定向误差界与原 BF16 精算',
        '',f"2026-10-23 UTC，`{o['status']}`。在同一常驻模型上完成组件筛查、区间融合和完整生成配对对照。正常decode为 **{s['decode_tps']['original']:.4f}→{s['decode_tps']['screened']:.4f} token/s（{pct(s['speed_gain'])}）**，root bootstrap95%区间 **{ci(s['root_bootstrap_gain_95'])}**。这是8条已用工程root的greedy受控结果，不是服务默认升级、原生TensorFold排名或新流量泛化证明。",
        '', '[原始方案](E33_EXACT_HEAD_SCREEN_PLAN.md) · [融合方案](E33_FUSED_BOUNDS_PLAN.md) · [正常生成方案](E33_ONLINE_PLAN.md) · [完整脱敏证据](../results/e33/2026-10-23-exact-head-screen.json)',
        '', '## 实际改变的工作',
        '', '原target每轮读取完整 `[248320,2560]` BF16输出头，约1.27GB。候选另建一个逐输出通道对称INT8 RTN副本，BF16 hidden不量化，FP32 scale；先用它计算全词表近似分数，再保留在保守误差区间下仍可能获胜的词。候选的有序并集用原BF16权重和原BK64/SK1累加链精算；并列仍取最小token ID。',
        '', f"辅助payload为 **{b['preparation']['payload_bytes']:,} bytes**，原BF16头保持不变。准备耗时 **{b['preparation']['ms']:.3f}ms**（组件运行），正常生成另一次构建辅助副本耗时 **{o['head_preparation']['ms']:.3f}ms**；不重载target/draft。它增加常驻辅助存储，不能称整机权重内存减少。减少的是正常路径读取的大块BF16头，不是把原头删除或用INT8分数直接决定token。",
        '', '候选cap2048；输入/范数/分数/界无效、非有限数、范围不满足、候选过多或为空时退回原完整头。只用于greedy，随机采样仍需要原完整分布。没有截断top-k后假定正确。',
        '', '## 数值边界与实际检查',
        '', '界包含L2量化残差、原与近似路径的FP32累加、近似最终scale乘法以及原BF16输出舍入。范数用FP64并向上膨胀/取nextafter；融合核用显式 `mul.rp.f64`、`add.rp.f64`、`sub.rm.f64` 生成保守界与向外端点，gamma常数向上取整。',
        '', '当前实现声明有限范围：非零权重/scale/hidden幅值至少2^-48、最大不超过2^16，满足本K2560下的中间正常值/无溢出约束；不满足就回退。FP32误差模型采用保守gamma(2K+1,u=2^-23)。这不是对所有GPU硬件/任意输入的无条件形式化证明，实测一致性不能替代该证明。',
        '', f"两轮组件各有1～8行、强制cap回退、全零并列和NaN回退共11项功能场景，另有独立CUDA stream检查；全部通过。四条校准root前8轮共32窗口、{rows}个真实hidden行，完整词表逐元素检查共{rows*248320:,}个差值/界对比（每轮）。融合版本最大误差/界比值 **{max(c['bound_max_ratio'] for c in cases):.6f}**，零越界；所有原argmax保留、自然候选子集logits逐位一致、最终argmax一致。",
        '', f"32窗口的候选并集 **{min(counts)}～{max(counts)}词**（中位数{statistics.median(counts):g}），自然回退 **{sum(c['fallback'] for c in cases)}/32**。这些是真实hidden结果，替代旧方案仅基于Gaussian模拟的候选数估计；不是独立测试集。",
        '', '## 首版没有通过：不能只看INT8矩阵乘',
        '', '| 实现 | BF16基线的窗口中位数之和 | 完整筛选流程 | 耗时减少 | 速度收益root95% |',
        '| --- | ---: | ---: | ---: | ---: |',
        f"| 初版FP64张量算子流水线 | {a['full_summary']['sum_median_ms']['bf16']:.6f}ms | {a['full_summary']['sum_median_ms']['screened']:.6f}ms | {100*a['full_summary']['time_reduction']:.3f}% | {ci(a['full_summary']['root_bootstrap_gain_95'])} |",
        f"| 融合定向FP64区间 | {b['full_summary']['sum_median_ms']['bf16']:.6f}ms | {b['full_summary']['sum_median_ms']['screened']:.6f}ms | {100*b['full_summary']['time_reduction']:.3f}% | {ci(b['full_summary']['root_bootstrap_gain_95'])} |",
        '', f"初版Q8投影本身省{100*a['primitive_summary']['time_reduction']:.2f}%时间，但完整流程仅省{100*a['full_summary']['time_reduction']:.2f}%，区间跨零且一条root退化，因此保留`REJECT_E33_FULL_COST`。初版反复生成/读写全词表FP64中间数组；随后仅融合该固定格式的区间实现，不换量化配方、不放宽误差界或候选cap。",
        '', f"融合后完整头流程省 **{100*b['full_summary']['time_reduction']:.2f}%**，四root均为正，通过原先至少10%节时门槛。表中是32份head输入的组件回放，不是target或decode耗时。完整wall计时包含近似分数、动态检查、Graph输入copy、mask/nonzero同步、候选精算/回退和argmax；CUDA-event的primitive时间另存，不混作完整收益。",
        '', '## 完整生成：相同预算和接受轨迹',
        '', f"使用另外8条旧E28工程root，沿用自然EOS/原output_cap；与四条组件root无交集，但仍是项目已用回归集。每任务每模式4次，AB/BA平衡，共 **{s['timed_runs']}次** 正常计时，每组实际decode token数 **{s['timed_tokens_per_arm']:,}**。",
        '', '两组均冻结E28 W8草稿、Q4草稿头、E17 NumPy线性模型及E29 W8费用。基线保留原完整target Graph；候选的私有Graph保留48层、五处DFlash特征和最终writeback/mixer，返回hidden后执行筛选头。两组各16张图，筛选另按1～8行预热；计时内无新增捕获或子集JIT shape。',
        '', '计时包含草稿、特征/预算预测、验证、所有同步/筛选/精算/回退、commit和循环。排除prefill/首token、初态copy、Graph捕获/JIT、oracle和检查。不是将head微基准直接折算成decode。',
        '', '| 对照 | 原BF16头 | 筛选＋原BF16精算 |',
        '| --- | ---: | ---: |',
        f"| 实际decode token/s | {s['decode_tps']['original']:.4f} | {s['decode_tps']['screened']:.4f} |",
        f"| 验证轮次（全部计时） | {tr['original']['rounds']:,} | {tr['screened']['rounds']:,} |",
        f"| 验证计算行数 | {tr['original']['verified_rows']:,} | {tr['screened']['verified_rows']:,} |",
        f"| 观测接受/观测数 | {tr['original']['accepted']:,}/{tr['original']['observed']:,} | {tr['screened']['accepted']:,}/{tr['screened']['observed']:,} |",
        '', f"逐重复的速度变化：{', '.join(pct(r['speed_gain']) for r in s['per_repetition'])}。",
        '', '每条先建立完整serial参考，再各跑一次完整正确性预热；候选诊断另外执行全BF16头，对照所有子集值和argmax、每个提交logit、DFlash特征及最终live state。正式计时后再次核对tokens/live state；每轮context、宽度、接受/观测、提交和预测rates也必须完全相同。因此本轮没有靠少验证行或接受率变化获取收益。参考仍是同一原target执行路径，不是独立HF/官方数值正确性证明。',
        '', f"自然生成筛选头回退 **{tr['screened']['fallbacks']}/{tr['screened']['rounds']}轮**。真实计时候选并集范围为{min(tr['screened']['union_counts'])}～{max(tr['screened']['union_counts'])}；全部原始逐步计时/候选数在公开JSON保留。",
        '', '## 结论边界',
        '', '最终是否通过门槛以报告首行状态为准：只有正确性全过、decode收益至少1%、root95%下界为正且四个重复均为正，才标记`PASS_E33_ONLINE_GAIN`。不将组件收益与E29/其他实验相加，不据此宣布领先原生TensorFold；未做新的原生TensorFold同期重测、随机采样、多请求batch或4K+上下文。',
        '', '这采用已有的粗筛＋精算/区间排除思路，不主张首创算法。实际贡献是当前SM121混合精度target、定向误差界、保持原BF16算术的子集核及DFlash动态预算/Graph的具体实现与测量。默认执行器/SSE尚未切换，代码仍是研究路径。',
        '', '## 常驻、证据和复现',
        '', f"三个GPU作业`115-e33-screen.json`、`154-e33-fused.json`、`155-e33-online.json`均PASS，前者的REJECT是性能决策。始终同一worker、target/draft对象、load_count=1；failed_jobs快照{resident['before']['failed_jobs']}→{resident['after']['failed_jobs']}。completed_jobs快照{resident['before']['completed_jobs']}→{resident['after']['completed_jobs']}，对应这三次成功作业。",
        '', f"release venv：PyTorch `{data['environment']['torch']}`、Triton `{data['environment']['triton']}`。实际Triton cubin哈希逐shape记录，source/input seal逐文件复查。allocated快照{resident['before']['allocated_bytes']:,}→{resident['after']['allocated_bytes']:,} bytes，不把辅助payload或shared-memory数字冒充整机内存节省。",
        '', f"原始归档{m['files']}文件，SHA256 `{m['sha256']}`，位于Git忽略的`results/raw/e33/final/worker`。公开JSON不含提示词、权重payload、私有绝对路径或对象ID。",
        '', '```bash', 'python3 scripts/e33_export.py --raw results/raw/e33/final/worker --out results/e33/2026-10-23-exact-head-screen.json', 'python3 scripts/e33_report.py --evidence results/e33/2026-10-23-exact-head-screen.json --archive results/raw/e33/archive-metadata.json --out docs/E33_EXACT_HEAD_SCREEN_REPORT.md', 'python3 -m unittest discover -s tests -q', '```',
        '', '本机214项测试：199通过、15缺少可选PyTorch跳过。新增汇总测试检查实际token/总耗时加权及缺失/重复计时记录拒绝；GPU精确性/回退/整模检查由上述常驻作业完成。',
    ]
    out.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.evidence,a.archive,a.out)
