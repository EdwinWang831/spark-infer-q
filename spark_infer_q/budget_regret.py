"""Training-only decision-regret mining under a frozen, approximate cost model."""
import math


def budget_regret(emitted, costs_ms, width, reference_rate, max_rows=None):
    """Nonnegative Dinkelbach action regret in equivalent milliseconds.

    The reference rate is fixed for the cohort. This is an offline supervision
    signal under its supplied cost model, not measured recoverable wall time.
    """
    if len(emitted)!=len(costs_ms) or not emitted:
        raise ValueError('one emitted count and cost per action required')
    limit=len(emitted) if max_rows is None else min(max_rows,len(emitted))
    if not 1<=width<=limit or not math.isfinite(reference_rate) or reference_rate<=0:
        raise ValueError('invalid feasible width or reference rate')
    if any(not math.isfinite(c) or c<=0 for c in costs_ms):raise ValueError('invalid costs')
    if any(not math.isfinite(e) or e<1 for e in emitted):raise ValueError('invalid progress')
    utilities=[e/reference_rate-c for e,c in zip(emitted[:limit],costs_ms[:limit])]
    best=max(range(limit),key=lambda i:utilities[i])
    return dict(oracle_rows=best+1,regret_ms=max(0.,utilities[best]-utilities[width-1]),
                predicted_rows=width,extra_rows=max(0,width-best-1),missing_rows=max(0,best+1-width))


def regret_weight(regret_ms, scale_ms, cap=4.):
    if any(not math.isfinite(x) for x in (regret_ms,scale_ms,cap)) or regret_ms<0 or scale_ms<=0 or cap<1:
        raise ValueError('invalid regret weight parameters')
    return 1+(cap-1)*min(1.,regret_ms/scale_ms)


def mine_training_positions(records, quotas, per_prompt=8, family_fraction=.2):
    """Training-only hard pool with language, prompt, and family diversity caps."""
    if any(r['split']!='train' for r in records):raise ValueError('held-out rows cannot enter a training pool')
    chosen=[]
    for language,quota in quotas.items():
        prompts={};families={};limit=max(1,math.ceil(quota*family_fraction));selected=0
        candidates=sorted((r for r in records if r['language']==language),key=lambda r:(-r['regret_ms'],r['id'],r['position']))
        for row in candidates:
            if selected>=quota:break
            if row['regret_ms']<=0:continue
            if prompts.get(row['id'],0)>=per_prompt or families.get(row['family'],0)>=limit:continue
            chosen.append(row);selected+=1
            prompts[row['id']]=prompts.get(row['id'],0)+1;families[row['family']]=families.get(row['family'],0)+1
    return chosen
