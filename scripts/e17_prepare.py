"""Diagnose existing errors and seal training-only reweighting; never refit on E16 test."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from spark_infer_q.budget_regret import budget_regret,regret_weight,mine_training_positions
from spark_infer_q.confidence_head import choose_rows
from spark_infer_q.local_policy import ProgressModel


def prepare(root,out):
    out.mkdir(parents=True,exist_ok=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    trained=json.loads((root/'e16-training.json').read_text());assert trained['status']=='PASS_E16_TRAINING'
    info=json.loads((root/'e16-selected-model.json').read_text());assert trained['dataset_sha256']==info['dataset_sha256']
    model=ProgressModel(info['metadata'])
    # Historical calibration reference only; no test/EOS oracle information enters features.
    reference=trained['evaluations']['calibration']['oracle']
    indexed=[];sources={}
    for index in range(12):
        meta=json.loads((root/f'e16-collection-{index:02d}.json').read_text())
        path=root/meta['dataset'];assert sha(path)==meta['dataset_sha256'];sources[path.name]=sha(path)
        for line in path.open():
            rec=json.loads(line)
            if rec['split']!='train':continue
            for row in rec['rows']:
                c=info['costs']['long' if row['context_rows']>=768 else 'short'];paid=row['max_rows']>1
                base=[v+(c['native_draft_ms'] if paid else 0.) for v in c['verify_commit_ms']]
                extra=c['feature_increment_ms']+info['cpu_predict_ms'] if paid else 0.
                costs=[v+extra for v in base]
                rates=model.predict(row['x']);width=choose_rows(rates,costs,row['max_rows'])
                result=budget_regret(row['emitted'],costs,width,reference,row['max_rows'])
                indexed.append(dict(id=rec['id'],group=rec['group'],family=rec['family'],language=rec['language'],split='train',
                    position=row['position'],context_rows=row['context_rows'],remaining=row['remaining'],**result))
    scale=sorted(r['regret_ms'] for r in indexed)[int(.9*(len(indexed)-1))]
    assert scale>0
    for row in indexed:row['weight']=regret_weight(row['regret_ms'],scale)
    quotas=dict(en=205,zh=205,mixed=102)
    pool=mine_training_positions(indexed,quotas)
    def summarize(rows):
        total=sum(r['regret_ms'] for r in rows);positive=sorted((r['regret_ms'] for r in rows),reverse=True)
        return dict(positions=len(rows),prompts=len({r['id'] for r in rows}),groups=len({r['group'] for r in rows}),
            nonzero_regret=sum(r['regret_ms']>1e-10 for r in rows),approx_equivalent_ms_sum=total,
            top_10pct_share=sum(positive[:max(1,len(rows)//10)])/total if total else None,
            oversized=sum(r['extra_rows']>0 for r in rows),undersized=sum(r['missing_rows']>0 for r in rows),
            language_counts=dict(Counter(r['language'] for r in rows)),family_counts=dict(Counter(r['family'] for r in rows)))
    # Observed width costs can be compared to predicted costs, but counterfactual
    # actual width costs were not measured online; do not claim full attribution.
    online=json.loads((root/'e16-online.json').read_text());errors=defaultdict(list)
    for task in online['tasks']:
        for timing in task['timings']:
            if timing['mode']!=online['selected_arm']:continue
            for step in timing['steps']:
                c=info['costs']['long' if step['context_rows']>=768 else 'short']
                pred=c['verify_commit_ms'][step['width']-1];actual=step['verify_ms']+step['commit_ms']
                errors[(task['language'],step['width'])].append((actual,pred))
    observed=[]
    for (language,width),values in sorted(errors.items()):
        observed.append(dict(language=language,width=width,samples=len(values),
            predicted_ms=statistics.mean(b for a,b in values),actual_ms=statistics.mean(a for a,b in values),
            mae_ms=statistics.mean(abs(a-b) for a,b in values),signed_relative_bias=statistics.mean((a-b)/b for a,b in values)))
    weighting=dict(status='PREPARED_NOT_TRAINED',source_dataset_sha256=trained['dataset_sha256'],sources=sources,
        mining_model_sha256=sha(root/'e16-selected-model.json'),reference_rate=reference,p90_regret_scale_ms=scale,
        weight_cap=4.,normalization='normalize within each conditional head training subset; calibrate unweighted on fresh calibration prompts',
        training_only=True,rows=indexed)
    pool_artifact=dict(status='PREPARED_NOT_TRAINED',quotas=quotas,actual=summarize(pool),per_prompt_cap=8,family_fraction_cap=.2,rows=pool)
    report=dict(status='PASS_E17_PREPARATION',training=summarize(indexed),hard_pool=summarize(pool),
        weighting_scale_ms=scale,reference_rate=reference,observed_width_cost_diagnostics=observed,
        limit='offline equivalent regret uses approximate frozen costs; sums are not measured recoverable wall time; online alternative widths need paired measurements',
        source_sha256={'e17_prepare.py':sha(Path(__file__)),'spark_infer_q/budget_regret.py':sha(Path(__file__).resolve().parents[1]/'spark_infer_q/budget_regret.py')})
    for name,value in [('training-weights.json',weighting),('hard-pool.json',pool_artifact),('preparation.json',report)]:
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();prepare(a.raw,a.out)
