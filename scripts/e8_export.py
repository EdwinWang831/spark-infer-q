"""Check complete E8 prompt evidence and export the portable result."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    report=json.loads(a.raw.read_text());assert report['status']=='PASS_PROMPT_ONLY'
    repo=Path(__file__).resolve().parents[1]
    for name,sha in report['harness_sha256'].items():
        path=repo/name if name.startswith('spark_infer_q/') else repo/'scripts'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,name
    cases=report['correctness'];timings=report['timings']
    assert len(cases)==81 and len(timings)==160
    assert Counter(c['width'] for c in cases)=={2:27,4:27,8:27}
    assert all(c['logits_features_state_exact'] and c['continuation_steps_exact']==8 for c in cases)
    assert len({(c['id'],c['width']) for c in cases})==81
    ids={c['id'] for c in cases if not c['id'].startswith('tail-')};assert len(ids)==10
    for width in (1,2,4,8):
        medians=[];tokens=0
        for name in ids:
            rows=[t for t in timings if t['id']==name and t['width']==width]
            assert len(rows)==4 and {r['repetition'] for r in rows}==set(range(4))
            assert len({r['pending_token'] for r in timings if r['id']==name})==1
            assert all(r['prompt_ms']>0 for r in rows)
            medians.append(statistics.median(r['prompt_ms'] for r in rows));tokens+=rows[0]['prompt_tokens']
        summary=report['summary'][str(width)]
        assert abs(sum(medians)-summary['sum_median_prompt_ms'])<1e-6
        assert summary['prompt_tokens']==tokens
    report['repeat_diagnostics']=[]
    for rep in range(4):
        totals={str(w):sum(r['prompt_ms'] for r in timings if r['width']==w and r['repetition']==rep) for w in (1,2,4,8)}
        report['repeat_diagnostics'].append({'repetition':rep,'sum_prompt_ms':totals,'width8_speedup':totals['1']/totals['8']})
    report['analysis_harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['limitations']=['prompt-only target experiment; no Q4 draft model resident and no full speculative generation',
        'host prompt time excludes tokenization and wire transport; not TTFT',
        'short reused prompts on one GPU; no quality, new-input generalization or native speed-lead claim',
        'default execution remains unchanged pending complete-generation integration']
    output=json.dumps(report,indent=2,allow_nan=False)+'\n'
    for private in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/'):assert private not in output
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(output)
    print(json.dumps({w:{k:v for k,v in s.items() if k!='tasks'} for w,s in report['summary'].items()},indent=2))


if __name__=='__main__':main()
