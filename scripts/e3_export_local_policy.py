#!/usr/bin/env python3
"""Validate completed frozen-policy measurements before public export."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from spark_infer_q.local_policy import MODES,mode_order


def export(source,output,repo):
    report=json.loads(source.read_text())
    assert report['status']=='PASS_BOUNDED_LOCAL_POLICY'
    for name,expected in report['harness_sha256'].items():
        path=repo/name if '/' in name else repo/'scripts'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    assert hashlib.sha256(json.dumps(report['models'],sort_keys=True).encode()).hexdigest()==report['frozen_model_sha256']
    splits=report['dataset'];tasks=report['tasks']
    assert (len(splits['train']),len(splits['calibration']),len(tasks))==(16,6,8)
    prompts=[r['prompt'] for r in splits['train']+splits['calibration']+tasks]
    assert len(set(prompts))==30
    repeats=report['measurement_contract']['repetitions']
    assert repeats>=7 and repeats%7==0
    counts={mode:Counter() for mode in MODES[1:]}
    draft_calls=Counter();predict_ms=Counter();draft_ms=Counter()
    for ti,task in enumerate(tasks):
        assert task['stopped_by_eos']
        assert len(task['correctness'])==6
        assert all(r['tokens_equal'] and r['kept_logits_bit_exact'] and r['live_state_bit_exact'] for r in task['correctness'])
        assert len(task['timings'])==repeats*len(MODES)
        for repetition in range(repeats):
            rows=sorted((r for r in task['timings'] if r['repetition']==repetition),key=lambda r:r['order'])
            assert tuple(r['mode'] for r in rows)==mode_order(repetition,ti)
            assert all(r['tokens_equal'] and r['live_state_bit_exact'] for r in rows)
        for mode in MODES:
            values=[r['decode_ms'] for r in task['timings'] if r['mode']==mode]
            assert statistics.median(values)==task['summary'][mode]['median_ms']
        for row in task['timings']:
            mode=row['mode']
            if mode=='serial':continue
            counts[mode].update(s['width'] for s in row['steps'])
            draft_calls[mode]+=sum(s['draft_invoked'] for s in row['steps'])
            predict_ms[mode]+=sum(s['pre_ms']+s['post_ms'] for s in row['steps'])
            draft_ms[mode]+=sum(s['draft_ms'] for s in row['steps'])
        answer=task['text'].split('</think>')[-1].replace('<|im_end|>','').replace('<|endoftext|>','').strip()
        tid=task['id']
        check=None
        if tid=='test_product':check=answer=='72'
        elif tid=='test_city':check='tokyo' in answer.lower()
        elif tid=='test_translation':check='晚安' in answer or '晚上好' in answer
        elif tid=='test_json':
            try:check=json.loads(answer)=={'animal':'cat','age':4}
            except ValueError:check=False
        elif tid=='test_reason_math':check='82' in answer
        elif tid=='test_explanation':check=any(word in answer for word in ('草稿','拒绝','验证','额外'))
        task['selected_literal_or_property_check']=check
    assert len(report['heldout_prediction'])==8
    for mode in MODES:
        assert sum(t['summary'][mode]['median_ms'] for t in tasks)==report['summary'][mode]['sum_medians_ms']
    summary=report['summary']
    summary['width_counts']={mode:dict(sorted(c.items())) for mode,c in counts.items()}
    summary['draft_calls']=dict(draft_calls)
    summary['prediction_and_decision_total_ms']=dict(predict_ms)
    summary['draft_total_ms']=dict(draft_ms)
    summary['maximum_live_position']=max(t['prompt_tokens']+len(t['token_ids'])-1 for t in tasks)
    best=min(('fixed1','fixed2','fixed3'),key=lambda mode:summary[mode]['sum_medians_ms'])
    summary['best_aggregate_fixed_mode']=best
    for mode in ('gate','local'):
        summary[mode]['ratio_over_best_fixed']=summary[best]['sum_medians_ms']/summary[mode]['sum_medians_ms']
        summary[mode]['ratio_over_native_dynamic']=summary['dynamic']['sum_medians_ms']/summary[mode]['sum_medians_ms']
    public=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for private in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/','training_pre','training_post','cal_pre','cal_post'):
        assert private not in public,private
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(public)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    print(json.dumps(export(args.source,args.output,args.repo),indent=2))
