#!/usr/bin/env python3
"""Check paired draft microtimings and full decode results before export."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

MODES=('serial','original','cached','eos')


def order(repetition,task_index):
    offset=(repetition+task_index)%4
    return MODES[offset:]+MODES[:offset]


def export(source,output,repo):
    report=json.loads(source.read_text())
    assert report['status']=='PASS_BOUNDED_DRAFT_FASTPATH'
    for name,expected in report['harness_sha256'].items():
        path=repo/name if '/' in name else repo/'scripts'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    previous=json.loads((repo/'results/e3/2026-10-20-local-policy.json').read_text())
    assert report['reference']['previous_policy_model_sha256']==previous['frozen_model_sha256']
    previous_tokens={t['id']:t['token_ids'] for t in previous['tasks']}
    assert report['math_preflight']['rope_bit_exact'] and report['math_preflight']['all_norms_bit_exact']
    assert len(report['tasks'])==8
    repeats=report['measurement_contract']['repetitions']
    assert repeats>=4 and repeats%4==0
    micro={mode:0. for mode in ('original','cached')}
    uncropped,actual,reduced_steps=0,0,0
    for ti,task in enumerate(report['tasks']):
        assert task['stopped_by_eos'] and task['token_ids']==previous_tokens[task['id']]
        assert len(task['correctness'])==3
        assert all(c['hidden_and_proposals_exact'] and c['kept_logits_bit_exact'] and c['tokens_equal'] and c['live_state_bit_exact'] for c in task['correctness'])
        assert len(task['timings'])==repeats*4
        for rep in range(repeats):
            rows=sorted((r for r in task['timings'] if r['repetition']==rep),key=lambda r:r['order'])
            assert tuple(r['mode'] for r in rows)==order(rep,ti)
            assert all(r['tokens_equal'] and r['live_state_bit_exact'] for r in rows)
        for mode in MODES:
            assert statistics.median(r['decode_ms'] for r in task['timings'] if r['mode']==mode)==task['summary'][mode]['median_ms']
        for mode in micro:
            assert len(task['micro_draft_ms'][mode])==6
            micro[mode]+=statistics.median(task['micro_draft_ms'][mode])
        for row in task['timings']:
            if row['mode']=='eos':
                for step in row['steps']:
                    uncropped+=step['uncropped_rows'];actual+=step['rows']
                    reduced_steps+=int(step['rows']<step['uncropped_rows'])
    summary=report['summary']
    for mode in MODES:
        assert sum(t['summary'][mode]['median_ms'] for t in report['tasks'])==summary[mode]['sum_medians_ms']
    summary['cached_ratio_over_original']=summary['original']['sum_medians_ms']/summary['cached']['sum_medians_ms']
    summary['eos_ratio_over_original']=summary['original']['sum_medians_ms']/summary['eos']['sum_medians_ms']
    summary['eos_ratio_over_cached']=summary['cached']['sum_medians_ms']/summary['eos']['sum_medians_ms']
    summary['eos_verifier_rows']={'uncropped':uncropped,'actual':actual,'reduced_steps':reduced_steps}
    summary['draft_micro_sum_of_prompt_medians_ms']=micro
    summary['draft_micro_ratio']=micro['original']/micro['cached']
    summary['maximum_live_position']=max(t['prompt_tokens']+len(t['token_ids'])-1 for t in report['tasks'])
    public=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for private in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/'):
        assert private not in public,private
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(public)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args();print(json.dumps(export(a.source,a.output,a.repo),indent=2))
