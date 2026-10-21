#!/usr/bin/env python3
"""Validate successful eight-row evidence and export controlled public results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from spark_infer_q.wide_budget import MODES,order


def export(source,output,repo,attempt=None):
    report=json.loads(source.read_text())
    assert report['status']=='PASS_BOUNDED_WIDE_BLOCK'
    for name,expected in report['harness_sha256'].items():
        path=repo/name if '/' in name else repo/'scripts'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    assert report['controlled_budget']=={'full':8,'zero':1,'partial':2}
    contracts=report['state_contract']
    assert len(contracts)==44
    assert {(r['rows'],r['keep']) for r in contracts}=={(rows,keep) for rows in range(1,9) for keep in range(rows+1)}
    assert sum(r['zero_keep_discard'] for r in contracts)==8
    assert all(r['logits_features_exact'] and r['live_state_exact'] and r['continuation_exact'] for r in contracts)
    assert report['legacy_four_row_exact']
    fp=next(p for p in report['draft_parity'] if p['cases'][0]['dtype']=='torch.float32')
    for case in fp['cases']:
        assert case['block_size']==8 and len(case['proposal_ids'])==7
        assert case['hidden_relative_l2']<1e-3 and case['logits_relative_l2']<1e-3 and case['candidate_set_overlap']>=.95
    previous=json.loads((repo/'results/e3/2026-10-20-local-policy.json').read_text())
    previous_tokens={t['id']:t['token_ids'] for t in previous['tasks']}
    counts=Counter();effective=Counter();repeats=report['measurement_contract']['repetitions']
    assert len(report['tasks'])==8 and repeats>=5 and repeats%5==0
    for ti,task in enumerate(report['tasks']):
        assert task['stopped_by_eos'] and task['token_ids']==previous_tokens[task['id']]
        assert len(task['correctness'])==4
        assert all(c['tokens_equal'] and c['kept_logits_bit_exact'] and c['live_state_bit_exact'] for c in task['correctness'])
        for check in task['correctness']:
            for step in check['steps']:
                step['pre_observation_rates']=step.pop('lagged_rates')
        assert len(task['timings'])==repeats*len(MODES)
        for rep in range(repeats):
            rows=sorted((r for r in task['timings'] if r['repetition']==rep),key=lambda r:r['order'])
            assert tuple(r['mode'] for r in rows)==order(rep,ti)
            assert all(r['tokens_equal'] and r['live_state_bit_exact'] for r in rows)
        for mode in MODES:
            assert statistics.median(r['decode_ms'] for r in task['timings'] if r['mode']==mode)==task['summary'][mode]['median_ms']
        for row in task['timings']:
            for step in row['steps']:
                # Recorded after observe(): N-1 rates, not the N-2 total-budget input.
                step['pre_observation_rates']=step.pop('lagged_rates')
                if row['mode']=='dynamic7':
                    counts[step['width']]+=1;effective[step['verified_rows']]+=1
    summary=report['summary']
    for mode in MODES:
        assert sum(t['summary'][mode]['median_ms'] for t in report['tasks'])==summary[mode]['sum_medians_ms']
    summary['chosen_width_counts']=dict(sorted(counts.items()))
    summary['effective_width_counts']=dict(sorted(effective.items()))
    for mode in ('fixed3_wide','fixed7','dynamic7'):
        summary[mode]['ratio_over_fixed3']=summary['fixed3']['sum_medians_ms']/summary[mode]['sum_medians_ms']
    summary['dynamic7']['ratio_over_fixed7']=summary['fixed7']['sum_medians_ms']/summary['dynamic7']['sum_medians_ms']
    summary['dynamic7']['ratio_over_fixed3_wide']=summary['fixed3_wide']['sum_medians_ms']/summary['dynamic7']['sum_medians_ms']
    summary['maximum_live_position']=max(t['prompt_tokens']+len(t['token_ids'])-1 for t in report['tasks'])
    static_width=report['calibration']['frozen_static_width']
    assert 1<=static_width<=8 and len(report['static_control'])==8
    control_modes=('fixed_calibrated','fixed7','dynamic7')
    for ti,task in enumerate(report['static_control']):
        assert task['id']==report['tasks'][ti]['id']
        assert len(task['correctness'])==3 and len(task['timings'])==9
        assert all(c['tokens_equal'] and c['kept_logits_bit_exact'] and c['live_state_bit_exact'] for c in task['correctness'])
        for rep in range(3):
            rows=sorted((r for r in task['timings'] if r['repetition']==rep),key=lambda r:r['order'])
            offset=(rep+ti)%3
            assert tuple(r['mode'] for r in rows)==control_modes[offset:]+control_modes[:offset]
            assert all(r['tokens_equal'] and r['live_state_bit_exact'] for r in rows)
        for row in task['timings']:
            for step in row['steps']:
                step['pre_observation_rates']=step.pop('lagged_rates')
                if row['mode']=='fixed_calibrated':assert step['width']==static_width
        for mode in control_modes:
            assert statistics.median(r['decode_ms'] for r in task['timings'] if r['mode']==mode)==task['summary'][mode]
    for mode in control_modes:
        assert sum(t['summary'][mode] for t in report['static_control'])==report['static_control_summary'][mode]
    control=report['static_control_summary']
    summary['static_control_dynamic_over_frozen_fixed']=control['fixed_calibrated']/control['dynamic7']
    summary['static_control_dynamic_over_fixed7']=control['fixed7']/control['dynamic7']
    if attempt:
        first=json.loads(attempt.read_text())
        assert first['status']=='FAIL'
        report['previous_attempts']=[{'status':first['status'],'stage':first['stage'],'error':first['error'],
            'harness_sha256':first['harness_sha256'],
            'resolution':'test driver now discards a zero-keep speculative state instead of calling commit(keep=0); no failed timing included'}]
    text=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for private in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/'):
        assert private not in text,private
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--attempt',type=Path)
    a=p.parse_args();print(json.dumps(export(a.source,a.output,a.repo,a.attempt),indent=2))
