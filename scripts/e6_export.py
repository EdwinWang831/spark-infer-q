"""Validate the measured design and retain all outputs, runs and limitations."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

from e6_checks import answer_check
from e6_tasks import TASKS,CALIBRATION,HEADS,POLICIES,order


def summarize(report):
    assert report['status']=='PASS_BOUNDED_DRAFT_HEAD' and report['stage']=='complete'
    assert report['kernel']['status']=='PASS_Q4_STORED_WEIGHT_ARITHMETIC'
    assert len(report['kernel']['cases'])==12
    assert report['reference']['target_head_sha256_before']==report['reference']['target_head_sha256_after']
    assert [t['id'] for t in report['tasks']]==[t['id'] for t in TASKS]
    assert [t['id'] for t in report['calibration']['prompts']]==[t['id'] for t in CALIBRATION]
    assert report['measurement_contract']['repetitions']>=4 and report['measurement_contract']['repetitions']%4==0
    assert len(report['profile'])==4 and all(p['baseline_adapter_exact'] for p in report['profile'])
    for head in HEADS:
        meta=report['heads'][head]
        assert meta['effective_precision']==('q4' if head.startswith('q4') else 'bf16')
        assert meta['rows']==(report['preflight']['subset_rows'] if head.endswith('subset') else report['preflight']['vocab_size'])
    counts=dict(timing_runs=0,oracle_runs=0,committed_logit_rows=0,timed_live_state_checks=0,
        naturally_completed_tasks=0,capped_tasks=0,decode_tokens=0)
    aggregates={};profile={}
    for task,definition in zip(report['tasks'],TASKS,strict=True):
        for k,v in definition.items():assert task[k]==v,(task['id'],k)
        assert task['prompt_ids'] and 1<=len(task['token_ids'])<=task['limit']
        assert task['stopped_by_eos'] or len(task['token_ids'])==task['limit']
        counts['naturally_completed_tasks' if task['stopped_by_eos'] else 'capped_tasks']+=1
        counts['decode_tokens']+=len(task['token_ids'])-1
        task['answer_check']=answer_check(task,task['text'],task['stopped_by_eos'])
        assert len(task['correctness'])==8
        assert {(r['head'],r['policy']) for r in task['correctness']}=={(h,p) for h in HEADS for p in POLICIES}
        for check in task['correctness']:
            assert check['tokens_equal'] and check['committed_logits_exact'] and check['live_state_exact']
            counts['oracle_runs']+=1;counts['committed_logit_rows']+=sum(s['emitted_tokens'] for s in check['steps'])
        runs=task['timings'];assert len(runs)==8*report['measurement_contract']['repetitions']
        for rep in range(report['measurement_contract']['repetitions']):
            block=[r for r in runs if r['repetition']==rep]
            assert [(r['head'],r['policy']) for r in block]==order(rep,TASKS.index(definition))
            assert [r['order'] for r in block]==list(range(8))
        task['summary']={}
        for head in HEADS:
            for policy in POLICIES:
                rows=[r for r in runs if r['head']==head and r['policy']==policy]
                for row in rows:
                    assert row['tokens_equal'] and row['live_state_exact'] and row['decode_ms']>0
                    assert sum(s['emitted_tokens'] for s in row['steps'])==len(task['token_ids'])-1
                    for s in row['steps']:
                        assert 1<=s['verified_rows']<=s['width']<=8
                        assert 0<=s['accepted_drafts']<=s['observed_drafts']<=s['verified_rows']-1
                median=statistics.median(r['decode_ms'] for r in rows)
                task['summary'][head+'_'+policy]=dict(median_decode_ms=median,
                    min_ms=min(r['decode_ms'] for r in rows),max_ms=max(r['decode_ms'] for r in rows),
                    decode_tokens_per_s=(len(task['token_ids'])-1)*1000/median)
                counts['timing_runs']+=len(rows);counts['timed_live_state_checks']+=len(rows)
    for head in HEADS:
        fields=('backbone_ms','head_topk_ms','selector_host_ms','proposer_ms')
        profile[head]={f:statistics.median(s[f] for p in report['profile'] for s in p['heads'][head]['samples']) for f in fields}
        for metric in ('full_top16_overlap','logits_relative_l2','logits_max_abs'):
            profile[head][metric]=(max if metric=='logits_max_abs' else statistics.mean)(p['heads'][head][metric] for p in report['profile'])
        profile[head]['full_top1_agreement']=sum(p['heads'][head]['full_top1_matches'] for p in report['profile'])/28
        profile[head]['full_proposal_agreement']=sum(p['heads'][head]['full_proposal_matches'] for p in report['profile'])/28
        for policy in POLICIES:
            key=head+'_'+policy
            total=sum(t['summary'][key]['median_decode_ms'] for t in report['tasks'])
            rows=[r for t in report['tasks'] for r in t['timings'] if r['head']==head and r['policy']==policy]
            steps=[s for r in rows for s in r['steps']]
            accepted=sum(s['accepted_drafts'] for s in steps);observed=sum(s['observed_drafts'] for s in steps)
            aggregates[key]=dict(sum_task_medians_ms=total,decode_tokens_per_s=counts['decode_tokens']*1000/total,
                accepted_drafts=accepted,observed_drafts=observed,accepted_over_observed=accepted/observed if observed else None,
                emitted_tokens_per_iteration=sum(s['emitted_tokens'] for s in steps)/len(steps),
                widths=dict(collections.Counter(s['width'] for s in steps)))
    for head in HEADS:
        for policy in POLICIES:
            rows=[r for t in report['tasks'] for r in t['timings'] if r['head']==head and r['policy']==policy]
            decode_sum=sum(r['decode_ms'] for r in rows)
            aggregates[head+'_'+policy]['component_fraction_of_decode']={
                component:sum(s[component] for r in rows for s in r['steps'])/decode_sum
                for component in ('draft_ms','verify_ms','commit_ms')}
            ratios=[]
            for rep in range(report['measurement_contract']['repetitions']):
                base=sum(r['decode_ms'] for t in report['tasks'] for r in t['timings'] if r['head']=='bf16_full' and r['policy']==policy and r['repetition']==rep)
                candidate=sum(r['decode_ms'] for r in rows if r['repetition']==rep)
                ratios.append(base/candidate)
            aggregates[head+'_'+policy]['paired_repetition_ratios_to_bf16_full']=ratios
    for policy in POLICIES:
        base=aggregates['bf16_full_'+policy]['sum_task_medians_ms']
        for head in HEADS:aggregates[head+'_'+policy]['speed_ratio_to_bf16_full_same_policy']=base/aggregates[head+'_'+policy]['sum_task_medians_ms']
    for head in HEADS:
        aggregates[head+'_dynamic7']['speed_ratio_to_same_head_fixed7']=aggregates[head+'_fixed7']['sum_task_medians_ms']/aggregates[head+'_dynamic7']['sum_task_medians_ms']
    assert [c['id'] for c in report['cold_controls']]==[t['id'] for t in TASKS[:2]]
    cold={h:0. for h in HEADS}
    for control in report['cold_controls']:
        assert len(control['timings'])==8
        for h in HEADS:
            rows=[r for r in control['timings'] if r['head']==h]
            assert len(rows)==2 and all(r['tokens_equal'] and r['total_ms']>=r['decode_ms'] for r in rows)
            cold[h]+=statistics.median(r['total_ms'] for r in rows)
    counts['cold_controls']=16
    return dict(status=report['status'],summary=counts,profile_summary=profile,aggregates=aggregates,
        cold_sum_task_medians_ms=cold,evidence=report,
        limits=['single Spark only; optional provider interface is not cross-GPU validation',
                'BF16 target head retained; compressed draft head is additional payload',
                'all heads resident; all comparisons use the same executor and their own serial oracle',
                'greedy only; no stochastic sampling, long-context, batching or full-engine claim',
                'separate new short-task study, not directly ranked against E4 native MTP TPS',
                'calibration proposal agreement is a small diagnostic, not general BF16 fidelity or quality'])


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    report=json.loads(a.input.read_text());repo=Path(__file__).resolve().parents[1]
    for name,sha in report['harness_sha256'].items():
        path=repo/name if name.startswith('spark_infer_q/') else repo/'scripts'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,name
    result=summarize(report)
    result['analysis_harness_sha256']={name:hashlib.sha256((repo/'scripts'/name).read_bytes()).hexdigest() for name in ('e6_export.py','e6_checks.py')}
    text=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for private in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/'):assert private not in text,private
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(json.dumps({k:v for k,v in result.items() if k not in ('evidence',)},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
