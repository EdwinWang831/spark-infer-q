"""Validate ABBA measurements and summarize without assuming cross-engine parity."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from e4_tasks import TASKS, MODES, ordered_modes


def summarize(study, sessions, identity, servers):
    assert study['status']=='PASS_MATCHED_MEASUREMENTS'
    assert [(s['engine'],s['session']) for s in study['sessions']]==[('native',0),('ours',0),('ours',1),('native',1)]
    assert len(sessions)==4 and len(servers)==2
    for s in study['sessions']:assert s['status']=='PASS_MEASURED'
    index={(s['engine'],s['session']):s for s in sessions}
    assert len(index)==4
    tasks=[];histogram=Counter()
    for ti,t in enumerate(TASKS):
        rec={**t,'modes':{},'candidate_correctness':[]}
        for engine,modes in [('native',('serial','native_mtp')),('ours',MODES)]:
            parts=[]
            for sid in range(2):
                s=index[engine,sid];assert s['status']=='PASS_MEASURED' and len(s['tasks'])==len(TASKS)
                p=s['tasks'][ti];assert all(p[k]==t[k] for k in t)
                if engine=='ours':
                    c=p['correctness'];assert c['kept_logits_exact'] and c['live_state_exact']
                    assert c['modes']==['fixed6','fixed7','dynamic7']
                    rec['candidate_correctness'].append(c)
                assert len(p['timings'])==len(modes)*2
                for rep in range(sid*2,sid*2+2):
                    rows=sorted((r for r in p['timings'] if r['repetition']==rep),key=lambda r:r['order'])
                    expected=(modes if (rep+ti)%2==0 else modes[::-1]) if engine=='native' else ordered_modes(rep,ti)
                    assert tuple(r['mode'] for r in rows)==tuple(expected)
                    for row in rows:
                        assert row['runtime']['enable_thinking']==t['thinking']
                        assert row['usage']['completion_tokens']>0
                        assert row['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)==0
                        if engine=='ours':
                            assert row['extra']['token_ids']==p['correctness']['token_ids']
                            assert row['extra']['prompt_ids']==p['correctness']['prompt_ids']
                            assert len(row['extra']['token_ids'])==row['usage']['completion_tokens']
                        assert row['wall_ms']>0 and row['runtime']['tokens_per_second']>0
                        row['decode_ms']=(row['usage']['completion_tokens']-1)*1000/row['runtime']['tokens_per_second']
                        row['engine']=engine
                        if engine=='ours' and row['mode']=='dynamic7':histogram.update(s['width'] for s in row['extra']['steps'])
                parts.extend(p['timings'])
            for mode in modes:
                rows=[r for r in parts if r['mode']==mode]
                assert len(rows)==4
                hashes={r['runtime']['token_sha'] for r in rows}
                counts={r['usage']['completion_tokens'] for r in rows}
                rec['modes'][engine+'_'+mode]={'timings':rows,'stable_token_hash':len(hashes)==1,
                    'completion_token_counts':sorted(counts),'token_hashes':sorted(hashes),
                    **{k:statistics.median(r[k] for r in rows) for k in ('wall_ms','wire_ttft_ms','decode_ms')},
                    'min_decode_ms':min(r['decode_ms'] for r in rows),'max_decode_ms':max(r['decode_ms'] for r in rows),
                    'reached_length_limit':all(r['finish_reason']=='length' and r['usage']['completion_tokens']==t['limit'] for r in rows),
                    'all_stopped':all(r['finish_reason']=='stop' for r in rows),
                    'answer_check':rows[0]['answer_check'],
                    'answer_check_consistent':all(r['answer_check']==rows[0]['answer_check'] for r in rows),
                    'answer_check_pass_runs':sum(r['answer_check'].get('pass') is True for r in rows),
                    'nonempty_runs':sum(r['answer_check'].get('nonempty') is True for r in rows)}
            if engine=='native':
                rec['native_prompt_ids']=index[engine,0]['tasks'][ti]['prompt_ids']
                assert rec['native_prompt_ids']==index[engine,1]['tasks'][ti]['prompt_ids']
        c=rec['candidate_correctness'];assert c[0]['token_ids']==c[1]['token_ids'] and c[0]['prompt_ids']==c[1]['prompt_ids']
        rec['prompt_ids_equal']=rec['native_prompt_ids']==c[0]['prompt_ids']
        own_hash=hashlib.sha256(','.join(map(str,c[0]['token_ids'])).encode()).hexdigest()[:12]
        rec['native_draft_vs_serial_hash_equal']=rec['modes']['native_serial']['token_hashes']==rec['modes']['native_native_mtp']['token_hashes']
        rec['native_draft_vs_serial_count_equal']=rec['modes']['native_serial']['completion_token_counts']==rec['modes']['native_native_mtp']['completion_token_counts']
        rec['native_vs_candidate_hash_equal']=rec['modes']['native_serial']['token_hashes']==[own_hash]
        rec['native_vs_candidate_token_count_equal']=rec['modes']['native_serial']['completion_token_counts']==[len(c[0]['token_ids'])]
        tasks.append(rec)
    assert all(t['prompt_ids_equal'] for t in tasks),'chat template mismatch; do not claim matched-input comparison'
    mode_names=list(tasks[0]['modes'])
    aggregates={}
    for arm in ('natural','capped','all'):
        chosen=[t for t in tasks if arm=='all' or t['arm']==arm]
        ag={}
        for mode in mode_names:
            values=[t['modes'][mode] for t in chosen]
            assert all(v['stable_token_hash'] and len(v['completion_token_counts'])==1 for v in values),'non-deterministic output'
            decode_tokens=sum(v['completion_token_counts'][0]-1 for v in values)
            ms=sum(v['decode_ms'] for v in values)
            ag[mode]={'tasks':len(chosen),'decode_tokens':decode_tokens,'sum_median_decode_ms':ms,
                'decode_tokens_per_s':decode_tokens*1000/ms,'sum_median_wall_ms':sum(v['wall_ms'] for v in values),
                'sum_median_ttft_ms':sum(v['wire_ttft_ms'] for v in values),
                'completed_to_stop':sum(v['all_stopped'] for v in values),'reached_cap':sum(v['reached_length_limit'] for v in values)}
        aggregates[arm]=ag
    phase_costs={}
    for mode in ('fixed6','fixed7','dynamic7'):
        rows=[r for t in tasks for r in t['modes']['ours_'+mode]['timings']]
        sums={name:sum(s[name] for r in rows for s in r['extra']['steps']) for name in ('draft_ms','verify_ms','commit_ms')}
        total=sum(r['decode_ms'] for r in rows)
        sums.update(total_decode_ms=total,other_ms=total-sum(sums.values()),steps=sum(len(r['extra']['steps']) for r in rows))
        phase_costs[mode]=sums
    return {'status':'MEASURED_NOT_FULL_ENGINE_VALIDATION','study':study,'native_identity':identity,'candidate_servers':servers,
        'tasks':tasks,'summary':{'timing_runs':sum(len(m['timings']) for t in tasks for m in t['modes'].values()),
            'candidate_nonserial_oracle_runs':sum(len(c['modes']) for t in tasks for c in t['candidate_correctness']),
            'cross_engine_hash_agreement_cases':sum(t['native_vs_candidate_hash_equal'] for t in tasks),
            'native_spec_vs_serial_hash_agreement_cases':sum(t['native_draft_vs_serial_hash_equal'] for t in tasks),
            'all_prompt_ids_equal':True,'dynamic_width_counts':dict(sorted(histogram.items())),
            'aggregates':aggregates,'candidate_phase_costs':phase_costs},
        'limits':['native API exposes truncated token SHA, not complete output IDs or logits/state',
            'native generic-only warmup versus candidate per-case oracle warmup; original native first repetitions are slower; see separate post-hoc warmup control',
            'cross-engine output and quality differences must accompany speed comparisons',
            'short context only; no multi-request, vision, stochastic sampling or long-context result',
            'candidate uses serial prompt arithmetic; target prefill and official draft BF16 fidelity remain open',
            'streaming facade is an experiment adapter, not a deployed serving engine']}


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    def read(name):return json.loads((a.directory/name).read_text())
    study=read('study.json')
    repo=Path(__file__).resolve().parents[1]
    for name,sha in study['harness_sha256'].items():
        f=repo/name if name.startswith('spark_infer_q/') else a.directory/name if name=='calibration.json' else repo/'scripts'/name
        assert hashlib.sha256(f.read_bytes()).hexdigest()==sha,name
    result=summarize(study,[read(f'{engine}-{sid}.json') for engine in ('native','ours') for sid in range(2)],
        read('native-identity.json'),[read(f'ours-server-{sid}.json') for sid in range(2)])
    out=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for private in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/'):assert private not in out,private
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(out)
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
