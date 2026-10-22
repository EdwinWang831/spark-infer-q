"""Validate paired E11 projection evidence, retaining runtime budget changes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from e7_tasks import TASKS

MODES=('dynamic7_kernel_original','dynamic7_kernel_partials')


def summarize(report,server,baseline):
    assert report['status']=='PASS_INTEGRATED_PROJECTION'
    assert report['contract']['modes']==list(MODES) and report['contract']['repetitions']==4
    assert server['status']=='READY' and server['default_prefill_rows']==8
    assert server['projection_scratch_bytes']==819200
    assert server['harness_sha256']==report['harness_sha256']
    assert server['head']['precision']=='q4' and server['head']['rows']==248320
    assert server['head']['incremental_payload_bytes']==397312000
    assert server['calibration']==baseline['candidate_servers'][0]['calibration']
    assert len(report['tasks'])==len(TASKS)==10
    tasks=[];hist={m:Counter() for m in MODES};oracle_rows=0
    for ti,(rec,task) in enumerate(zip(report['tasks'],TASKS)):
        assert all(rec[k]==v for k,v in task.items())
        c=rec['correctness'];old=baseline['tasks'][ti]['candidate_correctness'][0]
        assert c['token_ids']==old['token_ids'] and c['prompt_ids']==old['prompt_ids']
        assert c['serial_logit_rows']==len(c['token_ids'])-1
        assert [(x['mode'],x['prefill_rows']) for x in c['checks']]==[('serial',8),('fixed7',8),('dynamic7',1),('dynamic7',8),('fixed7_kernel_partials',8),('dynamic7_kernel_partials',8)]
        assert all(x['kept_logits_exact'] and x['live_state_exact'] for x in c['checks'])
        oracle_rows+=c['serial_logit_rows']*6
        assert len(rec['timings'])==8
        for rep in range(4):
            rows=sorted((r for r in rec['timings'] if r['repetition']==rep),key=lambda r:r['order'])
            assert [r['mode'] for r in rows]==list(MODES if (rep+ti)%2==0 else MODES[::-1])
        out={'id':task['id'],'arm':task['arm'],'prompt_tokens':len(c['prompt_ids']),
            'completion_tokens':len(c['token_ids']),'modes':{}}
        for mode in MODES:
            rows=[r for r in rec['timings'] if r['mode']==mode];assert len(rows)==4
            for row in rows:
                assert row['extra']['token_ids']==c['token_ids'] and row['extra']['prompt_ids']==c['prompt_ids']
                assert row['usage']['completion_tokens']==len(c['token_ids'])
                assert row['usage']['prompt_tokens']==len(c['prompt_ids'])
                assert row['usage']['prompt_tokens_details']['cached_tokens']==0
                assert row['runtime']['prefill_rows']==8
                assert row['runtime']['projection_kernel']==mode.rsplit('_',1)[-1]
                counts=row['runtime']['projection_counts']
                if mode==MODES[0]:assert counts['eligible_original']>0 and counts['changed_to_partials']==0
                else:assert counts['changed_to_partials']>0 and counts['eligible_original']==0
                assert row['runtime']['enable_thinking']==task['thinking']
                assert row['post_timing_check']==dict(tokens_exact=True,live_state_exact=True)
                assert row['runtime']['token_sha']==hashlib.sha256(','.join(map(str,c['token_ids'])).encode()).hexdigest()[:12]
                assert (row['finish_reason']=='stop')==c['stopped_by_eos']
                row['prompt_ms']=row['runtime']['prefill_seconds']*1000
                row['decode_ms']=row['runtime']['decode_seconds']*1000
                assert abs(row['decode_ms']-(len(c['token_ids'])-1)*1000/row['runtime']['tokens_per_second'])<1e-6
                assert all(row[k]>0 for k in ('wall_ms','wire_ttft_ms','prompt_ms','decode_ms'))
                hist[mode].update(s['width'] for s in row['extra']['steps'])
                if 'pass_check' in row['answer_check']:assert row['answer_check']['pass_check']
                if 'nonempty' in row['answer_check']:assert row['answer_check']['nonempty']
            out['modes'][mode]={k:statistics.median(r[k] for r in rows) for k in ('wall_ms','wire_ttft_ms','prompt_ms','decode_ms')}
            out['modes'][mode]['range_ms']={k:[min(r[k] for r in rows),max(r[k] for r in rows)] for k in ('wall_ms','wire_ttft_ms','prompt_ms','decode_ms')}
        tasks.append(out)
    aggregates={}
    for arm in ('natural','capped','all'):
        chosen=[t for t in tasks if arm=='all' or t['arm']==arm];ag={}
        for mode in MODES:
            sums={k:sum(t['modes'][mode][k] for t in chosen) for k in ('wall_ms','wire_ttft_ms','prompt_ms','decode_ms')}
            tokens=sum(t['completion_tokens']-1 for t in chosen)
            ag[mode]={'tasks':len(chosen),'decode_tokens':tokens,'sum_medians':sums,
                'decode_tokens_per_s':tokens*1000/sums['decode_ms']}
        ag['partials_vs_original']={k:ag[MODES[0]]['sum_medians'][k]/ag[MODES[1]]['sum_medians'][k] for k in ('wall_ms','wire_ttft_ms','prompt_ms','decode_ms')}
        aggregates[arm]=ag
    repeats=[]
    for rep in range(4):
        sums={mode:{k:sum(r[k] for t in report['tasks'] for r in t['timings'] if r['mode']==mode and r['repetition']==rep)
            for k in ('wall_ms','wire_ttft_ms','prompt_ms','decode_ms')} for mode in MODES}
        repeats.append({'repetition':rep,'sum_ms':sums,'partials_vs_original':{k:sums[MODES[0]][k]/sums[MODES[1]][k] for k in sums[MODES[0]]}})
    return {'status':'PASS_INTEGRATED_PROJECTION_BOUNDED','evidence':report,'server':server,
        'summary':{'timings':80,'complete_warmup_runs':70,'full_generation_oracle_runs':60,
            'candidate_full_generation_oracle_runs':20,'original_path_full_generation_oracle_runs':40,
            'committed_decode_oracle_rows':oracle_rows,'timed_post_request_state_checks':80,
            'aggregates':aggregates,'tasks':tasks,'repeat_diagnostics':repeats,
            'dynamic_width_counts':{m:dict(sorted(h.items())) for m,h in hist.items()},
            'projection_calls':{m:sum(r['runtime']['projection_counts']['changed_to_partials' if m==MODES[1] else 'eligible_original'] for t in report['tasks'] for r in t['timings'] if r['mode']==m) for m in MODES},
            'phase_ms':{m:{k:sum(step[k] for t in report['tasks'] for r in t['timings'] if r['mode']==m for step in r['extra']['steps']) for k in ('draft_ms','verify_ms','commit_ms')} for m in MODES},
            'verification_rows':{m:{k:sum(step[k] for t in report['tasks'] for r in t['timings'] if r['mode']==m for step in r['extra']['steps']) for k in ('verified_rows','emitted','accepted','observed')} for m in MODES}},
        'limits':['one load, four paired repeats, ten reused short-context tasks; not cross-load or unseen-input generalization',
            'native TensorFold not rerun; E7 cannot establish a new same-run native ranking',
            'Python synchronous SSE experiment, not full DeepInfer Worker/async or multi-request throughput',
            'runtime budget feedback remains active; changing kernel cost can change verification widths and work',
            'baseline includes the same disabled routing wrapper, whereas historical E8 had no wrapper',
            'same-path greedy exactness does not resolve independent target-prefill arithmetic or official BF16 draft fidelity']}


def main():
    p=argparse.ArgumentParser();p.add_argument('raw_dir',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    report=json.loads((a.raw_dir/'e11-integrated.json').read_text());repo=Path(__file__).resolve().parents[1]
    for name,sha in report['harness_sha256'].items():
        if name=='e7-frozen.json':f=repo/'results/e7/2026-10-21-tensorfold-match.json'
        elif name=='e6-frozen.json':f=repo/'results/e6/2026-10-21-draft-head.json'
        elif name.endswith('.json'):f=a.raw_dir/name
        elif name.startswith('spark_infer_q/'):f=repo/name
        else:f=repo/'scripts'/name
        assert hashlib.sha256(f.read_bytes()).hexdigest()==sha,name
    result=summarize(report,json.loads((a.raw_dir/'ours-server-0.json').read_text()),
        json.loads((repo/'results/e7/2026-10-21-tensorfold-match.json').read_text()))
    events={k:int(value) for k,value in (line.split() for line in (a.raw_dir/'memory.events').read_text().splitlines())}
    assert events['oom']==events['oom_kill']==0
    result['final_memory_events']=events
    result['analysis_harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for private in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/'):assert private not in output
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(output)
    print(json.dumps(result['summary']['aggregates'],indent=2))


if __name__=='__main__':main()
