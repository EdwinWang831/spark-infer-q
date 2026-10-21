"""Join untraced controls and traced CPU/GPU data without claiming additive times."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics

TASKS=('zh_json','thinking_math','length_code')


def distribution(rows):
    total=sum(r['runtime']['decode_seconds']*1000 for r in rows)
    phases={k:sum(s[k] for r in rows for s in r['extra']['steps']) for k in ('draft_ms','verify_ms','commit_ms')}
    phases['other_ms']=total-sum(phases.values());assert phases['other_ms']>=0
    return {'runs':len(rows),'steps':sum(len(r['extra']['steps']) for r in rows),'decode_ms':total,
        'phase_ms':phases,'phase_percent':{k:v*100/total for k,v in phases.items()}}


def summarize(study,server,traces,baseline):
    assert study['status']=='PASS_PROFILE_REQUESTS' and len(study['tasks'])==3
    assert server['harness_sha256']==study['harness_sha256'] and server['default_prefill_rows']==8
    assert server['head']['precision']=='q4' and server['head']['rows']==248320
    assert server['calibration']==baseline['server']['calibration']
    assert [t['id'] for t in study['tasks']]==list(TASKS)
    assert len(traces)==3 and {t['task'] for t in traces}==set(TASKS)
    old={t['id']:t for t in baseline['evidence']['tasks']};entries=[];all_control=[];all_profile=[]
    for rec in study['tasks']:
        c=rec['correctness'];assert c['token_ids']==old[rec['id']]['correctness']['token_ids']
        assert c['prompt_ids']==old[rec['id']]['correctness']['prompt_ids']
        assert len(c['checks'])==4 and all(r['kept_logits_exact'] and r['live_state_exact'] for r in c['checks'])
        timings=rec['timings'];assert len(timings)==5 and [r['repetition'] for r in timings]==list(range(5))
        for i,row in enumerate(timings):
            assert row['runtime']['profiled']==(i==2) and row['runtime']['prefill_rows']==8
            assert row['extra']['token_ids']==c['token_ids'] and row['extra']['prompt_ids']==c['prompt_ids']
            assert row['post_timing_check']=={'tokens_exact':True,'live_state_exact':True}
        controls=[r for r in timings if not r['runtime']['profiled']];profile=timings[2]
        trace=next(t for t in traces if t['task']==rec['id'])
        assert abs(trace['decode_range_ms']-profile['runtime']['decode_seconds']*1000)<max(2,trace['decode_range_ms']*.01)
        assert trace['host_phases']['e9.phase.verify']['calls']==len(profile['extra']['steps'])
        assert trace['target_subphases']['e9.target.layer']['calls']==48*len(profile['extra']['steps'])
        assert trace['possible_data_loss_messages']==0
        median=statistics.median(r['runtime']['decode_seconds']*1000 for r in controls)
        entries.append({'id':rec['id'],'completion_tokens':len(c['token_ids']),'control_decode_median_ms':median,
            'control_decode_range_ms':[min(r['runtime']['decode_seconds']*1000 for r in controls),max(r['runtime']['decode_seconds']*1000 for r in controls)],
            'profile_decode_ms':profile['runtime']['decode_seconds']*1000,'profile_vs_control_time_ratio':profile['runtime']['decode_seconds']*1000/median,
            'control_width_counts':dict(Counter(s['width'] for r in controls for s in r['extra']['steps'])),
            'profile_width_counts':dict(Counter(s['width'] for s in profile['extra']['steps'])),
            'control_distribution':distribution(controls),'trace':trace})
        all_control.extend(controls);all_profile.append(profile)
    base_rows=[r for t in baseline['evidence']['tasks'] for r in t['timings'] if r['mode']=='dynamic7_rows8']
    host=defaultdict(float);wall=sum(t['decode_range_ms'] for t in traces)
    for trace in traces:
        for k,v in trace['host_phases'].items():host[k]+=v['host_ms']
    kernel=sum(t['gpu']['kernel_busy_union_ms'] for t in traces)
    busy=sum(t['gpu']['kernel_copy_memset_busy_union_ms'] for t in traces)
    return {'status':'PASS_PROFILE_DIAGNOSTIC','evidence':study,'server':server,
        'summary':{'baseline_e8_untraced':distribution(base_rows),'current_untraced_controls':distribution(all_control),
            'traced_requests':distribution(all_profile),'control_runs':12,'traced_runs':3,'complete_warmup_runs':15,
            'independent_full_generation_oracles':12,'timed_post_request_state_checks':15,
            'traced_host_ms':wall,'traced_host_phases':dict(host),'traced_host_phase_percent':{k:100*v/wall for k,v in host.items()},
            'traced_gpu_kernel_union_ms':kernel,'traced_gpu_activity_union_ms':busy,'traced_gpu_activity_percent':busy*100/wall,
            'traced_kernel_calls':sum(t['gpu']['kernel_calls'] for t in traces)},'tasks':entries,
        'limits':['three reused tasks and one model load; diagnosis, not a speedup or native ranking benchmark',
            'software tracing and NVTX can increase decode cost and change the runtime-adaptive width schedule',
            'untraced controls retain inactive Python wrappers; E8 full-run coarse distribution is separately retained',
            'CPU CUDA API and GPU intervals overlap; nested NVTX ranges are not additive',
            'no CPU stack samples; GPU idle intervals are not direct evidence of Python interpreter cost']}


def main():
    p=argparse.ArgumentParser();p.add_argument('raw_dir',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    study=json.loads((a.raw_dir/'profile-study.json').read_text())
    for name,sha in study['harness_sha256'].items():
        if name=='e7-frozen.json':f=repo/'results/e7/2026-10-21-tensorfold-match.json'
        elif name=='e6-frozen.json':f=repo/'results/e6/2026-10-21-draft-head.json'
        elif name.endswith('.json'):f=a.raw_dir/name
        elif name.startswith('spark_infer_q/'):f=repo/name
        else:f=repo/'scripts'/name
        assert hashlib.sha256(f.read_bytes()).hexdigest()==sha,name
    traces=[json.loads(p.read_text()) for p in sorted(a.raw_dir.glob('analysis-*.json'))];assert len(traces)==3
    result=summarize(study,json.loads((a.raw_dir/'ours-server-0.json').read_text()),traces,
        json.loads((repo/'results/e8/2026-10-21-integrated-prompt.json').read_text()))
    result['analysis_harness_sha256']={name:hashlib.sha256((repo/'scripts'/name).read_bytes()).hexdigest() for name in ('e9_export.py','e9_analyze.py')}
    output=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for private in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/'):assert private not in output
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(output)
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
