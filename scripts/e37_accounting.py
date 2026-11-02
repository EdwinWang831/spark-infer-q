"""Pure accounting for asynchronous GPU traces and paired prompt timings."""
import math


def union_us(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end<start:raise ValueError('negative interval')
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    return sum(b-a for a,b in merged)


def family(name):
    if 'nvfp4_expert_kernel' in name:return 'routed_experts'
    if 'qmmf' in name or '_fp4mm' in name or 'prompt16' in name:return 'quantized_projection'
    if '_b16mm' in name:return 'bf16_projection'
    if 'chain_kernel<' in name or 'replay_kernel<' in name:return 'gdn_recurrence'
    if any(s in name.lower() for s in ('attention','attn','qsa','kv8')):return 'attention'
    return 'other'


def analyze_trace(trace,root_name):
    events=trace['traceEvents']
    roots=[e for e in events if e.get('cat')=='user_annotation' and e.get('ph')=='X' and e.get('name')==root_name]
    if len(roots)!=1:raise ValueError('one prompt diagnostic range required')
    root=roots[0];start,end=root['ts'],root['ts']+root['dur']
    def within(e):return e.get('ph')=='X' and start<=e.get('ts',-1) and e['ts']+e.get('dur',0)<=end+1
    kernels=[e for e in events if e.get('cat')=='kernel' and within(e)]
    if not kernels:raise ValueError('no actual GPU kernels')
    memory=[e for e in events if e.get('cat') in ('gpu_memcpy','gpu_memset') and within(e)]
    api=[e for e in events if e.get('cat')=='cuda_runtime' and within(e)]
    groups={}
    for e in kernels:
        if not math.isfinite(e['dur']) or e['dur']<0:raise ValueError('invalid kernel time')
        name=family(e['name']);row=groups.setdefault(name,dict(calls=0,kernel_ms=0))
        row['calls']+=1;row['kernel_ms']+=e['dur']/1000
    total=sum(e['dur'] for e in kernels)/1000
    top={}
    for e in kernels:
        row=top.setdefault(e['name'],dict(name=e['name'],calls=0,kernel_ms=0))
        row['calls']+=1;row['kernel_ms']+=e['dur']/1000
    for row in groups.values():row['fraction_of_kernel_sum']=row['kernel_ms']/total
    busy=union_us([(e['ts'],e['ts']+e['dur']) for e in kernels])/1000
    activity=union_us([(e['ts'],e['ts']+e['dur']) for e in kernels+memory])/1000
    return dict(profiled_root_ms=root['dur']/1000,kernel_calls=len(kernels),kernel_duration_sum_ms=total,
        kernel_busy_union_ms=busy,device_activity_union_ms=activity,
        profiled_wall_outside_device_activity_ms=max(0,root['dur']/1000-activity),
        cuda_api_union_ms=union_us([(e['ts'],e['ts']+e['dur']) for e in api])/1000,
        memory_activity_calls=len(memory),by_kernel_family=groups,
        top_kernels=sorted(top.values(),key=lambda d:d['kernel_ms'],reverse=True)[:20],
        scope='diagnostic profiler perturbation; kernel sum is not wall, SM utilization or DRAM traffic')


def summarize_timings(rows,inputs):
    names={c['name']:c['length'] for c in inputs};modes=('fresh','pool')
    expected={(n,r,m) for n in names for r in range(6) for m in modes}
    if len(rows)!=len(expected):raise ValueError('missing/repeated timing runs')
    keyed={}
    for row in rows:
        key=(row['name'],row['repetition'],row['mode'])
        if key not in expected or key in keyed:raise ValueError('missing/repeated timing runs')
        if type(row['wall_ms']) not in (int,float) or not math.isfinite(row['wall_ms']) or row['wall_ms']<=0:
            raise ValueError('positive finite timing required')
        if row['prompt_tokens']!=names[row['name']] or row['state_exact'] is not True:
            raise ValueError('workload/state mismatch')
        keyed[key]=row
    for name in names:
        hashes={r['output_sha256'] for r in rows if r['name']==name}
        if len(hashes)!=1:raise ValueError('paired output mismatch')
    summary={}
    for mode in modes:
        subset=[r for r in rows if r['mode']==mode]
        tokens=sum(r['prompt_tokens'] for r in subset);ms=sum(r['wall_ms'] for r in subset)
        summary[mode]=dict(prompt_tokens=tokens,wall_ms=ms,prompt_tps=1000*tokens/ms)
    cases=[]
    for name in names:
        times={m:sum(keyed[name,r,m]['wall_ms'] for r in range(6))/6 for m in modes}
        cases.append(dict(name=name,mean_wall_ms=times,wall_reduction=1-times['pool']/times['fresh']))
    reps=[]
    for r in range(6):
        times={m:sum(keyed[n,r,m]['wall_ms'] for n in names) for m in modes}
        reps.append(dict(repetition=r,wall_reduction=1-times['pool']/times['fresh']))
    reduction=1-summary['pool']['wall_ms']/summary['fresh']['wall_ms']
    gate=reduction>=.02 and sum(r['wall_reduction']>0 for r in reps)>=5 and all(c['wall_reduction']>=-.02 for c in cases)
    return dict(modes=summary,per_case=cases,per_repetition=reps,wall_reduction=reduction,
                cost_gate_passed=gate,decision='KEEP_POOL_CANDIDATE' if gate else 'KEEP_E36_FRESH')
