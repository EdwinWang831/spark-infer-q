"""Actual CUPTI kernel attribution inside synchronized host phase ranges."""
from bisect import bisect_right
from collections import defaultdict


def family(name):
    if 'nvfp4_expert_kernel' in name:return 'experts'
    if 'qmmf_kernel' in name:return 'quantized_projections'
    if '_b16mm' in name:return 'bf16_matmul'
    if name in ('_scores','_select_tiles') or any(k in name.lower() for k in ('radix','select_top','indexer','_score_blocks','_pick')):return 'qsa_selection'
    if name in ('_chunks','_chunks8','_merge','_merge8'):return 'attention'
    return 'other'


def union_us(intervals):
    total=0.;end=None
    for a,b in sorted(intervals):
        if b<a:raise ValueError('negative interval')
        if end is None or a>=end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total


def analyze(trace,phase='e50.verify'):
    ranges=sorted((e for e in trace['traceEvents'] if e.get('cat')=='user_annotation' and e.get('ph')=='X' and e['name']==phase),key=lambda e:e['ts'])
    if not ranges:raise ValueError('missing phase '+phase)
    starts=[e['ts'] for e in ranges];kernels=[];activities=[]
    for e in trace['traceEvents']:
        if e.get('ph')!='X' or e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'):continue
        i=bisect_right(starts,e['ts'])-1
        if i<0 or e['ts']+e['dur']>ranges[i]['ts']+ranges[i]['dur']+1.:continue
        activities.append((e['ts'],e['ts']+e['dur']))
        if e['cat']=='kernel':kernels.append(e)
    names=defaultdict(lambda:dict(calls=0,total_us=0.));groups=defaultdict(lambda:dict(calls=0,total_us=0.))
    for e in kernels:
        for mapping,key in ((names,e['name']),(groups,family(e['name']))):
            mapping[key]['calls']+=1;mapping[key]['total_us']+=e['dur']
    total=sum(e['dur'] for e in kernels)
    return dict(phase=phase,rounds=len(ranges),wall_us=sum(r['dur'] for r in ranges),kernel_calls=len(kernels),kernel_sum_us=total,
        kernel_union_us=union_us([(e['ts'],e['ts']+e['dur']) for e in kernels]),activity_union_us=union_us(activities),
        families={k:dict(**v,fraction=v['total_us']/total if total else 0.) for k,v in groups.items()},
        top_kernels=sorted([dict(name=n,**v) for n,v in names.items()],key=lambda v:-v['total_us'])[:25])
