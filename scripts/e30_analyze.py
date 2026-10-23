"""Analyze actual Chrome kernel events, excluding CUDA user-annotation parents."""
from bisect import bisect_right
from collections import defaultdict


def union_us(intervals):
    total=0.;end=None
    for a,b in sorted(intervals):
        if b<a:raise ValueError('negative event duration')
        if end is None or a>=end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total


def family(name):
    if 'nvfp4_expert_kernel' in name:return 'nvfp4_experts'
    if 'qmmf_kernel' in name:return 'quantized_projections'
    if name=='_b16mm':return 'bf16_matmul'
    return 'other'


def analyze_trace(trace):
    events=trace['traceEvents']
    phases=sorted((e for e in events if e.get('cat')=='user_annotation' and e.get('ph')=='X' and e['name']=='e30.verify'),key=lambda e:e['ts'])
    if not phases:raise ValueError('missing target verification ranges')
    if any(a['ts']+a['dur']>b['ts'] for a,b in zip(phases,phases[1:])):raise ValueError('overlapping target ranges')
    starts=[e['ts'] for e in phases];kernels=[];activities=[]
    for e in events:
        if e.get('ph')!='X' or e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'):continue
        i=bisect_right(starts,e['ts'])-1
        if i<0 or e['ts']+e['dur']>phases[i]['ts']+phases[i]['dur']+1.:continue
        if e['dur']<0:raise ValueError('negative CUDA duration')
        activities.append((e['ts'],e['ts']+e['dur']))
        if e['cat']=='kernel':kernels.append(e)
    groups=defaultdict(lambda:dict(calls=0,total_us=0.))
    names=defaultdict(lambda:dict(calls=0,total_us=0.))
    grids=defaultdict(lambda:dict(calls=0,total_us=0.))
    for e in kernels:
        for mapping,key in ((groups,family(e['name'])),(names,e['name'])):
            mapping[key]['calls']+=1;mapping[key]['total_us']+=e['dur']
        if e['name']=='_b16mm':
            key=tuple(e.get('args',{}).get('grid',[]));grids[key]['calls']+=1;grids[key]['total_us']+=e['dur']
    total=sum(e['dur'] for e in kernels)
    return dict(rounds=len(phases),verify_wall_us=sum(e['dur'] for e in phases),kernel_calls=len(kernels),
        kernel_sum_us=total,kernel_union_us=union_us([(e['ts'],e['ts']+e['dur']) for e in kernels]),
        activity_union_us=union_us(activities),
        families={n:dict(**v,fraction_of_kernel_sum=v['total_us']/total if total else 0.) for n,v in sorted(groups.items())},
        bf16_grids=[dict(grid=list(k),**v) for k,v in sorted(grids.items())],
        top_kernels=sorted([dict(name=n,**v) for n,v in names.items()],key=lambda v:-v['total_us'])[:20])
