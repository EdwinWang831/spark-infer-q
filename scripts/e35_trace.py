"""Measure actual shared/routed kernel overlap, excluding CUDA annotation parents."""
from e30_analyze import union_us


def merge(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<a:raise ValueError('negative duration')
        if not out or a>out[-1][1]:out.append([a,b])
        else:out[-1][1]=max(out[-1][1],b)
    return out


def intersection_us(left,right):
    left,right=merge(left),merge(right)
    i=j=0;total=0.
    while i<len(left) and j<len(right):
        a,b=left[i];c,d=right[j]
        total+=max(0,min(b,d)-max(a,c))
        if b<d:i+=1
        else:j+=1
    return total


def analyze(trace):
    phases=[e for e in trace['traceEvents'] if e.get('cat')=='user_annotation' and e.get('ph')=='X' and e.get('name')=='e35.moe']
    if len(phases)!=1:raise ValueError('one measured MoE scope required')
    phase=phases[0]
    kernels=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X'
        and e['ts']>=phase['ts'] and e['ts']+e['dur']<=phase['ts']+phase['dur']+1]
    routed=[e for e in kernels if 'nvfp4_expert_kernel' in e['name']]
    shared=[e for e in kernels if 'qmmf_kernel' in e['name']]
    if not routed or not shared:raise ValueError('missing real routed/shared kernels')
    def intervals(events):return [(e['ts'],e['ts']+e['dur']) for e in events]
    return dict(kernel_calls=len(kernels),kernel_sum_us=sum(e['dur'] for e in kernels),
        kernel_active_union_us=union_us(intervals(kernels)),routed_calls=len(routed),shared_calls=len(shared),
        routed_active_us=union_us(intervals(routed)),shared_active_us=union_us(intervals(shared)),
        routed_shared_overlap_us=intersection_us(intervals(routed),intervals(shared)),
        routed_streams=sorted({e['args']['stream'] for e in routed}),
        shared_streams=sorted({e['args']['stream'] for e in shared}))
