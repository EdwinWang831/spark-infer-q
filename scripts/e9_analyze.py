"""Portable Nsight interval accounting; never equate GPU gaps with Python time."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


def merged(intervals,lo=None,hi=None):
    clipped=[]
    for a,b in intervals:
        if lo is not None:a=max(a,lo)
        if hi is not None:b=min(b,hi)
        if b>a:clipped.append((a,b))
    result=[]
    for a,b in sorted(clipped):
        if result and a<=result[-1][1]:result[-1]=(result[-1][0],max(b,result[-1][1]))
        else:result.append((a,b))
    return result


def duration(intervals,lo=None,hi=None):return sum(b-a for a,b in merged(intervals,lo,hi))


def intersection_duration(left,right):
    aa,bb=merged(left),merged(right);i=j=total=0
    while i<len(aa) and j<len(bb):
        total+=max(0,min(aa[i][1],bb[j][1])-max(aa[i][0],bb[j][0]))
        if aa[i][1]<bb[j][1]:i+=1
        else:j+=1
    return total


class RangeIndex:
    def __init__(self,ranges):
        self.rows=sorted(ranges,key=lambda r:r['start']);self.starts=[r['start'] for r in self.rows]
        assert all(a['end']<=b['start'] for a,b in zip(self.rows,self.rows[1:])), 'categories overlap'
    def find(self,a,b):
        i=bisect_right(self.starts,a)-1
        if i>=0 and b<=self.rows[i]['end']:return self.rows[i]['name']
        return 'unclassified'


def analyze(path):
    c=sqlite3.connect(path);c.row_factory=sqlite3.Row
    tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    def rows(name):return [dict(r) for r in c.execute('select * from '+name)] if name in tables else []
    strings={r['id']:r['value'] for r in rows('StringIds')}
    nvtx=[]
    for r in rows('NVTX_EVENTS'):
        name=r.get('text') or strings.get(r.get('textId'),'')
        if name.startswith('e9.') and r.get('end') is not None:nvtx.append(dict(start=r['start'],end=r['end'],name=name,tid=r['globalTid']))
    roots=[r for r in nvtx if r['name'].startswith('e9.decode.')];assert len(roots)==1
    root=roots[0];lo,hi=root['start'],root['end'];wall=hi-lo;assert wall>0
    nvtx=[r for r in nvtx if r['tid']==root['tid'] and lo<=r['start']<r['end']<=hi]
    groups={prefix:RangeIndex([r for r in nvtx if r['name'].startswith('e9.'+prefix+'.')]) for prefix in ('phase','target','draft','selector')}
    def host_group(prefix):
        out={}
        for name in sorted({r['name'] for r in groups[prefix].rows}):
            entries=[r for r in groups[prefix].rows if r['name']==name]
            ns=duration((r['start'],r['end']) for r in entries)
            out[name]={'host_ms':ns/1e6,'decode_percent':ns*100/wall,'calls':len(entries)}
        return out
    apis=[]
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        for r in rows(table):
            if lo<=r['start']<r['end']<=hi:
                r['name']=strings[r['nameId']];apis.append(r)
    main=[r for r in apis if r['globalTid']==root['tid']]
    by_cid=defaultdict(list)
    for r in apis:by_cid[r['correlationId']].append(r)
    kernels=[r for r in rows('CUPTI_ACTIVITY_KIND_KERNEL') if r['end']>lo and r['start']<hi]
    assert kernels and main,'missing CUDA activity'
    kernel_intervals=[(r['start'],r['end']) for r in kernels]
    kernel_groups=defaultdict(lambda:{'calls':0,'sum_ns':0,'intervals':[]})
    launched_by={p:defaultdict(list) for p in groups};correlated=0
    for r in kernels:
        name=strings.get(r['shortName'],str(r['shortName']))
        group=kernel_groups[name];group['calls']+=1;group['sum_ns']+=r['end']-r['start'];group['intervals'].append((r['start'],r['end']))
        candidates=by_cid.get(r['correlationId'],[])
        if candidates:
            correlated+=1;api=min(candidates,key=lambda r:r['end']-r['start'])
            for prefix,index in groups.items():launched_by[prefix][index.find(api['start'],api['end'])].append((r['start'],r['end']))
        else:
            for prefix in groups:launched_by[prefix]['unclassified'].append((r['start'],r['end']))
    copies=[r for r in rows('CUPTI_ACTIVITY_KIND_MEMCPY') if r['end']>lo and r['start']<hi]
    memsets=[r for r in rows('CUPTI_ACTIVITY_KIND_MEMSET') if r['end']>lo and r['start']<hi]
    activity=kernel_intervals+[(r['start'],r['end']) for r in copies+memsets]
    busy=duration(activity,lo,hi);kernel_busy=duration(kernel_intervals,lo,hi)
    api_groups=defaultdict(list)
    for r in main:api_groups[r['name']].append((r['start'],r['end']))
    sync=[(r['start'],r['end']) for r in main if 'Synchronize' in r['name']]
    launch=[(r['start'],r['end']) for r in main if 'Launch' in r['name']]
    copy_counts=defaultdict(lambda:{'calls':0,'bytes':0,'sum_ns':0})
    for r in copies:
        group=copy_counts[str(r['copyKind'])];group['calls']+=1;group['bytes']+=r['bytes'];group['sum_ns']+=r['end']-r['start']
    enum_rows=rows('ENUM_CUDA_MEMCPY_OPER')
    enum={str(r['id']):r.get('label',r.get('name',str(r))) for r in enum_rows}
    result={'task':root['name'].removeprefix('e9.decode.'),'decode_range_ms':wall/1e6,
        'host_phases':host_group('phase'),'target_subphases':host_group('target'),
        'draft_subphases':host_group('draft'),'selector_subphases':host_group('selector'),
        'gpu':{'kernel_calls':len(kernels),'kernel_busy_union_ms':kernel_busy/1e6,'kernel_busy_percent':kernel_busy*100/wall,
            'kernel_duration_sum_ms':sum(r['end']-r['start'] for r in kernels)/1e6,
            'kernel_copy_memset_busy_union_ms':busy/1e6,'kernel_copy_memset_busy_percent':busy*100/wall,
            'no_recorded_gpu_activity_ms':(wall-busy)/1e6,'correlated_kernel_calls':correlated,
            'kernel_by_launch_phase':{p:{k:duration(v,lo,hi)/1e6 for k,v in values.items()} for p,values in launched_by.items()},
            'top_kernels':sorted([{'name':k,'calls':v['calls'],'duration_sum_ms':v['sum_ns']/1e6,
                'busy_union_ms':duration(v['intervals'],lo,hi)/1e6} for k,v in kernel_groups.items()],key=lambda r:r['duration_sum_ms'],reverse=True)[:25],
            'memcpy':{enum.get(k,k):{**v,'sum_ms':v['sum_ns']/1e6} for k,v in copy_counts.items()},'memset_calls':len(memsets)},
        'cpu_cuda':{'main_thread_api_calls':len(main),'main_thread_api_union_ms':duration([(r['start'],r['end']) for r in main])/1e6,
            'sync_calls':sum('Synchronize' in r['name'] for r in main),'sync_union_ms':duration(sync)/1e6,
            'sync_overlap_gpu_activity_ms':intersection_duration(sync,activity)/1e6,
            'launch_calls':sum('Launch' in r['name'] for r in main),'launch_union_ms':duration(launch)/1e6,
            'launch_overlap_gpu_activity_ms':intersection_duration(launch,activity)/1e6,
            'api_by_name':{k:{'calls':len(v),'union_ms':duration(v)/1e6,'sum_ms':sum(b-a for a,b in v)/1e6} for k,v in sorted(api_groups.items())}},
        'possible_data_loss_messages':sum(any(word in r['text'].lower() for word in ('dropped','lost events','buffer overflow')) for r in rows('DIAGNOSTIC_EVENT')),
        'trace_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'limits':['NVTX host phases are inclusive of GPU waits; nested groups must not be added to parent totals',
            'CPU CUDA API intervals overlap GPU execution and are not an additive time partition',
            'no recorded GPU activity is not equivalent to Python interpreter overhead',
            'kernel-active fraction is timeline coverage, not SM utilization or memory bandwidth utilization',
            'kernel durations grouped by launch phase may overlap across phases; the global union is authoritative']}
    covered=sum(v['host_ms'] for v in result['host_phases'].values());assert covered<=wall/1e6+1e-4
    result['unlabeled_host_ms']=wall/1e6-covered
    c.close();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('sqlite',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=analyze(a.sqlite);out=json.dumps(result,indent=2,allow_nan=False)+'\n'
    for private in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/'):assert private not in out
    a.output.write_text(out);print(json.dumps({'task':result['task'],'decode_ms':result['decode_range_ms'],
        'gpu_active_percent':result['gpu']['kernel_copy_memset_busy_percent'],'kernel_calls':result['gpu']['kernel_calls']}))


if __name__=='__main__':main()
