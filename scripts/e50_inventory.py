"""Read-only host/CUDA ownership accounting; no allocation-limit changes."""
import os
from pathlib import Path
from types import ModuleType


def execute(session,source,tag,args,load,report):
    torch=session.torch
    def values(path):
        result={}
        for line in Path(path).read_text().splitlines():
            parts=line.split()
            if len(parts)>=2 and parts[1].isdigit():result[parts[0].rstrip(':')]=int(parts[1])*(1024 if len(parts)>2 and parts[2]=='kB' else 1)
        return result
    groups={};seen=set()
    def tensors(v,depth=0):
        if id(v) in seen or depth>20:return
        seen.add(id(v))
        if isinstance(v,torch.Tensor):
            st=v.untyped_storage();key=(str(v.device),st.data_ptr());groups[key]=st.nbytes()
        elif isinstance(v,dict):
            for x in v.values():tensors(x,depth+1)
        elif isinstance(v,(list,tuple)):
            for x in v:tensors(x,depth+1)
        elif not isinstance(v,(ModuleType,type)) and hasattr(v,'__dict__'):
            for x in vars(v).values():tensors(x,depth+1)
    per={}
    for label,owner in (('target',session.weights),('draft',session.draft),('projection',session.projection)):
        before=set(groups);tensors(owner)
        per[label]={device:sum(n for key,n in groups.items() if key not in before and key[0]==device)
            for device in {key[0] for key in groups if key not in before}}
    tables={}
    for layer in session.weights.layers:
        ple=getattr(layer,'ple',None)
        if ple is not None:
            t=ple.table;tables[id(t)]=dict(kind=type(t).__name__,logical_mapped_bytes=getattr(t,'nbytes',None),bits=getattr(t,'bits',None))
    maps={};kind='other'
    for line in Path('/proc/self/smaps').read_text().splitlines():
        if '-' in line.split()[0] and len(line.split())>=5:
            kind='checkpoint_files' if '.safetensors' in line else 'other_file' if len(line.split())>=6 and line.split()[-1].startswith('/') else 'anonymous_or_device'
        elif line.startswith(('Rss:','Pss:','Locked:','Anonymous:')):
            k,n,*_=line.split();maps.setdefault(kind,{}).setdefault(k[:-1],0);maps[kind][k[:-1]]+=int(n)*1024
    free,total=torch.cuda.mem_get_info();allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
    report.update(status='PASS_E50_MEMORY_INVENTORY',pid=os.getpid(),host_meminfo=values('/proc/meminfo'),
        process_rollup=values('/proc/self/smaps_rollup'),mapped_pss_by_category=maps,torch_storage_bytes_by_owner=per,
        ple_tables=list(tables.values()),ple_logical_mapped_bytes=sum(t['logical_mapped_bytes'] or 0 for t in tables.values()),
        cuda_allocated_bytes=allocated,cuda_reserved_bytes=reserved,cuda_cached_slack_bytes=reserved-allocated,
        cuda_driver_free_bytes=free,cuda_driver_total_bytes=total,cuda_driver_used_minus_torch_reserved_bytes=total-free-reserved,
        no_limit_change=True,no_cache_clearing=True,scope='logical tensor/mapping sizes differ from physical RSS/PSS; driver-used residual is not uniquely attributable')
    return report
