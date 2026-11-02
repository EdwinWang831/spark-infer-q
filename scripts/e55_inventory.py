"""Read-only unique-storage ownership audit of the already loaded session."""
import gc
import importlib
from types import ModuleType,FunctionType,MethodType,BuiltinFunctionType


def inventory(torch,roots):
    records={}
    for label,root in roots.items():
        seen=set()
        def visit(value,path):
            if id(value) in seen:return
            seen.add(id(value))
            if isinstance(value,torch.Tensor):
                if value.is_cuda:
                    storage=value.untyped_storage();key=(str(value.device),storage.data_ptr())
                    item=records.setdefault(key,dict(bytes=storage.nbytes(),owners={},shapes=[]))
                    item['owners'].setdefault(label,[]).append(path)
                    shape=dict(shape=list(value.shape),dtype=str(value.dtype))
                    if shape not in item['shapes']:item['shapes'].append(shape)
            elif isinstance(value,(ModuleType,type,FunctionType,MethodType,BuiltinFunctionType)):return
            elif isinstance(value,dict):
                for key,child in value.items():visit(child,path+'.'+str(key))
            elif isinstance(value,(tuple,list)):
                for i,child in enumerate(value):visit(child,path+'.'+str(i))
            elif hasattr(value,'__dict__'):
                for key,child in vars(value).items():visit(child,path+'.'+key)
        visit(root,label)
    records=sorted(records.values(),key=lambda x:-x['bytes'])
    per_root={name:dict(reachable_bytes=sum(r['bytes'] for r in records if name in r['owners']),
        exclusive_bytes=sum(r['bytes'] for r in records if list(r['owners'])==[name])) for name in roots}
    return dict(unique_bytes=sum(r['bytes'] for r in records),storage_count=len(records),per_root=per_root,storages=records)


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e55_run.py')
    optimized=importlib.import_module(tag+'.kaifra_optimized')
    roots=dict(weights=session.weights,legacy_target=session.target,draft=session.draft,projection=session.projection)
    gc.collect();torch.cuda.synchronize();before=torch.cuda.memory_allocated()
    report.update(status='RUNNING_E55_INVENTORY',baseline_inventory=inventory(torch,roots),
        inference_executed=False,session_modified=False)
    helper.save(source,report)
    provider=None
    try:
        provider,_=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=262160,
            query_graphs=True,request_projection_cache=True)
        torch.cuda.synchronize()
        roots['optimized_draft']=provider.worker
        report.update(optimized_inventory=inventory(torch,roots),
            optimized_provider_allocation_bytes=torch.cuda.memory_allocated()-before,
            packed_mlp=provider.worker.w8_manifest)
        report['status']='PASS_E55_INVENTORY';helper.save(source,report)
    finally:
        roots.pop('optimized_draft',None)
        if provider is not None:provider.worker.close()
        provider=None;gc.collect();torch.cuda.synchronize()
    report['after_release_allocated_bytes']=torch.cuda.memory_allocated()
    assert report['after_release_allocated_bytes']==before
    return report
