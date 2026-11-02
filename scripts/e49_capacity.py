"""Measure real full-capacity layouts before risking a four-request allocation."""
import gc
import importlib
import json
from pathlib import Path

LIMIT=88*1024**3
RESERVE=8*1024**3


def tensors_bytes(torch,obj):
    seen=set();storages={}
    def visit(v):
        if id(v) in seen:return
        seen.add(id(v))
        if isinstance(v,torch.Tensor):
            if v.is_cuda:storages[v.untyped_storage().data_ptr()]=v.untyped_storage().nbytes()
        elif isinstance(v,dict):
            for a in v.values():visit(a)
        elif isinstance(v,(list,tuple)):
            for a in v:visit(a)
        elif hasattr(v,'__dict__'):
            for a in vars(v).values():visit(a)
    visit(obj)
    return sum(storages.values())


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e49_run.py');old=load('scripts/e46_study.py')
    long=importlib.import_module(tag+'.long_target');ops=importlib.import_module(tag+'.prefill_tensorfold')
    target=provider=workspace=state=None
    def snap():torch.cuda.synchronize();return old.memory(torch)
    report.update(status='RUNNING_E49_CAPACITY',slots_requested=4,kv_dtype='int8',allocation_limit_bytes=LIMIT,
        host_reserve_bytes=RESERVE,actual_long_requests_executed=0,quality_gate_promoted=False)
    try:
        with torch.inference_mode():
            before=snap();report['before']=before
            target=long.LongTarget(session.weights,capacity=262160,kv_dtype='int8')
            report['after_eager_buffers']=snap()
            expert,digest=helper.expert(session,tag);report['expert_binary_sha256']=digest
            provider=ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,expert_backend=expert,kv_dtype='int8')
            workspace=provider.allocate(1024)
            common=snap();report['after_common_prefill']=common
            state=target.state();one=snap();report['after_one_state']=one
            delta=one['allocated_bytes']-common['allocated_bytes']
            storage=tensors_bytes(torch,state)
            cache=sum(tensors_bytes(torch,c) for c in state.kc)
            index=sum(t.untyped_storage().nbytes() for t in state.ikc)+sum(t.untyped_storage().nbytes() for t in state.pooled)
            lower=common['allocated_bytes']+4*delta
            report.update(one_state_allocation_delta_bytes=delta,one_state_unique_storage_bytes=storage,
                one_state_kv_storage_bytes=cache,one_state_index_pool_storage_bytes=index,
                four_state_allocation_lower_bound_bytes=lower,
                four_state_projected_host_available_bytes=one['MemAvailable']-3*delta,
                excluded_from_lower_bound=['private decode Graph pools','screened head auxiliaries','four draft caches/features'],
                four_state_execution_validated=False)
            assert delta>=storage>0
            fits=lower<=LIMIT and report['four_state_projected_host_available_bytes']>=RESERVE
            report['capacity_lower_bound_fits']=fits
            report['status']='PASS_E49_CAPACITY_LOWER_BOUND' if fits else 'STOP_E49_FOUR_CAPACITY'
            helper.save(source,report)
    finally:
        state=workspace=provider=target=None
        gc.collect();torch.cuda.synchronize()
    report['after_release']=snap()
    assert report['after_release']['allocated_bytes']<=report['baseline_bytes']
    return report
