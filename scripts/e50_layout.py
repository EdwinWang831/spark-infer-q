"""Allocate one empty state per format to measure storage, never eight states."""
import gc
import json


def storage_bytes(torch,value):
    seen=set();storages={}
    def visit(v):
        if id(v) in seen:return
        seen.add(id(v))
        if isinstance(v,torch.Tensor):
            if v.is_cuda:storages[v.untyped_storage().data_ptr()]=v.untyped_storage().nbytes()
        elif isinstance(v,dict):
            for x in v.values():visit(x)
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
        elif hasattr(v,'__dict__'):
            for x in vars(v).values():visit(x)
    visit(value);return sum(storages.values())


def execute(session,source,tag,args,load,report):
    from tensorfold.families.qwen4_exp.cuda.state import State
    torch=session.torch;helper=load('scripts/e50_run.py');old=load('scripts/e46_study.py')
    p=session.root/'jobs/done/520-e49-capacity-v1.json';job=json.loads(p.read_text());base=job['result']
    assert job['status']=='PASS' and base['target_object_id']==id(session.weights)
    report.update(status='RUNNING_E50_STATE_LAYOUT',formats=[],source_capacity_job_sha256=helper.sha(p),
        quality_validated=False,inference_executed=False,eight_execution_validated=False,default_promoted=False,
        scope='one_empty_state_per_format; existing_external_int4_storage_only; common_workspace_basis_from_E49')
    for fmt in ('int8','fp8','int4'):
        old.guard(torch);before=torch.cuda.memory_allocated();state=None
        try:
            state=State(session.weights,262160,8,fmt,limit=262160);torch.cuda.synchronize()
            delta=torch.cuda.memory_allocated()-before;total=storage_bytes(torch,state)
            kv=sum(c.nbytes for c in state.kc)
            index=sum(t.untyped_storage().nbytes() for t in state.ikc)+sum(t.untyped_storage().nbytes() for t in state.pooled)
            report['formats'].append(dict(kv_dtype=fmt,state_allocation_bytes=delta,state_storage_bytes=total,
                kv_storage_bytes=kv,index_pool_bytes=index,
                eight_state_common_allocation_forecast_bytes=base['after_common_prefill']['allocated_bytes']+8*delta,
                after_allocation=old.memory(torch)))
            helper.save(source,report)
        finally:
            state=None;gc.collect();torch.cuda.synchronize()
        assert torch.cuda.memory_allocated()==before
    report.update(status='PASS_E50_STATE_LAYOUT',after_release_allocated_bytes=torch.cuda.memory_allocated())
    return report
