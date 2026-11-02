"""Read-only ownership diagnosis for the extra 128 MiB allocation."""
import gc
import hashlib
import json
import types


def diagnose(session,source,manifest):
    torch=session.torch;gc.collect();torch.cuda.synchronize()
    report=dict(status='PASS_E38_MEMORY_DIAGNOSTIC',source_sha256=manifest,
        allocated_bytes=torch.cuda.memory_allocated(),tensors=[],model_reloaded=False,
        target_object_id=id(session.weights),draft_object_id=id(session.draft))
    objects=gc.get_objects()
    for obj in objects:
        if type(obj) is not torch.Tensor or obj.device.type!='cuda':continue
        if obj.numel()*obj.element_size()!=128*1024**2:continue
        row=dict(shape=list(obj.shape),dtype=str(obj.dtype),referrers=[])
        for ref in gc.get_referrers(obj):
            if ref is objects:continue
            item=dict(type=type(ref).__name__)
            if isinstance(ref,dict):
                item['module']=ref.get('__name__') if isinstance(ref.get('__name__'),str) else None
                item['keys']=[str(k) for k,v in ref.items() if v is obj][:15]
            if isinstance(ref,types.CellType):
                item['closure_functions']=[]
                for tup in gc.get_referrers(ref):
                    if type(tup) is tuple:
                        for fn in gc.get_referrers(tup):
                            if isinstance(fn,types.FunctionType):item['closure_functions'].append(dict(name=fn.__qualname__,module=fn.__module__))
            row['referrers'].append(item)
        report['tensors'].append(row)
    report['private_modules']=[n for n in __import__('sys').modules if n.startswith('_e38_')]
    (source/'result.json').write_text(json.dumps(report,indent=2)+'\n');return report
