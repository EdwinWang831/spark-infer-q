"""Detach only the completed E38 private observer that retains its flush tensor."""
import gc
import hashlib
import json
from pathlib import Path
from types import FunctionType,ModuleType


def detach(namespace,original_moe,expected_name):
    if namespace.get('__name__')!=expected_name or namespace.get('_e36_closed') is not True:
        raise ValueError('only the closed owned forward may be detached')
    observed=namespace.get('nvfp4_moe');block=namespace.get('moe_block')
    if (not isinstance(observed,ModuleType) or observed.__name__!=expected_name.removesuffix('_forward')+'_observed'
            or not isinstance(block,FunctionType) or block.__qualname__!='component.<locals>.observed_block'):
        raise ValueError('unexpected observer ownership')
    cells=dict(zip(block.__code__.co_freevars,block.__closure__ or ()))
    base=cells['block'].cell_contents
    if base.__globals__ is not namespace or base.__name__!='moe_block':
        raise ValueError('original private function identity differs')
    namespace['moe_block']=base;namespace['nvfp4_moe']=original_moe


def detach_finished(session,folder,result):
    torch=session.torch
    from tensorfold.families.qwen4_exp.cuda import nvfp4_moe
    assert result['status']=='PASS_E38_COMPONENT'
    assert result['target_object_id']==id(session.weights) and result['draft_object_id']==id(session.draft)
    expected='_e38_component_'+hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()[:10]+'_forward'
    before=torch.cuda.memory_allocated();objects=gc.get_objects();count=0
    for obj in objects:
        if type(obj) is dict and obj.get('__name__')==expected:
            detach(obj,nvfp4_moe,expected);count+=1
    assert count in (0,1),count
    del objects,obj
    gc.collect();torch.cuda.synchronize();after=torch.cuda.memory_allocated()
    assert after<=result['baseline_bytes'],(before,after,result['baseline_bytes'])
    return dict(allocated_before=before,allocated_bytes=after,detached_closed_private_forwards=count,
                released_bytes=before-after,baseline_bytes=result['baseline_bytes'])


def cleanup(session,source,manifest):
    from tensorfold.families.qwen4_exp.cuda import nvfp4_moe
    folder=session.root.parent/'e38-prefill-20261024-component-v1'
    result=json.loads((folder/'result.json').read_text())
    details=detach_finished(session,folder,result)
    assert details['released_bytes']==128*1024**2
    report=dict(status='PASS_E38_OBSERVER_CLEANUP',source_sha256=manifest,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),model_reloaded=False,
        **details,
        original_module_unchanged=nvfp4_moe.moe.__globals__ is nvfp4_moe.__dict__)
    assert report['original_module_unchanged']
    (source/'result.json').write_text(json.dumps(report,indent=2)+'\n');return report
