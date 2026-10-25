"""Source-sealed E44 build, component and audit phases on the resident queue."""
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import CodeType,ModuleType


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(source,result):
    p=source/'result.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');t.replace(p)


def run(session,args):
    source=Path(args['source_root']);manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in manifest.items():assert sha(source/name)==digest,name
    frozen=json.loads((session.root/'artifacts/e33-online-seal.json').read_text())
    for name,digest in frozen['external_source_sha256'].items():assert sha(session.root.parent/'e2-load/reference/tensorfold'/name)==digest,name
    assert not session.torch.backends.cuda.matmul.allow_tf32 and session.torch.get_float32_matmul_precision()=='highest'
    tag='_e44_'+hashlib.sha256(str(source).encode()).hexdigest()[:12]
    pkg=ModuleType(tag);pkg.__path__=[str(source/'spark_infer_q')];sys.modules[tag]=pkg
    def load(name):
        key=tag+'.'+Path(name).stem;spec=importlib.util.spec_from_file_location(key,source/name)
        module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module);return module
    gc.collect();session.torch.cuda.synchronize();baseline=session.torch.cuda.memory_allocated()
    common=dict(source_sha256=manifest,external_source_sha256=frozen['external_source_sha256'],
        target_object_id=id(session.weights),draft_object_id=id(session.draft),model_reloaded=False,baseline_bytes=baseline)
    try:
        if args['phase']=='build':result=load('scripts/e44_build.py').build(session,source,manifest)
        elif args['phase']=='component':result=load('scripts/e44_component.py').component(session,source,tag,args,load)
        elif args['phase']=='audit':result=audit(session,source,args)
        elif args['phase']=='integration':result=load('scripts/e44_integration.py').integration(session,source,tag,args,load)
        elif args['phase']=='study':result=load('scripts/e44_study.py').study(session,source,tag,args,load)
        else:raise ValueError('explicit phase required')
        result.update(common);save(source,result);return result
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
        gc.collect();session.torch.cuda.synchronize()


def audit(session,source,args):
    import torch.distributed as dist
    from tensorfold.families.qwen4_exp.cuda import forward,nvfp4_moe
    from tensorfold.cuda.nvfp4 import experts
    p=session.root/'jobs/done'/args['study_job'];job=json.loads(p.read_text());assert job['status']=='PASS' and job['load_count']==1
    r=job['result'];assert r['status'] in ('PASS_E44_COMPONENT','PASS_E44_STUDY','PASS_E44_BUILD','PASS_E44_INTEGRATION')
    stage=session.root.parent/('e44-prefill-20261024-'+args['study_version']);manifest=json.loads((stage/'manifest.json').read_text())
    assert manifest==r['source_sha256']
    for name,digest in manifest.items():assert sha(stage/name)==digest,name
    assert sha(session.root/('e44_job_'+args['study_version'].replace('-','_')+'.py'))==manifest['scripts/e44_run.py']
    assert id(session.weights)==r['target_object_id'] and id(session.draft)==r['draft_object_id']
    native_env=sys.modules.get('deepinfer.environ')
    assert not dist.is_initialized() and os.environ.get('DEEPINFER_USE_NEXTN_DECODE') is None
    assert native_env is None or native_env.envs.DEEPINFER_USE_NEXTN_DECODE==0
    for module,names in ((forward,('_mm','_prefill_chain','layer_forward','stage','commit')),(nvfp4_moe,('moe',)),(experts,('gate_up','down'))):
        expected={c.co_name:c for c in compile(Path(module.__file__).read_bytes(),module.__file__,'exec').co_consts if isinstance(c,CodeType)}
        for name in names:
            fn=getattr(module,name);inner=inspect.unwrap(fn)
            assert inner.__globals__ is module.__dict__ and inner.__module__==module.__name__ and inner.__code__==expected[name]
            if inner is not fn:
                c=inspect.getclosurevars(fn).nonlocals;assert name=='commit' and c['func'] is inner and type(c['ctx_factory'].__self__).__name__=='no_grad'
    oldtag='_e44_'+hashlib.sha256(str(stage).encode()).hexdigest()[:12]
    assert not any(n==oldtag or n.startswith(oldtag+'.') for n in sys.modules)
    gc.collect();session.torch.cuda.synchronize();allocated=session.torch.cuda.memory_allocated();assert allocated<=r['baseline_bytes'],(allocated,r['baseline_bytes'])
    return dict(status='PASS_E44_AUDIT',studied_status=r['status'],study_job_sha256=sha(p),allocated_bytes=allocated,
        study_baseline_bytes=r['baseline_bytes'],same_resident_objects=True,original_operators_unchanged=True,
        native_group_released=True,native_environment_restored=True,native_module_loaded=native_env is not None,owned_modules_released=True)
