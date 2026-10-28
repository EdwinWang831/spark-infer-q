"""Source-sealed serialized E57 runner; reuse existing resident objects."""
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import ModuleType,CodeType


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(source,result):
    p=source/'result.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');t.replace(p)


def run(session,args):
    source=Path(args['source_root']);manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in manifest.items():assert sha(source/name)==digest,name
    frozen=json.loads((session.root/'artifacts/e33-online-seal.json').read_text())
    for name,digest in frozen['external_source_sha256'].items():
        assert sha(session.root.parent/'e2-load/reference/tensorfold'/name)==digest,name
    tag='_e57_'+hashlib.sha256(str(source).encode()).hexdigest()[:12]
    pkg=ModuleType(tag);pkg.__path__=[str(source/'spark_infer_q')];sys.modules[tag]=pkg
    def load(name):
        key=tag+'.'+Path(name).stem;spec=importlib.util.spec_from_file_location(key,source/name)
        module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module);return module
    gc.collect();session.torch.cuda.synchronize()
    report=dict(status='RUNNING_E57',source_sha256=manifest,external_source_sha256=frozen['external_source_sha256'],
        worker_pid=os.getpid(),load_count=1,target_object_id=id(session.weights),draft_object_id=id(session.draft),
        model_reloaded=False,baseline_bytes=session.torch.cuda.memory_allocated(),default_promoted=False,
        eight_execution_validated=False,fp8_bf16_quality_parity=False)
    save(source,report)
    try:
        if args['phase']=='audit':result=audit(session,source,args)
        else:result=load('scripts/e57_'+args['phase']+'.py').execute(session,source,tag,args,load,report)
        report.update(result);save(source,report);return report
    except BaseException as exc:
        report.update(status='FAIL_E57_'+args['phase'].upper(),error_type=type(exc).__name__,error=str(exc)[:3000])
        save(source,report);raise
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
        gc.collect();session.torch.cuda.synchronize()


def audit(session,source,args):
    import torch.distributed as dist
    from tensorfold.families.qwen4_exp.cuda import forward,nvfp4_moe
    path=session.root/'jobs/done'/args['study_job'];job=json.loads(path.read_text())
    assert job['status']=='PASS', 'upstream job failed; no passing experiment audit'
    result=job['result']
    assert job['status']=='PASS' and job['worker_pid']==os.getpid() and job['load_count']==1
    assert result['target_object_id']==id(session.weights) and result['draft_object_id']==id(session.draft)
    assert not dist.is_initialized() and os.environ.get('DEEPINFER_USE_NEXTN_DECODE') is None
    for module,names in ((forward,('_mm','_prefill_chain','layer_forward','attn_block','stage','compute','commit')),(nvfp4_moe,('moe',))):
        expected={c.co_name:c for c in compile(Path(module.__file__).read_bytes(),module.__file__,'exec').co_consts if isinstance(c,CodeType)}
        for name in names:
            fn=inspect.unwrap(getattr(module,name))
            assert fn.__globals__ is module.__dict__ and fn.__code__==expected[name],name
    gc.collect();session.torch.cuda.synchronize();allocated=session.torch.cuda.memory_allocated()
    assert allocated==result['baseline_bytes'],(allocated,result['baseline_bytes'])
    studied=session.root.parent/('e57-affine-20261028-'+args['study_job'].split('-e57-',1)[1][:-5])
    assert result==json.loads((studied/'result.json').read_text())
    prior_tag='_e57_'+hashlib.sha256(str(studied).encode()).hexdigest()[:12]
    assert not any(n==prior_tag or n.startswith(prior_tag+'.') for n in sys.modules)
    return dict(status='PASS_E57_AUDIT',audited_report_sha256=sha(studied/'result.json'),audited_job=args['study_job'],audited_job_sha256=sha(path),
        audited_status=result['status'],allocated_bytes=allocated,original_operators_unchanged=True,
        same_resident_objects=True,owned_modules_released=True)
