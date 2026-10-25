"""Source-sealed direct256K resident execution and post-return audit."""
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
    tag='_e49_'+hashlib.sha256(str(source).encode()).hexdigest()[:12]
    pkg=ModuleType(tag);pkg.__path__=[str(source/'spark_infer_q')];sys.modules[tag]=pkg
    def load(name):
        key=tag+'.'+Path(name).stem;spec=importlib.util.spec_from_file_location(key,source/name)
        module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module);return module
    gc.collect();session.torch.cuda.synchronize()
    report=dict(status='RUNNING_E49',source_sha256=manifest,external_source_sha256=frozen['external_source_sha256'],
        target_object_id=id(session.weights),draft_object_id=id(session.draft),model_reloaded=False,
        baseline_bytes=session.torch.cuda.memory_allocated(),physical_capacity=262160,logical_capacity=262144,
        history_window_rows=2040,default_promoted=False,context_1m_validated=False)
    save(source,report)
    try:
        if args['phase']=='audit':result=audit(session,source,args)
        else:result=load('scripts/e49_'+args['phase']+'.py').execute(session,source,tag,args,load,report)
        report.update(result);save(source,report);return report
    except BaseException as exc:
        report.update(status='FAIL_E49_'+args['phase'].upper(),error_type=type(exc).__name__,error=str(exc)[:3000]);save(source,report);raise
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
        gc.collect();session.torch.cuda.synchronize()


def make_inputs(session):
    texts=dict(english64='Explain how a cache differs from a queue. ',boundary2049='Write a Python function that keeps unique items in order and state its invariant. ',
        mixed3073='A cache reuses computed results. 缓存保存可复用的结果，队列保存待处理任务。 Compare both mechanisms. ',
        code4096='Write a Python function that keeps unique items in order and state its invariant. ')
    result=[]
    for name,n in (('english64',64),('boundary2049',2049),('mixed3073',3073),('code4096',4096)):
        ids=session.tokenizer.encode(texts[name]*1000,add_special_tokens=False)[:n];assert len(ids)==n
        result.append(dict(name=name,length=n,ids=ids,token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
    return result


def expert(session,tag):
    module=importlib.import_module(tag+'.prefill_experts')
    p=session.root.parent/'e44-prefill-20261024-build-v1/build/binary/spark_q_e44_pipeline_v1.so'
    digest='a5b15bc9ad18e7aea4008f3aba0210856b9b4a4bd764580cbd987b64659b0977'
    return module.PrefillExpertTile.from_file(p,expected_sha256=digest),digest


def audit(session,source,args):
    import torch.distributed as dist
    from tensorfold.families.qwen4_exp.cuda import forward,nvfp4_moe
    p=session.root/'jobs/done'/args['study_job'];job=json.loads(p.read_text())
    stage=session.root.parent/('e49-long-20261025-'+args['study_version']);result=json.loads((stage/'result.json').read_text())
    assert job['worker_pid']==os.getpid() and job['load_count']==1
    assert result['target_object_id']==id(session.weights) and result['draft_object_id']==id(session.draft)
    manifest=json.loads((stage/'manifest.json').read_text());assert manifest==result['source_sha256']
    for name,digest in manifest.items():assert sha(stage/name)==digest,name
    assert sha(session.root/('e49_job_'+args['study_version'].replace('-','_')+'.py'))==manifest['scripts/e49_run.py']
    assert not dist.is_initialized() and os.environ.get('DEEPINFER_USE_NEXTN_DECODE') is None
    env=sys.modules.get('deepinfer.environ');assert env is None or env.envs.DEEPINFER_USE_NEXTN_DECODE==0
    for module,names in ((forward,('_mm','_prefill_chain','layer_forward','stage','commit')),(nvfp4_moe,('moe',))):
        expected={c.co_name:c for c in compile(Path(module.__file__).read_bytes(),module.__file__,'exec').co_consts if isinstance(c,CodeType)}
        for name in names:
            fn=getattr(module,name);inner=inspect.unwrap(fn)
            assert inner.__globals__ is module.__dict__ and inner.__module__==module.__name__ and inner.__code__==expected[name]
            if inner is not fn:
                c=inspect.getclosurevars(fn).nonlocals;assert name=='commit' and c['func'] is inner and type(c['ctx_factory'].__self__).__name__=='no_grad'
    oldtag='_e49_'+hashlib.sha256(str(stage).encode()).hexdigest()[:12]
    assert not any(n==oldtag or n.startswith(oldtag+'.') for n in sys.modules)
    gc.collect();session.torch.cuda.synchronize();allocated=session.torch.cuda.memory_allocated()
    assert allocated<=result['baseline_bytes'],(allocated,result['baseline_bytes'])
    return dict(status='PASS_E49_AUDIT',audited_job_status=job['status'],audited_phase_status=result['status'],
        study_job_sha256=sha(p),study_baseline_bytes=result['baseline_bytes'],allocated_bytes=allocated,
        same_resident_objects=True,original_operators_unchanged=True,native_group_released=True,
        native_environment_restored=True,owned_modules_released=True)
