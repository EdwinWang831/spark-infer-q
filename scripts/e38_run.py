"""Queued E38 phases; all model calls use the existing resident weights."""
from collections import Counter
from contextlib import ExitStack
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

SEED='246-e36-candidate-v1.json'
SEED_SHA='542d558988acb4c6a91c8b6dc8a73dbf567004036cf3ec973dadb16c0de19853'
NAMES=('code','english','mixed','new_json')


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def load(path,tag):
    spec=importlib.util.spec_from_file_location(tag,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[tag]=mod;spec.loader.exec_module(mod);return mod


def private_module(upstream,tag):
    if tag in sys.modules:raise ValueError('fresh private module tag required')
    mod=ModuleType(tag);mod.__package__=upstream.__package__;mod.__file__=upstream.__file__
    sys.modules[tag]=mod
    try:exec(compile(Path(upstream.__file__).read_bytes(),upstream.__file__,'exec'),mod.__dict__)
    except BaseException:
        sys.modules.pop(tag,None);raise
    return mod


def run(session,arguments):
    source=Path(arguments['source_root']);manifest=json.loads((source/'manifest.json').read_text())
    for n,h in manifest.items():assert digest(source/n)==h,n
    seal=json.loads((session.root/'artifacts/e33-online-seal.json').read_text())
    for n,h in seal['external_source_sha256'].items():
        assert digest(session.root.parent/'e2-load/reference/tensorfold'/n)==h,n
    torch=session.torch
    assert not torch.backends.cuda.matmul.allow_tf32 and torch.get_float32_matmul_precision()=='highest'
    if arguments['phase']=='build':
        tag='_e38_build_'+digest(source/'scripts/e38_build.py')[:12]
        try:return load(source/'scripts/e38_build.py',tag).build(session,source,manifest)
        finally:sys.modules.pop(tag,None)
    if arguments['phase']=='component':
        tag='_e38_component_run_'+digest(source/'scripts/e38_component.py')[:12]
        try:result=load(source/'scripts/e38_component.py',tag).component(session,source,manifest,(digest,load,private_module))
        finally:sys.modules.pop(tag,None)
        cleanup_tag='_e38_component_cleanup_'+digest(source/'scripts/e38_cleanup.py')[:12]
        try:
            cleanup=load(source/'scripts/e38_cleanup.py',cleanup_tag)
            result['automatic_cleanup']=cleanup.detach_finished(session,source,result)
            (source/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            return result
        finally:sys.modules.pop(cleanup_tag,None)
    if arguments['phase']=='audit':
        tag='_e38_audit_'+digest(source/'scripts/e38_audit.py')[:12]
        try:
            mod=load(source/'scripts/e38_audit.py',tag)
            sys.modules.pop(tag,None)
            return mod.audit(session,source,manifest)
        finally:sys.modules.pop(tag,None)
    if arguments['phase']=='diagnose':
        tag='_e38_memory_'+digest(source/'scripts/e38_memory.py')[:12]
        try:return load(source/'scripts/e38_memory.py',tag).diagnose(session,source,manifest)
        finally:sys.modules.pop(tag,None)
    if arguments['phase']=='cleanup':
        tag='_e38_cleanup_'+digest(source/'scripts/e38_cleanup.py')[:12]
        try:return load(source/'scripts/e38_cleanup.py',tag).cleanup(session,source,manifest)
        finally:sys.modules.pop(tag,None)
    if arguments['phase']!='geometry':raise ValueError('phase implementation not staged yet')
    seed=session.root/'jobs/done'/SEED;assert digest(seed)==SEED_SHA
    old=json.loads(seed.read_text())['result']
    for n in ('scripts/e36_prefill_candidate.py','scripts/e36_diagnose.py'):
        assert manifest[n]==old['source_sha256'][n]
    inputs=[next(c for c in old['candidate_inputs'] if c['name']==n) for n in NAMES]
    tag='_e38_'+hashlib.sha256(str(source).encode()).hexdigest()[:12];modules=[]
    def owned(path,suffix):
        name=tag+'_'+suffix;modules.append(name);return load(source/path,name)
    adapter=owned('scripts/e36_prefill_candidate.py','adapter')
    helper=owned('scripts/e36_diagnose.py','helper');model=owned('scripts/e37_model.py','model')
    from tensorfold.families.qwen4_exp.cuda import nvfp4_moe,forward as upstream
    originals={n:getattr(upstream,n) for n in ('moe_block','gdn_block','_mm','_prefill_chain')}
    gc.collect();torch.cuda.synchronize();baseline=torch.cuda.memory_allocated()
    report=dict(status='RUNNING_E38_GEOMETRY',source_sha256=manifest,external_source_sha256=seal['external_source_sha256'],
        target_object_id=id(session.weights),draft_object_id=id(session.draft),model_reloaded=False,
        tf32_disabled=True,baseline_bytes=baseline,seed_job=SEED,seed_sha256=SEED_SHA,inputs=inputs,cases=[])
    path=source/'result.json'
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    try:
        with adapter.using_forward(session,tag+'_forward') as (f,_):
            modules.append(tag+'_moe');local=private_module(nvfp4_moe,tag+'_moe')
            nvx=ModuleType(tag+'_nvx');nvx.__dict__.update(local.nvx.__dict__);local.nvx=nvx
            times={}
            for name in ('gate_up','down'):
                original=getattr(nvx,name)
                def timed(*a,_name=name,_fn=original,**kw):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record();value=_fn(*a,**kw);end.record();times[_name]=(begin,end);return value
                setattr(nvx,name,timed)
            current={}
            original_moe=local.moe
            def observed(x,xs,router,ex,buf,cfg):
                original_moe(x,xs,router,ex,buf,cfg);torch.cuda.synchronize()
                picked=buf.pick[:len(x),:cfg.num_experts_per_tok].tolist()
                counts=Counter(e for row in picked for e in row)
                assert all(0<=e<ex.routed for e in counts)
                hist=Counter(counts.values());a=sum((v+15)//16 for v in counts.values());b=sum((v+31)//32 for v in counts.values())
                current['layers'].append(dict(layer=len(current['layers']),routed_pairs=sum(counts.values()),
                    active_experts=len(counts),rows_per_expert_histogram={str(k):v for k,v in sorted(hist.items())},
                    tiles16=a,tiles32=b,capacity16=a*16,capacity32=b*32,
                    experts_above16=sum(v>16 for v in counts.values()),
                    paired32_items=sum(v//32+int(v%32>16) for v in counts.values()),
                    gate_up_ms=times['gate_up'][0].elapsed_time(times['gate_up'][1]),
                    down_ms=times['down'][0].elapsed_time(times['down'][1])))
                return buf
            local.moe=observed;f.nvfp4_moe=local;h=model.PromptHarness(session,f)
            for case in inputs:
                current=dict(name=case['name'],length=case['length'],layers=[]);report['cases'].append(current);save()
                value=h.prompt(case['ids']);out=helper.tensor_hash(value[0],value[1])
                gold={r['output_sha256'] for r in old['timings'] if r['name']==case['name']}
                assert len(gold)==1 and out in gold and len(current['layers'])==48
                current.update(output_sha256=out,logits_features_exact=True)
                del value;save()
            layers=[l for c in report['cases'] for l in c['layers']]
            report['summary']=dict(layers=len(layers),tiles16=sum(l['tiles16'] for l in layers),
                tiles32=sum(l['tiles32'] for l in layers),paired32_items=sum(l['paired32_items'] for l in layers),
                routed_pairs=sum(l['routed_pairs'] for l in layers))
            report['summary']['potential_weight_stage_load_reduction']=1-report['summary']['tiles32']/report['summary']['tiles16']
            report.update(status='PASS_E38_GEOMETRY',upstream_functions_unchanged=all(getattr(upstream,n) is v for n,v in originals.items()))
            assert report['upstream_functions_unchanged'];save();return report
    finally:
        for name in modules:sys.modules.pop(name,None)
