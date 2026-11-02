"""One-variable wider owned prefill screen over the resident model."""
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from types import CodeType,ModuleType


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(source,result):
    path=source/'result.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def load_helpers(source,tag):
    spec=importlib.util.spec_from_file_location(tag+'.helpers',source/'scripts/e41_run.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def prompt_gate(rows):
    arms=('chunk512','chunk1024');longs=('code1024','mixed1536');cases={}
    assert len(rows)==32 and all(r['name'] in ('english64',)+longs and r['mode'] in arms for r in rows)
    assert all(r['exact'] is True and type(r['repetition']) is int and
               type(r['wall_ms']) in (int,float) and math.isfinite(r['wall_ms']) and
               0<r['wall_ms']<1e6 for r in rows)
    for name in ('english64',)+longs:
        group=[r for r in rows if r['name']==name];count=4 if name=='english64' else 6
        assert len(group)==count*2
        for mode in arms:
            assert sorted(r['repetition'] for r in group if r['mode']==mode)==list(range(count))
        costs={mode:sum(r['wall_ms'] for r in group if r['mode']==mode) for mode in arms}
        cases[name]=dict(costs=costs,reduction=1-costs[arms[1]]/costs[arms[0]])
    total={mode:sum(cases[n]['costs'][mode] for n in longs) for mode in arms}
    wins=sum(sum(r['wall_ms'] for r in rows if r['name'] in longs and r['repetition']==rep and r['mode']==arms[1])<
             sum(r['wall_ms'] for r in rows if r['name'] in longs and r['repetition']==rep and r['mode']==arms[0]) for rep in range(6))
    reduction=1-total[arms[1]]/total[arms[0]]
    def within(value,limit):return value<=limit or math.isclose(value,limit,rel_tol=1e-12,abs_tol=0.)
    return dict(passed=within(total[arms[1]],.97*total[arms[0]]) and wins>=4 and
                all(within(cases[n]['costs'][arms[1]],1.02*cases[n]['costs'][arms[0]]) for n in longs) and
                within(cases['english64']['costs'][arms[1]],1.03*cases['english64']['costs'][arms[0]]),per_case=cases,long_total_ms=total,
                long_reduction=reduction,long_repetition_wins=wins)


def run(session,args):
    source=Path(args['source_root']);manifest=json.loads((source/'manifest.json').read_text())
    for path,digest in manifest.items():assert sha(source/path)==digest,path
    frozen=json.loads((session.root/'artifacts/e33-online-seal.json').read_text())
    for path,digest in frozen['external_source_sha256'].items():
        assert sha(session.root.parent/'e2-load/reference/tensorfold'/path)==digest,path
    assert not session.torch.backends.cuda.matmul.allow_tf32
    assert session.torch.get_float32_matmul_precision()=='highest'
    tag='_e42_'+hashlib.sha256(str(source).encode()).hexdigest()[:12]
    package=ModuleType(tag);package.__path__=[str(source/'spark_infer_q')];sys.modules[tag]=package
    try:
        if args['phase']=='audit':result=audit(session,source,manifest,args)
        else:result=study(session,source,tag,manifest)
        result.update(source_sha256=manifest,external_source_sha256=frozen['external_source_sha256'],model_reloaded=False)
        save(source,result);return result
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
        gc.collect();session.torch.cuda.synchronize()


def study(session,source,tag,manifest):
    torch=session.torch;gc.collect();torch.cuda.synchronize();baseline=torch.cuda.memory_allocated()
    report=dict(status='RUNNING_E42',source_sha256=manifest,baseline_bytes=baseline,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),numeric=[],timings=[],
        requests=[],lifecycle=[],http=[],physical_capacity=2048,logical_capacity=1792,
        token_budget=1024,output_cap=64,default_promoted=False,production_bound_changed=False)
    helpers=load_helpers(source,tag);prefill=importlib.import_module(tag+'.prefill')
    ops=importlib.import_module(tag+'.prefill_tensorfold');graph=importlib.import_module(tag+'.kaifra_graph')
    from e3_wide_target import WideTarget
    class WiderProvider(ops.TensorFoldPrefillProvider):max_chunk_rows=1024
    executors={mode:prefill.PrefillExecutor(WiderProvider(session.weights,torch,capacity=2048),chunk_rows=rows)
               for mode,rows in (('chunk512',512),('chunk1024',1024))}
    target=None
    try:
        with torch.inference_mode():
            texts={'english64':'Explain how a cache differs from a queue. ',
                'code513':'Write Python code that preserves the order of unique items. Explain the invariant and test edge cases. ',
                'code1024':'Write Python code that preserves the order of unique items. Explain the invariant and test edge cases. ',
                'mixed1025':'A cache stores reusable results. 缓存保存重复计算结果。 Compare a queue with a cache. ',
                'chinese1536':'请解释事务的原子性和隔离性，分析两个请求同时修改同一条记录，并举例。 ',
                'mixed1536':'A cache stores reusable results. 缓存保存重复计算结果。 Compare a queue with a cache. '}
            cases={}
            for name,n in (('english64',64),('code513',513),('code1024',1024),('mixed1025',1025),('chinese1536',1536),('mixed1536',1536)):
                ids=session.tokenizer.encode(texts[name]*400,add_special_tokens=False)[:n];assert len(ids)==n
                cases[name]=dict(name=name,length=n,ids=ids,token_sha256=helpers.token_sha(ids))
            report['inputs']=[{k:v for k,v in c.items() if k!='ids'} for c in cases.values()]
            save(source,report)
            target=graph.GraphTarget(WideTarget(session.weights,capacity=2048));target.warmup()
            control=WideTarget(session.weights,capacity=2048);buckets=set()
            for name in ('english64','code513','code1024','mixed1025','chinese1536'):
                case=cases[name];report['active']='numeric:'+name;save(source,report)
                a=control.state();b=target.state()
                try:
                    x=executors['chunk512'].advance(a,case['ids']);y=executors['chunk1024'].advance(b,case['ids'])
                    row=dict(name=name,length=case['length'],chunks512=list(x.chunks),chunks1024=list(y.chunks),
                        logits_exact=torch.equal(x.logits,y.logits),features_exact=torch.equal(x.features,y.features),
                        state_differences=control.state_difference(a,b),windows=[])
                    report['numeric'].append(row)
                    if not (row['logits_exact'] and row['features_exact'] and not row['state_differences']):
                        for label,left,right in (('logits',x.logits,y.logits),('features',x.features,y.features)):
                            row[label+'_mismatches']=int((left!=right).sum().item())
                            row[label+'_max_abs']=float((left.float()-right.float()).abs().max().item())
                        report['status']='STOP_E42_NUMERICS';save(source,report);return report
                    for width in range(1,9):
                        ids=[case['ids'][i] for i in range(width)];keep=width if width%2 else width//2
                        parity=b.cur[0];replays=target.replays
                        lg,ft,buf=control.run(a,ids);control.commit(a,buf,width,keep)
                        g,f,buffer=target.run(b,ids);target.commit(b,buffer,width,keep)
                        differences=control.state_difference(a,b)
                        exact=torch.equal(lg,g) and torch.equal(ft,f) and not differences
                        row['windows'].append(dict(width=width,keep=keep,parity=parity,exact=exact,state_differences=differences))
                        assert exact and target.replays==replays+1
                        buckets.add((width,parity))
                    del x,y,lg,ft,buf,g,f,buffer
                finally:target.release_state(b)
                del a,b;gc.collect();save(source,report)
            assert buckets=={(r,p) for r in range(1,9) for p in (0,1)}
            report['numeric_graph']=dict(target.capabilities,buckets_checked=[list(b) for b in sorted(buckets)])
            target.close();target=None
            # Allocate/warm both workspaces before timing. State creation remains inside timing.
            for name in ('code1024','mixed1536','english64'):
                case=cases[name];report['active']='prompt:'+name;save(source,report)
                reference=None
                for mode in ('chunk512','chunk1024'):
                    state=control.state();value=executors[mode].advance(state,case['ids'])
                    if reference is None:reference=(value.logits,value.features,state.clone())
                    else:assert torch.equal(reference[0],value.logits) and torch.equal(reference[1],value.features) and not control.state_difference(reference[2],state)
                    del state,value
                for rep in range(4 if name=='english64' else 6):
                    for mode in (('chunk512','chunk1024') if rep%2==0 else ('chunk1024','chunk512')):
                        torch.cuda.synchronize();start=time.perf_counter()
                        state=control.state();value=executors[mode].advance(state,case['ids'])
                        elapsed=(time.perf_counter()-start)*1000
                        exact=torch.equal(reference[0],value.logits) and torch.equal(reference[1],value.features) and not control.state_difference(reference[2],state)
                        assert exact
                        report['timings'].append(dict(name=name,mode=mode,repetition=rep,wall_ms=elapsed,exact=exact,prompt_tokens=case['length']))
                        del value,state
                    save(source,report)
                del reference;gc.collect()
            report['prompt_summary']=prompt_gate(report['timings']);save(source,report)
            if not report['prompt_summary']['passed']:
                report['status']='STOP_E42_COST_GATE';save(source,report);return report
            optimized=importlib.import_module(tag+'.kaifra_optimized')
            target=optimized.optimized_target(session);provider,metadata=optimized.optimized_provider(session)
            report['profile']=metadata;target.warmup()
            spec=importlib.import_module(tag+'.kaifra_speculative');native_mod=importlib.import_module(tag+'.native_scheduler')
            requests=helpers.inputs(session)[1][:2]+[cases[n] for n in ('code1024','mixed1536')]
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                def make(mode):
                    return spec.speculative_engine(native,target,provider,eos_ids=session.eos,
                        prior=session.calibration['conditional_prior'],mode='adaptive',capacity=1792,max_requests=1,
                        token_budget=1024,tokenizer=session.tokenizer,prefill_executor=executors['chunk1024' if mode=='owned' else mode])
                natural={}
                for case in requests:
                    reference=None;report['active']='requests:'+case['name'];save(source,report)
                    for rep in range(-1,4):
                        for mode in (('chunk512','chunk1024') if rep%2==0 else ('chunk1024','chunk512')):
                            rec,features,state,tokens=helpers.generate(session,target,make(mode),case['ids'],rep,mode)
                            rec.update(name=case['name'],prompt_tokens=case['length'],token_sha256=case['token_sha256'])
                            # Prompt segmentation differs intentionally; compare complete decode trace.
                            decode_trace=[m for m in rec['trace'] if not m['prompt']]
                            if reference is None:reference=(tokens,features,state.clone(),decode_trace,rec['positions'])
                            gold,ft,st,trace,positions=reference
                            exact=tokens==gold and torch.equal(features,ft) and not target.state_difference(st,state) and decode_trace==trace and rec['positions']==positions
                            assert exact
                            rec.update(outputs_exact=True,features_exact=True,state_exact=True,trace_exact=True)
                            if rep>=0:
                                assert rec['captures_before']==rec['captures_after']==16 and rec['kernels_before']==rec['kernels_after'] and not rec['graph_fallbacks'] and rec['head_fallbacks']==0
                            report['requests'].append(rec);save(source,report)
                    natural[case['name']]=reference[0];del reference,features,state,ft,st
                helpers.lifecycle(session,source,report,target,make,requests[1])
                helpers.http_checks(session,source,report,target,make,requests[1],natural['zh'],tag)
            report['optimized_graph']=target.capabilities
            report['status']='PASS_E42_STUDY';report.pop('active',None);save(source,report)
    except BaseException as exc:
        report.update(status='FAIL_E42_STUDY',error_type=type(exc).__name__,error=str(exc)[:2000]);save(source,report);raise
    finally:
        for executor in executors.values():executor.close()
        if target is not None:
            if target.leased:target.release_state(target.workspace)
            target.close()
        del target,executors
        gc.collect();torch.cuda.synchronize()
    return report


def audit(session,source,manifest,args):
    import torch.distributed as dist
    from deepinfer.environ import envs
    from tensorfold.families.qwen4_exp.cuda import forward
    jobpath=session.root/'jobs/done'/args['study_job'];job=json.loads(jobpath.read_text())
    assert job['status']=='PASS' and job['load_count']==1
    result=job['result'];assert result['status'] in ('PASS_E42_STUDY','STOP_E42_NUMERICS','STOP_E42_COST_GATE')
    stage=session.root.parent/('e42-prefill-20261024-'+args['study_version'])
    sealed=json.loads((stage/'manifest.json').read_text());assert sealed==result['source_sha256']
    for name,digest in sealed.items():
        assert sha(stage/name)==digest,name
        if name.startswith('spark_infer_q/'):assert manifest[name]==digest,name
    assert sha(session.root/('e42_job_'+args['study_version'].replace('-','_')+'.py'))==sealed['scripts/e42_run.py']
    assert id(session.weights)==result['target_object_id'] and id(session.draft)==result['draft_object_id']
    assert not dist.is_initialized() and os.environ.get('DEEPINFER_USE_NEXTN_DECODE') is None and envs.DEEPINFER_USE_NEXTN_DECODE==0
    compiled=compile(Path(forward.__file__).read_bytes(),forward.__file__,'exec')
    expected={c.co_name:c for c in compiled.co_consts if isinstance(c,CodeType)}
    for name in ('_mm','_prefill_chain','layer_forward','stage','commit'):
        fn=getattr(forward,name);inner=inspect.unwrap(fn)
        assert inner.__globals__ is forward.__dict__ and inner.__module__==forward.__name__ and inner.__code__==expected[name]
        if inner is not fn:
            c=inspect.getclosurevars(fn).nonlocals
            assert name=='commit' and c['func'] is inner and type(c['ctx_factory'].__self__).__name__=='no_grad'
    oldtag='_e42_'+hashlib.sha256(str(stage).encode()).hexdigest()[:12]
    assert not any(n==oldtag or n.startswith(oldtag+'.') for n in sys.modules)
    gc.collect();session.torch.cuda.synchronize();allocated=session.torch.cuda.memory_allocated()
    assert allocated<=result['baseline_bytes'],(allocated,result['baseline_bytes'])
    return dict(status='PASS_E42_AUDIT',study_status=result['status'],allocated_bytes=allocated,
        baseline_bytes=result['baseline_bytes'],study_job_sha256=sha(jobpath),same_resident_objects=True,
        native_group_released=True,native_environment_restored=True,forward_code_matches_frozen=True,
        owned_modules_released=True,validated_library_unchanged=True)
