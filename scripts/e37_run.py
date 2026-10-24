"""Same-resident E36 prompt profiling and one-variable GDN scratch experiment."""
from contextlib import ExitStack
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


FROZEN_JOB='246-e36-candidate-v1.json'
FROZEN_SHA='542d558988acb4c6a91c8b6dc8a73dbf567004036cf3ec973dadb16c0de19853'
NAMES=('code','english','mixed','new_json')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path,tag):
    spec=importlib.util.spec_from_file_location(tag,path);module=importlib.util.module_from_spec(spec)
    sys.modules[tag]=module;spec.loader.exec_module(module);return module


def run(session,arguments):
    source=Path(arguments['source_root']);manifest=json.loads((source/'manifest.json').read_text())
    for n,h in manifest.items():assert digest(source/n)==h,n
    frozen=json.loads((session.root/'artifacts/e33-online-seal.json').read_text())
    for n,h in frozen['external_source_sha256'].items():
        assert digest(session.root.parent/'e2-load/reference/tensorfold'/n)==h,n
    torch=session.torch
    assert not torch.backends.cuda.matmul.allow_tf32 and torch.get_float32_matmul_precision()=='highest'
    if arguments['phase']=='audit':return audit(session,source,manifest)
    seed=session.root/'jobs/done'/FROZEN_JOB;assert digest(seed)==FROZEN_SHA
    old=json.loads(seed.read_text())['result']
    for n in ('scripts/e36_prefill_candidate.py','scripts/e36_diagnose.py'):
        assert manifest[n]==old['source_sha256'][n],n
    cases=[next(c for c in old['candidate_inputs'] if c['name']==name) for name in NAMES]
    gold_hash={c['name']:next(r['output_sha256'] for r in old['timings'] if r['name']==c['name']) for c in cases}
    tag='_e37_'+hashlib.sha256(str(source).encode()).hexdigest()[:12]
    modules=[]
    def owned(path,suffix):
        name=tag+'_'+suffix;modules.append(name);return load(source/path,name)
    adapter=owned('scripts/e36_prefill_candidate.py','adapter')
    helpers=owned('scripts/e36_diagnose.py','helpers')
    model=owned('scripts/e37_model.py','model')
    pool_module=owned('spark_infer_q/prefill_scratch.py','pool')
    account=owned('scripts/e37_accounting.py','account')
    from tensorfold.families.qwen4_exp.cuda import forward as upstream
    from tensorfold.families.qwen4_exp.cuda.state import Buffers
    from e3_wide_target import WideTarget
    originals={n:getattr(upstream,n) for n in ('gdn_block','_mm','moe_block','_prefill_chain')}
    gc.collect();torch.cuda.synchronize();baseline=torch.cuda.memory_allocated()
    result=dict(status='RUNNING_E37',source_sha256=manifest,external_source_sha256=frozen['external_source_sha256'],
        seed_job=FROZEN_JOB,seed_sha256=FROZEN_SHA,inputs=cases,checks=[],timings=[],profiles=[],
        target_object_id=id(session.weights),draft_object_id=id(session.draft),model_reloaded=False,
        tf32_disabled=True,baseline_bytes=baseline,default_promoted=False,timing_scope='prompt_only_token_API')
    path=source/'result.json'
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    def difference(a,b):
        return WideTarget.state_difference(type('Owner',(),dict(torch=torch))(),a,b)
    try:
        with ExitStack() as stack:
            fresh,_=stack.enter_context(adapter.using_forward(session,tag+'_fresh'))
            pooled,_=stack.enter_context(adapter.using_forward(session,tag+'_pooled'))
            pool=model.install_pool(session,pooled,pool_module,[c['length'] for c in cases])
            harnesses=dict(fresh=model.PromptHarness(session,fresh),pool=model.PromptHarness(session,pooled))
            tensors=[getattr(pool.scratch,n) for n in ('k','v','g','b')]
            result['scratch']=dict(prepared_rows=pool.rows,storage_bytes=sum(t.numel()*t.element_size() for t in tensors),
                prepared_views=sorted(pool.views),storage_instances=1,fresh_allocations_per_prompt=36)
            gold_states={};decoder_buffer=Buffers(session.weights,8,1024,prefill=False)
            for c in cases:
                result['active']=c['name']+':correctness';save()
                a=harnesses['fresh'].prompt(c['ids']);b=harnesses['pool'].prompt(c['ids'])
                assert helpers.tensor_hash(a[0],a[1])==gold_hash[c['name']], 'fresh output differs from sealed E36'
                assert helpers.tensor_hash(b[0],b[1])==gold_hash[c['name']], 'pooled output differs from sealed E36'
                assert not difference(a[2],b[2]),'pooled state mismatch'
                gold_states[c['name']]=a[2].clone()
                row=dict(name=c['name'],length=c['length'],token_sha256=c['token_sha256'],
                    full_logits_exact=True,all_features_exact=True,state_exact=True,
                    output_sha256=gold_hash[c['name']],followup=[])
                anchor=a[3]
                for i in range(8):
                    left=model.decode_piece(session,upstream,a[2],decoder_buffer,anchor)
                    right=model.decode_piece(session,upstream,b[2],decoder_buffer,anchor)
                    assert torch.equal(left[0],right[0]) and torch.equal(left[1],right[1]) and not difference(a[2],b[2])
                    row['followup'].append(dict(step=i,exact=True));anchor=int(left[0].argmax())
                result['checks'].append(row);del a,b,left,right;gc.collect();save()
            for ci,c in enumerate(cases):
                for h in harnesses.values():
                    warm=h.prompt(c['ids']);del warm
                for repetition in range(6):
                    order=('fresh','pool') if (ci+repetition)%2==0 else ('pool','fresh')
                    for mode in order:
                        result['active']=dict(case=c['name'],mode=mode,repetition=repetition);save()
                        calls=pool.calls;torch.cuda.synchronize();start=time.perf_counter()
                        value=harnesses[mode].prompt(c['ids']);wall=(time.perf_counter()-start)*1000
                        output_hash=helpers.tensor_hash(value[0],value[1])
                        assert output_hash==gold_hash[c['name']] and not difference(value[2],gold_states[c['name']])
                        assert pool.calls-calls==(36 if mode=='pool' else 0)
                        result['timings'].append(dict(name=c['name'],repetition=repetition,mode=mode,
                            prompt_tokens=c['length'],wall_ms=wall,output_sha256=output_hash,state_exact=True))
                        del value
                    save()
                gc.collect()
            result['summary']=account.summarize_timings(result['timings'],cases)
            # Diagnostics are independent extra runs, never part of the paired speed table.
            for c in [x for x in cases if x['length']==512]:
                result['active']=c['name']+':events';save()
                rec=model.StageRecorder(torch);start=time.perf_counter()
                with rec.span('control'):
                    with model.instrument(fresh,rec):value=harnesses['fresh'].prompt(c['ids'],rec)
                event_wall=(time.perf_counter()-start)*1000
                groups=rec.result()
                assert helpers.tensor_hash(value[0],value[1])==gold_hash[c['name']] and not difference(value[2],gold_states[c['name']])
                del value
                result['active']=c['name']+':cupti';save()
                root='e37.prompt.'+c['name'];annotations=model.Annotations(torch)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False,profile_memory=False,with_stack=False) as prof:
                    with torch.profiler.record_function(root):
                        with model.instrument(fresh,annotations):value=harnesses['fresh'].prompt(c['ids'],annotations)
                trace=source/(c['name']+'-trace.json');prof.export_chrome_trace(str(trace))
                assert helpers.tensor_hash(value[0],value[1])==gold_hash[c['name']] and not difference(value[2],gold_states[c['name']])
                del value
                result['profiles'].append(dict(name=c['name'],event_instrumented_wall_ms=event_wall,
                    exclusive_stage_intervals=groups,trace=trace.name,trace_sha256=digest(trace),
                    root=root,cupti=account.analyze_trace(json.loads(trace.read_text()),root),exact=True))
                save()
            result.update(status='PASS_E37_PREFILL_PROFILE',pool_calls=pool.calls,
                upstream_functions_unchanged=all(getattr(upstream,n) is fn for n,fn in originals.items()))
            assert result['upstream_functions_unchanged'];result.pop('active',None);save()
            return result
    finally:
        for name in modules:sys.modules.pop(name,None)


def audit(session,source,manifest):
    job=session.root/'jobs/done/270-e37-measure-v1.json';data=json.loads(job.read_text())
    assert data['status']=='PASS'
    report=data['result'];assert report['status']=='PASS_E37_PREFILL_PROFILE'
    original=session.root.parent/'e37-prefill-20261024-measure-v1'
    for n,h in report['source_sha256'].items():assert digest(original/n)==h,n
    for k,obj in (('target_object_id',session.weights),('draft_object_id',session.draft)):
        assert report[k]==id(obj)
    frozen=json.loads((session.root/'artifacts/e33-online-seal.json').read_text())
    for key in ('source_sha256','input_sha256'):
        for n,h in frozen[key].items():assert digest(session.root/n)==h,n
    from tensorfold.families.qwen4_exp.cuda import forward
    assert forward.gdn_block.__module__==forward.__name__ and forward.gdn_block.__globals__ is forward.__dict__
    gc.collect();session.torch.cuda.synchronize();allocated=session.torch.cuda.memory_allocated()
    assert allocated<=report['baseline_bytes'],(allocated,report['baseline_bytes'])
    ext={n:digest(Path(sys.modules[n].__file__)) for n in ('spark_q_moe_linear','spark_q_moe_plan','spark_q_moe_prompt',
        'spark_q_block_gdn','spark_q_block_gdn_io','spark_q_block_gdn_shared')}
    return dict(status='PASS_E37_AUDIT',source_sha256=manifest,measurement_sha256=digest(job),
        baseline_bytes=report['baseline_bytes'],allocated_bytes=allocated,extensions_sha256=ext,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),model_reloaded=False,
        frozen_sources_verified=True,original_forward_restored=True,
        native_group_released=not session.torch.distributed.is_initialized())
