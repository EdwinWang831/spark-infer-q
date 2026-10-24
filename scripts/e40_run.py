"""Resident validation of the owned prefill module, never reloads weights."""
import gc
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import ModuleType


def run(session, arguments):
    source = Path(arguments['source_root'])
    manifest = json.loads((source/'manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    frozen = json.loads((session.root/'artifacts/e33-online-seal.json').read_text())
    external = session.root.parent/'e2-load/reference/tensorfold'
    for name, digest in frozen['external_source_sha256'].items():
        assert hashlib.sha256((external/name).read_bytes()).hexdigest() == digest, name
    tag = '_e40_'+hashlib.sha256(str(source).encode()).hexdigest()[:12]
    pkg = ModuleType(tag); pkg.__path__ = [str(source/'spark_infer_q')]; sys.modules[tag] = pkg
    try:
        if arguments.get('phase')=='inspect':
            import inspect,marshal
            from tensorfold.families.qwen4_exp.cuda import forward
            functions={}
            for name in ('_mm','_prefill_chain','layer_forward','stage','commit'):
                fn=getattr(forward,name)
                functions[name]=dict(module=fn.__module__,globals_module=fn.__globals__.get('__name__'),
                    qualname=fn.__qualname__,code_sha256=hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest(),
                    globals_are_upstream=fn.__globals__ is forward.__dict__,
                    closure_types={k:type(v).__name__ for k,v in inspect.getclosurevars(fn).nonlocals.items()})
            return dict(status='PASS_E40_INSPECT',functions=functions,allocated_bytes=session.torch.cuda.memory_allocated())
        if arguments.get('phase') == 'audit':
            import torch.distributed as dist
            import os
            import inspect
            from types import CodeType
            from deepinfer.environ import envs
            from tensorfold.families.qwen4_exp.cuda import forward
            done=session.root/'jobs/done'
            job=json.loads((done/'334-e40-validate-v2.json').read_text())
            assert job['status']=='PASS' and job['load_count']==1
            result=job['result'];assert result['status']=='PASS_E40_EXECUTOR'
            old=session.root.parent/'e40-prefill-20261024-validate-v2'
            seal=json.loads((old/'manifest.json').read_text())
            assert result['source_sha256']==seal
            assert hashlib.sha256((session.root/'e40_job_validate_v2.py').read_bytes()).hexdigest()==seal['scripts/e40_run.py']
            for name,digest in seal.items():
                assert hashlib.sha256((old/name).read_bytes()).hexdigest()==digest,name
                if name.startswith('spark_infer_q/'):
                    assert manifest[name]==digest, 'audited library differs: '+name
            requests_path=done/'344-e40-requests-v2.json'
            if requests_path.exists():
                measured=json.loads(requests_path.read_text())
                assert measured['status']=='PASS' and measured['result']['status']=='PASS_E40_REQUESTS'
                for name,digest in measured['result']['source_sha256'].items():
                    if name.startswith('spark_infer_q/'):assert manifest[name]==digest,name
            previous=json.loads((done/'323-e39-audit-v1.json').read_text())
            assert previous['worker_pid']==job['worker_pid']
            assert previous['result']['target_object_id']==id(session.weights)
            assert previous['result']['draft_object_id']==id(session.draft)
            assert os.environ.get('DEEPINFER_USE_NEXTN_DECODE') is None
            assert envs.DEEPINFER_USE_NEXTN_DECODE==0
            compiled=compile(Path(forward.__file__).read_bytes(),forward.__file__,'exec')
            expected={c.co_name:c for c in compiled.co_consts if isinstance(c,CodeType)}
            for name in ('_mm','_prefill_chain','layer_forward','stage','commit'):
                fn=getattr(forward,name)
                inner=inspect.unwrap(fn)
                assert inner.__globals__ is forward.__dict__ and inner.__module__==forward.__name__
                assert inner.__code__==expected[name],name
                if inner is not fn:
                    # commit's original @torch.no_grad() wrapper lives in
                    # torch.utils._contextlib; it is not a foreign patch.
                    closure=inspect.getclosurevars(fn).nonlocals
                    assert name=='commit' and fn.__globals__['__name__']=='torch.utils._contextlib'
                    assert closure['func'] is inner
                    assert type(closure['ctx_factory'].__self__).__name__=='no_grad'
            gc.collect(); session.torch.cuda.synchronize()
            assert not dist.is_initialized()
            allocated=session.torch.cuda.memory_allocated()
            # Other serial workspaces can retain their own caches between the
            # validation and request jobs. Compare our latest request job's
            # pre-allocation baseline, and keep the earlier baseline explicit.
            baseline=measured['result']['allocated_before'] if requests_path.exists() else result['allocated_before']
            assert allocated <= baseline,(allocated,baseline)
            result=dict(status='PASS_E40_AUDIT',allocated_bytes=allocated,baseline_bytes=baseline,
                validation_baseline_bytes=result['allocated_before'],
                same_resident_objects_as_e39=True,native_group_released=True,native_environment_restored=True,
                upstream_functions_unchanged=True,validated_library_unchanged=True,
                forward_code_matches_frozen=True,original_no_grad_wrapper_preserved=True,
                validation_job_sha256=hashlib.sha256((done/'334-e40-validate-v2.json').read_bytes()).hexdigest(),
                requests_job_sha256=hashlib.sha256(requests_path.read_bytes()).hexdigest() if requests_path.exists() else None,
                model_reloaded=False,source_sha256=manifest)
            (source/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            return result
        result = (requests(session,source,tag) if arguments.get('phase')=='requests' else
                  validate(session, source, tag))
        result.update(source_sha256=manifest, external_source_sha256=frozen['external_source_sha256'],
                      model_reloaded=False)
        return result
    finally:
        for name in list(sys.modules):
            if name == tag or name.startswith(tag+'.'): sys.modules.pop(name,None)
        gc.collect(); session.torch.cuda.synchronize()


def validate(session, source, tag):
    from e3_wide_target import WideTarget
    from tensorfold.families.qwen4_exp.cuda import forward
    from tensorfold.families.qwen4_exp.cuda.state import Buffers
    from spark_infer_q.prompt import prompt_state
    torch = session.torch
    module = importlib.import_module(tag+'.prefill')
    ops = importlib.import_module(tag+'.prefill_tensorfold')
    # Load E36's frozen input generator, without using its candidate adapter.
    spec = importlib.util.spec_from_file_location(tag+'.inputs',source/'scripts/e36_prefill_probe.py')
    input_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(input_module)
    cases = input_module.inputs(session)
    for name,n,text in (
        ('new_json',128,'Return a JSON object describing an LRU cache, including capacity, lookup, insertion and eviction. Use concrete examples. '),
        ('new_zh',384,'请解释数据库事务的原子性与隔离性，分析两个并发请求更新同一条记录时可能发生的情况，并给出具体例子。')):
        ids=session.tokenizer.encode(text*80,add_special_tokens=False)[:n]
        cases.append(dict(name=name,length=n,ids=ids,
                          token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
    before = torch.cuda.memory_allocated()
    originals = {name: getattr(forward,name) for name in ('_mm','_prefill_chain','layer_forward','stage','commit')}
    report = dict(status='RUNNING_E40', cases=[], extra=[], timings=[], requests=[],
                  allocated_before=before, complete_request_scope='native_token_API_not_HTTP',
                  default_promoted=False, long_context_validated=False)
    def save():
        p=source/'result.json'; tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    save()
    target = executor = None
    try:
        with torch.inference_mode():
            target = WideTarget(session.weights,capacity=1024)
            provider = ops.TensorFoldPrefillProvider(session.weights,torch,capacity=1024)
            executor = module.PrefillExecutor(provider,chunk_rows=512)
            for case in cases:
                report['active']=case['name'];save()
                left,features,logits = prompt_state(target,case['ids'])
                right=target.state();got=executor.advance(right,case['ids'])
                checked=dict(name=case['name'],length=case['length'],token_sha256=case['token_sha256'],
                             logits_exact=torch.equal(logits,got.logits),
                             features_exact=torch.equal(features,got.features),
                             state_differences=target.state_difference(left,right),followup=[])
                report['cases'].append(checked);save()
                assert checked['logits_exact'] and checked['features_exact'] and not checked['state_differences'],checked
                anchor=int(logits.argmax())
                for step in range(8):
                    a,af=target.step(left,[anchor]);b,bf=target.step(right,[anchor])
                    exact=(torch.equal(a,b) and torch.equal(af,bf) and not target.state_difference(left,right))
                    checked['followup'].append(dict(step=step,exact=exact));assert exact
                    anchor=int(a.argmax())
                del left,right,features,logits,got,a,af,b,bf
                gc.collect();save()
            # A second executor forces a multi-chunk prompt and nonzero-start
            # continuation. Its 129-row chunk leaves a short final tail.
            with module.PrefillExecutor(provider,chunk_rows=129) as chunked:
                ids=cases[3]['ids']
                a,features,logits=prompt_state(target,ids)
                b=target.state();x=chunked.advance(b,ids[:257]);y=chunked.advance(b,ids[257:])
                exact=(torch.equal(logits,y.logits) and torch.equal(features,torch.cat([x.features,y.features]))
                       and not target.state_difference(a,b))
                report['extra'].append(dict(case='multi_chunk_nonzero_start_tail',exact=exact,
                    chunks=[list(x.chunks),list(y.chunks)],positions=[x.end_position,y.end_position]));assert exact
                c=target.state();z=chunked.advance(c,ids,capture=False)
                exact=torch.equal(z.logits,logits) and z.features is None and not target.state_difference(a,c)
                report['extra'].append(dict(case='capture_disabled_independent_state',exact=exact));assert exact
                del a,b,c,x,y,z,features,logits
            # Same work and token preparation on both arms; preparation and
            # reusable workspace are outside timing, state allocation is in.
            for case in (cases[1],cases[3],cases[4]):
                ids=case['ids']
                def prompt(mode):
                    if mode=='rows8':
                        st,ft,lg=prompt_state(target,ids)
                    else:
                        st=target.state();value=executor.advance(st,ids);ft,lg=value.features,value.logits
                    token=int(lg.argmax());torch.cuda.synchronize()
                    return st,ft,lg,token
                for mode in ('rows8','owned'):
                    value=prompt(mode);del value
                for rep in range(4):
                    records={}
                    for mode in (('rows8','owned') if rep%2==0 else ('owned','rows8')):
                        torch.cuda.synchronize();start=time.perf_counter();value=prompt(mode)
                        elapsed=(time.perf_counter()-start)*1000
                        report['timings'].append(dict(name=case['name'],tokens=len(ids),mode=mode,
                                                      repetition=rep,prompt_ms=elapsed,token=value[3]))
                        records[mode]=value
                    a,b=records['rows8'],records['owned']
                    assert torch.equal(a[1],b[1]) and torch.equal(a[2],b[2]) and not target.state_difference(a[0],b[0])
                    del a,b,records,value;save()
            request_checks(session,source,tag,target,executor,report,save)
            assert all(getattr(forward,name) is value for name,value in originals.items())
            report['upstream_functions_unchanged']=True
            report['status']='PASS_E40_EXECUTOR';report.pop('active',None);save()
    except BaseException as exc:
        report.update(status='FAIL_E40_EXECUTOR',error_type=type(exc).__name__,error=str(exc)[:2000]);save()
        raise
    finally:
        if executor is not None: executor.close()
        del target,executor
        gc.collect();torch.cuda.synchronize()
    return report


def requests(session,source,tag):
    """Warm request A/B after validating the same executed library in v2."""
    from e3_wide_target import WideTarget
    prior=json.loads((session.root/'jobs/done/334-e40-validate-v2.json').read_text())
    manifest=json.loads((source/'manifest.json').read_text())
    assert prior['status']=='PASS' and prior['result']['status']=='PASS_E40_EXECUTOR'
    for name,digest in prior['result']['source_sha256'].items():
        if name.startswith('spark_infer_q/'): assert manifest[name]==digest,name
    module=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    torch=session.torch;torch.cuda.synchronize()
    report=dict(status='RUNNING_E40_REQUESTS',requests=[],extra=[],
        allocated_before=torch.cuda.memory_allocated(),validation_job_sha256=hashlib.sha256(
            (session.root/'jobs/done/334-e40-validate-v2.json').read_bytes()).hexdigest(),
        timing_scope='native_token_API_including_submit_drain_and_final_sync_not_HTTP',
        target_profile='eager_WideTarget',draft_profile='existing_K2_adaptive_provider',
        warmup_requests_per_case_per_arm=1,repetitions=4,default_promoted=False)
    def save():
        p=source/'result.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    save()
    with torch.inference_mode():
        target=WideTarget(session.weights,capacity=1024)
        with module.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=1024)) as executor:
            request_checks(session,source,tag,target,executor,report,save,warmed=True)
    report['status']='PASS_E40_REQUESTS';save()
    return report


def request_checks(session,source,tag,target,executor,report,save,*,warmed=False):
    runtime=importlib.import_module(tag+'.kaifra_runtime')
    spec=importlib.import_module(tag+'.kaifra_speculative')
    native_module=importlib.import_module(tag+'.native_scheduler')
    torch=session.torch
    prompts=[('en','Explain briefly how a cache differs from a queue.'),
             ('zh','用两句话解释缓存为什么能加速重复读取。')]
    roots=[]
    for name,text in prompts:
        ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        roots.append((name,ids))
    if warmed:
        text='Write a Python function that removes duplicates while preserving order. Explain the loop invariant and test empty inputs. '
        roots.append(('code512',session.tokenizer.encode(text*80,add_special_tokens=False)[:512]))
    for kind in ('target','dflash'):
        with native_module.NativeScheduler(session.root.parent/'DeepInfer',
                session.root.parent/'build-release/lib/deepinfer',
                max_drafts=0 if kind=='target' else 7) as native:
            for name,ids in roots:
                outputs={}
                reference_state=None
                # A warm-up per arm, then four AB/BA repetitions when warmed.
                for rep in (range(-1,4) if warmed else range(2)):
                    for mode in (('rows8','owned') if rep%2==0 else ('owned','rows8')):
                        prefill=executor if mode=='owned' else None
                        if kind=='target':
                            engine=runtime.KaifraEngine.from_native(native,target,eos_ids=session.eos,
                                capacity=768,max_requests=1,token_budget=512,prefill_executor=prefill)
                        else:
                            provider,curve,metadata=spec.session_provider(session,'adaptive')
                            engine=spec.speculative_engine(native,target,provider,eos_ids=session.eos,
                                prior=session.calibration['conditional_prior'],mode='adaptive',capacity=768,
                                max_requests=1,token_budget=512,prefill_executor=prefill)
                        with engine:
                            captured=[]
                            release=engine.bridge.backend.release
                            def record_release(handle,state):
                                captured.append(state.target if kind=='dflash' else state)
                                release(handle,state)
                            engine.bridge.backend.release=record_release
                            torch.cuda.synchronize();start=time.perf_counter()
                            handle=engine.submit_tokens('e40-'+name,ids,max_tokens=24)
                            events=[];trace=[]
                            while engine.active_requests:
                                events.extend(engine.step())
                                if kind=='dflash':
                                    trace.extend({k:m[k] for k in ('prompt','verified_rows','accepted','observed')}
                                                 for m in engine.last_step_metrics)
                            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                            tokens=[t for event in events for t in event.token_ids]
                            values=(tokens,[event.processed_position for event in events],trace)
                            if outputs: assert values==next(iter(outputs.values())),(kind,name,mode)
                            outputs[mode]=values
                            assert len(captured)==1
                            if reference_state is None: reference_state=captured[0]
                            assert not target.state_difference(reference_state,captured[0])
                            report['requests'].append(dict(kind=kind,name=name,mode=mode,repetition=rep,
                                prompt_tokens=len(ids),completion_tokens=len(tokens),request_ms=elapsed,
                                ttft_ms=events[0].ttft_seconds*1000,final_position=events[-1].processed_position,
                                tokens_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),trace=trace,
                                exact=True,state_exact=True,warmup=warmed and rep<0))
                            assert not engine.bridge.backend.states
                        save()
    # Target-only and speculative bridges require different NEXTN contexts.
    with native_module.NativeScheduler(session.root.parent/'DeepInfer',
            session.root.parent/'build-release/lib/deepinfer',max_drafts=0) as native:
        # Exercise cancellation after a committed prompt, followed by ID reuse.
        with runtime.KaifraEngine.from_native(native,target,eos_ids=session.eos,capacity=768,
                max_requests=1,token_budget=512,prefill_executor=executor) as engine:
            old=engine.submit_tokens('reuse',roots[0][1],max_tokens=24);engine.step();engine.cancel(old)
            cancelled=list(engine.drain());assert cancelled[-1].finish_reason=='cancelled'
            new=engine.submit_tokens('reuse',roots[0][1],max_tokens=1)
            assert new.generation>old.generation
            events=list(engine.drain());assert len(events)==1 and events[0].completion_tokens==1
            report['extra'].append(dict(case='native_cancel_reuse',exact=True))
