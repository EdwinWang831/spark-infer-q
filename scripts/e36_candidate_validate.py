"""Whole-target prefill candidate: exact gate before any prompt-only timing."""
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


def candidate(session,source,cases):
    torch,w=session.torch,session.weights
    from tensorfold.families.qwen4_exp.cuda import forward as upstream
    from tensorfold.families.qwen4_exp.cuda.state import Buffers,State
    from e3_wide_target import WideTarget
    tag='_e36_candidate_'+hashlib.sha256(str(source).encode()).hexdigest()[:10]
    adapter=load(source/'scripts/e36_prefill_candidate.py',tag+'_adapter')
    helpers=load(source/'scripts/e36_diagnose.py',tag+'_helpers')
    inputs=[dict(c) for c in cases]
    for name,n,text in (
        ('new_json',128,'Return a JSON object describing an LRU cache, including capacity, lookup, insertion and eviction. Use concrete examples. '),
        ('new_zh',384,'请解释数据库事务的原子性与隔离性，分析两个并发请求更新同一条记录时可能发生的情况，并给出具体例子。')):
        ids=session.tokenizer.encode(text*80,add_special_tokens=False)[:n];assert len(ids)==n
        inputs.append(dict(name=name,length=n,ids=ids,
            token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
            text_sha256=hashlib.sha256(text.encode()).hexdigest()))
    report=dict(status='RUNNING_E36_CANDIDATE',candidate_inputs=inputs,cases=[],timings=[],
        expected_case_count=7,full_model_fp32_reference=False,task_quality_evaluated=False,
        default_promoted=False,performance_evaluated=False,timing_scope='prompt_only_token_API',
        layer0_driven_candidate=True,allocated_before=torch.cuda.memory_allocated())
    path=source/'candidate-result.json'
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    def difference(a,b):
        return WideTarget.state_difference(type('Owner',(),dict(torch=torch))(),a,b)
    def piece(f,st,b,ids):
        n=len(ids);segs=f.stage(w,b,[(st,ids)])
        f._embed(w,b.ids[:n],w.cfg.streams,b.h[:n]);pending=None;features=[]
        for layer in w.layers:
            pending=f.layer_forward(layer,w,segs,b,n,pending)
            if layer.index in (5,15,25,35,45):
                shadow=b.h[:n].clone();proxy=copy.copy(b);proxy.pss=torch.empty_like(b.pss)
                f._writeback(shadow,proxy,n,w.cfg,pending)
                features.append(shadow.unflatten(-1,(4,2560)).mean(-2))
        logits=f.finish(w,w.mixer,b,n,pending).clone()[-1:]
        f.commit(w,st,b,n,n)
        return logits,torch.cat(features,-1)
    def prompt(f,b,ids):
        state=State(w,1024,8,'bf16',limit=1024);features=[]
        width=len(ids) if b.prefill else 8
        for at in range(0,len(ids),width):
            logits,ft=piece(f,state,b,ids[at:at+width]);features.append(ft)
        output=torch.cat(features,0);token=int(logits.argmax())
        torch.cuda.synchronize()
        return logits,output,state,token
    try:
        with adapter.using_forward(session,tag+'_forward') as (forward,counters):
            serial_buffer=Buffers(w,8,1024,prefill=False)
            candidate_buffer=Buffers(w,512,1024,prefill=True)
            candidate_buffer.moe.y=torch.empty_like(candidate_buffer.moe.y,dtype=torch.float32)
            for c in inputs:
                report['active']=c['name']+':correctness';save()
                left=prompt(upstream,serial_buffer,c['ids'])
                right=prompt(forward,candidate_buffer,c['ids'])
                state_diff=difference(left[2],right[2])
                checked=dict(name=c['name'],length=c['length'],token_sha256=c['token_sha256'],
                    logits_exact=torch.equal(left[0],right[0]),features_exact=torch.equal(left[1],right[1]),
                    state_exact=not state_diff,state_differences=state_diff,
                    logits=helpers.metrics(right[0],left[0]),features=helpers.metrics(right[1],left[1]),
                    tokens=dict(rows8=left[3],candidate=right[3]),followup=[])
                report['cases'].append(checked)
                if not all(checked[k] for k in ('logits_exact','features_exact','state_exact')):
                    report.update(status='REJECT_E36_CANDIDATE_NUMERICS',stop_case=c['name'],
                        reason='whole_target_prefill_gate_failed',counterfactual_counters=dict(counters))
                    report.pop('active',None);save();return report
                anchor=left[3]
                for step in range(8):
                    a=piece(upstream,left[2],serial_buffer,[anchor])
                    b=piece(upstream,right[2],serial_buffer,[anchor])
                    diff=difference(left[2],right[2]);exact=(torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and not diff)
                    checked['followup'].append(dict(step=step,exact=exact,state_differences=diff))
                    if not exact:
                        report.update(status='REJECT_E36_CANDIDATE_NUMERICS',stop_case=c['name'],
                            reason='prefill_decode_handoff_gate_failed',counterfactual_counters=dict(counters))
                        report.pop('active',None);save();return report
                    anchor=int(a[0].argmax())
                del left,right,a,b
                gc.collect();save()
            # No time is measured until every whole-target gate above passes.
            for c in inputs:
                for mode,f,buf in (('rows8',upstream,serial_buffer),('candidate',forward,candidate_buffer)):
                    warm=prompt(f,buf,c['ids']);del warm
                for repetition in range(4):
                    order=('rows8','candidate') if repetition%2==0 else ('candidate','rows8')
                    fingerprints={}
                    for mode in order:
                        f,buf=(upstream,serial_buffer) if mode=='rows8' else (forward,candidate_buffer)
                        report['active']=dict(name=c['name'],mode=mode,repetition=repetition);save()
                        torch.cuda.synchronize();start=time.perf_counter()
                        value=prompt(f,buf,c['ids']);elapsed=(time.perf_counter()-start)*1000
                        fingerprints[mode]=(helpers.tensor_hash(value[0],value[1]),value[3])
                        report['timings'].append(dict(name=c['name'],mode=mode,repetition=repetition,
                            prompt_tokens=c['length'],prompt_ms=elapsed,output_sha256=fingerprints[mode][0],
                            token=value[3]))
                        del value
                    assert fingerprints['rows8']==fingerprints['candidate'],'timed output changed'
                    save()
            report.update(status='PASS_E36_CANDIDATE_PROMPT',performance_evaluated=True,
                counterfactual_counters=dict(counters))
            report.pop('active',None);save();return report
    finally:
        for name in (tag+'_adapter',tag+'_helpers'):sys.modules.pop(name,None)
        gc.collect();torch.cuda.synchronize()
