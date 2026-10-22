"""Full-prompt decode-arithmetic chunk validation; no change to E7 sources/defaults."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();root=a.root
    sys.path[:0]=[str(root/'followup-deps'),str(root/'python-deps'),str(root)]
    import torch
    from transformers import AutoTokenizer
    from e3_target import load_target
    from e3_wide_target import WideTarget
    from e7_tasks import TASKS
    manifest=json.loads((root/'ready.json').read_text())
    for name,sha in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    report={'status':'LOADING','harness_sha256':manifest,'correctness':[],'timings':[],
        'contract':{'widths':[1,2,4,8],'prefill_flag':False,'capacity':1024,'repetitions':4,
            'timing':'synchronized host wall including state init, feature capture and context concatenation; excludes tokenization, load, validation and continuation',
            'scope':'target prompt and eight target-only continuation steps; not SSE TTFT or speculative serving',
            'default_changed':False,'tasks':'E7 ten reused prompts plus prompt prefixes lengths 1..17',
            'gates':'all full-row logits, five captured layer features, live state at every chunk boundary and eight greedy continuation steps bit exact'}}
    path=root/'artifacts/prefill.json'
    def save():path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def emit(item):print(json.dumps(item),flush=True)
    save()
    try:
        weights,loaded=load_target(root,a.checkpoint,emit)
        report['target']=loaded;target=WideTarget(weights,capacity=1024)
        tok=AutoTokenizer.from_pretrained(str(a.checkpoint),local_files_only=True,trust_remote_code=False)
        prepared=[]
        for task in TASKS:
            ids=tok.apply_chat_template([{'role':'user','content':task['prompt']}],tokenize=True,
                add_generation_prompt=True,enable_thinking=task['thinking'],return_dict=False)
            assert len(ids)+task['limit']<=1017
            prepared.append({'id':task['id'],'ids':ids})
        cases=prepared+[{'id':f'tail-{n}','ids':prepared[0]['ids'][:n]} for n in range(1,18)]
        assert len(prepared[0]['ids'])>=17
        report['status']='VALIDATING';save()
        def diff(label,x,y):
            if torch.equal(x,y):return
            xx,yy=x.float(),y.float()
            raise AssertionError({'tensor':label,'shape':list(x.shape),'different_elements':int((x!=y).sum()),
                'max_abs':float((xx-yy).abs().max()),'relative_l2':float((xx-yy).norm()/xx.norm().clamp_min(1e-12))})
        with torch.inference_mode():
            for case in cases:
                for width in (2,4,8):
                    ref=target.state();got=target.state();rows_checked=0
                    for begin in range(0,len(case['ids']),width):
                        ids=case['ids'][begin:begin+width]
                        gl=[];gf=[]
                        for token in ids:
                            lg,ft=target.step(ref,[token]);gl.append(lg);gf.append(ft)
                        lg,ft=target.step(got,ids)
                        diff('logits',torch.cat(gl),lg);diff('features',torch.cat(gf),ft)
                        state_diff=target.state_difference(ref,got)
                        assert not state_diff,(case['id'],width,begin,'state',state_diff)
                        rows_checked+=len(ids)
                    pending=int(lg[-1].argmax())
                    for i in range(8):
                        lref,fref=target.step(ref,[pending]);lgot,fgot=target.step(got,[pending])
                        diff('continuation_logits',lref,lgot);diff('continuation_features',fref,fgot)
                        assert not target.state_difference(ref,got),(case['id'],width,'continuation',i)
                        pending=int(lref[-1].argmax())
                    report['correctness'].append({'id':case['id'],'width':width,'prompt_rows':rows_checked,
                        'logits_features_state_exact':True,'continuation_steps_exact':8,'pending_token':pending})
                    del ref,got,gl,gf,lg,ft,lref,lgot,fref,fgot
                    save()
                emit({'validated_prompt':case['id'],'rows':len(case['ids'])})
            assert len(report['correctness'])==len(cases)*3
            gc.collect();torch.cuda.empty_cache()
            report['status']='TIMING';save()
            def prompt(ids,width):
                torch.cuda.synchronize();start=time.perf_counter()
                state=target.state();features=[]
                for begin in range(0,len(ids),width):
                    lg,ft=target.step(state,ids[begin:begin+width]);features.append(ft)
                context=torch.cat(features);pending=int(lg[-1].argmax())
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                assert len(context)==state.pos==len(ids)
                del state,context,features,lg,ft
                return elapsed,pending
            for ti,case in enumerate(prepared):
                for width in (1,2,4,8):prompt(case['ids'],width)
                pending=None
                for rep in range(4):
                    order=(1,2,4,8);offset=(ti+rep)%4;order=order[offset:]+order[:offset]
                    for slot,width in enumerate(order):
                        ms,token=prompt(case['ids'],width)
                        if pending is None:pending=token
                        assert pending==token
                        report['timings'].append({'id':case['id'],'prompt_tokens':len(case['ids']),
                            'width':width,'repetition':rep,'order':slot,'prompt_ms':ms,'pending_token':token})
                        save()
                emit({'timed_prompt':case['id']})
        summaries={}
        for width in (1,2,4,8):
            per_task=[]
            for case in prepared:
                rows=[t for t in report['timings'] if t['id']==case['id'] and t['width']==width]
                assert len(rows)==4
                per_task.append({'id':case['id'],'prompt_tokens':len(case['ids']),
                    'median_ms':statistics.median(t['prompt_ms'] for t in rows),
                    'min_ms':min(t['prompt_ms'] for t in rows),'max_ms':max(t['prompt_ms'] for t in rows)})
            ms=sum(t['median_ms'] for t in per_task);tokens=sum(t['prompt_tokens'] for t in per_task)
            summaries[str(width)]={'sum_median_prompt_ms':ms,'prompt_tokens':tokens,
                'prompt_tokens_per_s':tokens*1000/ms,'tasks':per_task}
        for v in summaries.values():v['speedup_vs_width1']=summaries['1']['sum_median_prompt_ms']/v['sum_median_prompt_ms']
        report['summary']=summaries;report['status']='PASS_PROMPT_ONLY'
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        report['finished_unix']=time.time()
        if torch.cuda.is_initialized():report['resource']={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        for name in ('memory.current','memory.peak','memory.events'):
            f=Path('/sys/fs/cgroup')/name
            if f.exists():report.setdefault('resource',{})[name]=f.read_text().strip()
        save();emit({'status':report['status'],'correctness_cases':len(report['correctness']),'timings':len(report['timings'])})


if __name__=='__main__':main()
