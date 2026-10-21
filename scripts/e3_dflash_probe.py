#!/usr/bin/env python3
"""Run bounded real-weight greedy speculation, state checks and upstream budgets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

SOURCE_SHA='b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--outputs',type=int,default=16)
    a=p.parse_args()
    root=a.root
    sys.path.insert(0,str(root))
    sys.path.insert(0,str(root/'build/varlen'))
    sys.path.insert(0,str(root/'python-deps'))
    report={'status':'RUNNING','stage':'preflight','quality_evaluation':'NOT_MEASURED',
            'sampling_scope':'GREEDY_ONLY','production_engine_integration':False,
            'prefill_numerical_review':'REMAINS_OPEN; use decode arithmetic for prompt too',
            'draft_math_scope':'eager BF16 projections, explicit FP32 attention; no SGLang full-forward parity claim',
            'source_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
                ['e3_dflash_probe.py','e3_target.py','e3_mechanism_checks.py','spark_infer_q/dflash2_reference.py','spark_infer_q/speculative.py']}}
    def save(): (root/'artifacts/dflash-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    def event(data):
        print(json.dumps(data),flush=True)
    def timed(fn):
        import torch
        torch.cuda.synchronize()
        t=time.perf_counter(); out=fn(); torch.cuda.synchronize()
        return out,(time.perf_counter()-t)*1000
    try:
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        from spark_infer_q.dflash2_reference import Draft,upstream_helpers
        from spark_infer_q.speculative import greedy_result
        from e3_mechanism_checks import checks,entry,params
        import spark_q_e3_varlen
        helpers=upstream_helpers(root/'reference/dflash.py',SOURCE_SHA)
        report['mechanisms']=checks(helpers,spark_q_e3_varlen)
        event({'preflight':report['mechanisms']})
        save()
        if a.preflight_only:
            report['status']='PASS_PREFLIGHT';save();return 0
        report['stage']='target_loading';save()
        from e3_target import load_target,Target
        w,meta=load_target(root,a.checkpoint,event)
        report['target']=meta
        target=Target(w)
        # Synthetic tokens only here: force every rejection boundary independently.
        st=target.state()
        features=[]
        for token in [100,101,102,103]:
            logits,feat=target.step(st,[token]);features.append(feat)
        initial=st.clone()
        initial_feat=torch.cat(features)
        anchor=int(logits[-1].argmax())
        baseline=[anchor]; base_logits=[]
        for _ in range(5):
            lg,_=target.step(st,[baseline[-1]],capture=False)
            base_logits.append(lg.cpu())
            baseline.append(int(lg[-1].argmax()))
        report['forced_state_cases']=[]
        for case,accepted,k in [('no_draft',0,0),('reject_first',0,3),('reject_middle',1,3),('reject_last',2,3),('all_accept',3,3),('eos_inside',3,3)]:
            test=initial.clone();ref=initial.clone()
            drafts=baseline[1:1+k].copy()
            if accepted<k: drafts[accepted]=(drafts[accepted]+1)%w.cfg.vocab
            ids=[anchor]+drafts
            lg,feat,buf=target.run(test,ids)
            out,keep,n,stopped=greedy_result(anchor,drafts,lg.argmax(-1).tolist(),
                                            eos_ids={baseline[2]} if case=='eos_inside' else ())
            assert out==baseline[1:1+len(out)],(case,out,baseline)
            expected_keep=2 if case=='eos_inside' else accepted+1
            assert keep==expected_keep,(case,keep)
            target.commit(test,buf,len(ids),keep)
            for j,token in enumerate(ids[:keep]):
                rl,rf=target.step(ref,[token])
                assert torch.equal(rl.cpu(),base_logits[j]),(case,'reference replay')
                assert torch.equal(lg[j:j+1].cpu(),base_logits[j]),(case,'verification logits')
                assert torch.equal(feat[j:j+1],rf),(case,'captured features')
            differences=target.state_difference(test,ref)
            assert not differences,(case,differences)
            follow=int(base_logits[keep-1][-1].argmax())
            a1,_=target.step(test,[follow],capture=False)
            a2,_=target.step(ref,[follow],capture=False)
            assert torch.equal(a1,a2),(case,'next logits')
            result={'case':case,'verified_rows':len(ids),'committed_rows':keep,'accepted_drafts':n,
                    'live_state_bit_exact':True,'verification_logits_bit_exact':True,
                    'captured_features_bit_exact':True,'following_logits_bit_exact':True,
                    'controlled_stop_id':case=='eos_inside'}
            report['forced_state_cases'].append(result);event(result);save()
            del test,ref
        del st,initial,initial_feat
        report['stage']='waiting_draft';save()
        deadline=time.monotonic()+1800
        while not (root/'checkpoint/fetch-manifest.json').exists():
            if time.monotonic()>deadline: raise TimeoutError('draft transfer not complete')
            time.sleep(2)
        manifest=json.loads((root/'checkpoint/fetch-manifest.json').read_text())
        for name,m in manifest['files'].items():
            assert hashlib.sha256((root/'checkpoint'/name).read_bytes()).hexdigest()==m['sha256']
        report['draft_download']=manifest
        draft=Draft(root/'checkpoint',w.embed[0],w.head.b.weight,helpers)
        report['draft_parameters']=draft.parameters
        event({'draft_loaded':draft.parameters,'cuda_allocated_bytes':torch.cuda.memory_allocated()})
        report['stage']='real_draft';report['requests']=[];save()
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(a.checkpoint/'tokenizer.json'))
        prompts=['The capital of France is','计算 17 + 25，并简短解释。','Write a Python function that adds two integers.']
        for prompt in prompts:
            text='<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n'
            ids=tokenizer.encode(text,add_special_tokens=False).ids
            assert 0<len(ids)<64
            state=target.state();fs=[]
            for token in ids:
                lg,feat=target.step(state,[token]);fs.append(feat)
            context=torch.cat(fs)
            first=int(lg[-1].argmax())
            baseline_state=state.clone()
            expected=[first]; expected_logits=[]
            t0=time.perf_counter()
            for _ in range(a.outputs-1):
                lg,_=target.step(baseline_state,[expected[-1]],capture=False)
                expected_logits.append(lg.cpu())
                expected.append(int(lg[-1].argmax()))
            baseline_ms=(time.perf_counter()-t0)*1000
            request={'prompt':prompt,'prompt_token_ids':ids,'baseline_token_ids':expected,
                     'baseline_text':tokenizer.decode(expected),'baseline_decode_ms':baseline_ms,'modes':[]}
            report['requests'].append(request);save()
            # Startup curves use measured full verification+commit cost, no cache graphs claimed.
            curves=[]
            for rows in range(1,5):
                samples=[]
                for repeat in range(2):
                    scratch=state.clone()
                    def cost():
                        lg,feat,b=target.run(scratch,expected[:rows])
                        target.commit(scratch,b,rows,rows)
                    _,ms=timed(cost);samples.append(ms)
                    del scratch
                curves.append((rows,sum(samples)/len(samples)))
            request['measured_verify_cost_ms']=curves
            # Draft is always a four-row block. Dynamic budget crops verification only.
            _,draft_ms=timed(lambda:draft.propose(context,first))
            request['draft_calibration_ms']=draft_ms
            budget=spark_q_e3_varlen.BudgetProbe()
            budget.curve(curves)
            for _ in range(5): budget.runtime(1,1,curves[0][1],draft_ms)
            for mode in ['fixed4','dynamic']:
                test=state.clone();ctx=context.clone();outputs=[first];steps=[]
                rates=[1.,1.,1.];prev=[1.,1.,1.];success=[0,0,0];trials=[0,0,0]
                t0=time.perf_counter()
                while len(outputs)<a.outputs:
                    width=4 if mode=='fixed4' else budget.schedule(params(),[entry(rates,prev,process=len(steps)+2,schedule=len(steps)+2)])[0]
                    # Respect output limit; always run drafter before crop, matching fixed-block DFlash2.
                    (proposals,detail),dm=timed(lambda:draft.propose(ctx,outputs[-1]))
                    k=min(width-1,a.outputs-len(outputs)-1)
                    proposed=proposals[:k]
                    start_pos=test.pos
                    (lg,feat,buf),vm=timed(lambda:target.run(test,[outputs[-1]]+proposed))
                    pred=lg.argmax(-1).tolist()
                    out,keep,n,stopped=greedy_result(outputs[-1],proposed,pred,max_new=a.outputs-len(outputs))
                    index=len(outputs)-1
                    assert out==expected[len(outputs):len(outputs)+len(out)],(prompt,mode,'output',outputs,out,expected)
                    for j in range(keep):
                        assert torch.equal(lg[j:j+1].cpu(),expected_logits[index+j]),(prompt,mode,'logits',index+j)
                    _,cm=timed(lambda:target.commit(test,buf,k+1,keep))
                    ctx=torch.cat([ctx,feat[:keep]])
                    outputs.extend(out)
                    assert len(ctx)==test.pos==start_pos+keep
                    for j in range(k):
                        if j==0 or n>=j:
                            trials[j]+=1;success[j]+=int(n>j)
                    prev=rates.copy()
                    conditional=[(s+1)/(t+2) for s,t in zip(success,trials)]
                    product=1.;rates=[]
                    for c in conditional: product*=c;rates.append(product)
                    budget.runtime(1,k+1,vm+cm,dm)
                    row={'width':width,'actual_verify_rows':k+1,'proposals':proposed,'accepted':n,
                         'emitted':out,'committed_position':test.pos,'draft_ms':dm,'verify_ms':vm,'commit_ms':cm,
                         'rates_for_next_round':rates.copy(),'selector_path':detail['path_indices']}
                    steps.append(row);event({'prompt':prompt,'mode':mode,**row})
                elapsed=(time.perf_counter()-t0)*1000
                differences=target.state_difference(test,baseline_state)
                assert not differences,(prompt,mode,'final state',differences)
                result={'mode':mode,'output_ids_equal':outputs==expected,'verification_logits_bit_exact':True,
                        'final_live_state_bit_exact':True,'steps':steps,'decode_ms':elapsed,
                        'speedup_vs_target_only':baseline_ms/elapsed,'accepted_drafts':sum(s['accepted'] for s in steps),
                        'verified_drafts':sum(len(s['proposals']) for s in steps),
                        'scheduled_widths':[s['width'] for s in steps]}
                request['modes'].append(result);save()
                del test,ctx
            save();del state,baseline_state,context
        report['status']='PASS_BOUNDED_GREEDY_INTEGRATION'
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        try:
            import torch
            report['resource']={'torch_allocated_bytes':torch.cuda.memory_allocated(),
                                'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            for name in ['memory.current','memory.peak','memory.events']:
                p=Path('/sys/fs/cgroup')/name
                if p.exists(): report['resource'][name]=p.read_text().strip()
        except Exception as exc: report['resource_error']=str(exc)
        save();event({'status':report['status'],'stage':report['stage']})
    return int(report['status']=='FAIL')


if __name__=='__main__': raise SystemExit(main())
