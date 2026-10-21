"""Held-out workloads, correctness outside timing, and balanced repetitions."""
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

TASKS=[
    {'id':'addition','prompt':'只输出 17 + 25 的计算结果，不要解释。','thinking':False,'limit':256},
    {'id':'capital','prompt':'What is the capital of France? Answer in one short sentence.','thinking':False,'limit':256},
    {'id':'python_add','prompt':'Write a Python function add(a, b) that returns their sum. Output only the function.','thinking':False,'limit':256},
    {'id':'translation','prompt':'将 Good morning 翻译成中文，只输出译文。','thinking':False,'limit':256},
    {'id':'json','prompt':'只输出合法 JSON 对象，字段 name 为 spark，字段 count 为整数 3。','thinking':False,'limit':256},
    {'id':'explanation','prompt':'用两句话解释为什么给模型权重量化可以减少内存占用。','thinking':False,'limit':256},
    {'id':'reasoning_math','prompt':'A shop has 7 boxes with 12 pencils in each. It sells 29 pencils. How many remain? Give the final number after reasoning.','thinking':True,'limit':1024},
    {'id':'reasoning_code','prompt':'Write a short Python function that returns the largest of three integers without using max(). Include one example call.','thinking':True,'limit':1024},
]


def selected_check(task,text):
    """Literal/property checks for these cases; not a general quality evaluator."""
    answer=text.split('</think>')[-1].replace('<|im_end|>','').replace('<|endoftext|>','').strip()
    tid=task['id']
    if tid=='addition':return answer=='42'
    if tid=='capital':return 'paris' in answer.lower()
    if tid=='translation':return any(s in answer for s in ('早上好','早安'))
    if tid=='reasoning_math':return '55' in answer
    if tid=='explanation':return '量化' in answer and any(s in answer for s in ('位','精度','比特'))
    if tid=='json':
        try:return json.loads(answer)=={'name':'spark','count':3}
        except ValueError:return False
    # Code samples are saved for review; matching a substring is not code execution.
    return None


def run(root,checkpoint,w,target,report,save,repetitions):
    import torch,transformers,tokenizers
    from transformers import AutoTokenizer
    from spark_infer_q.dflash2_reference import Draft,upstream_helpers
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.verification_budget import LaggedAcceptance,balanced_order
    from e3_mechanism_checks import entry,params
    from e3_draft_parity import compare
    import spark_q_e3_varlen
    manifest=json.loads((root/'checkpoint/fetch-manifest.json').read_text())
    weight_meta=manifest['files']['model.safetensors']
    assert weight_meta['sha256']=='d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f'
    h=hashlib.sha256()
    with (root/'checkpoint/model.safetensors').open('rb') as f:
        while chunk:=f.read(8<<20):h.update(chunk)
    assert h.hexdigest()==weight_meta['sha256']
    helpers=upstream_helpers(root/'reference/dflash.py','b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b')
    draft=Draft(root/'checkpoint',w.embed[0],w.head.b.weight,helpers,max_context=1536)
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    generation=json.loads((checkpoint/'generation_config.json').read_text())
    raw_eos=generation.get('eos_token_id',[])
    eos=set(raw_eos if isinstance(raw_eos,list) else [raw_eos]);eos.add(tokenizer.eos_token_id);eos.discard(None)
    report['environment']={'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,
                           'tf32':torch.backends.cuda.matmul.allow_tf32,'capacity':1536,'eos_ids':sorted(eos)}
    report['measurement_contract']={'repetitions':repetitions,'greedy':True,'timing':'synchronized host wall time, decode only',
       'excluded':['load','prompt','first output token','calibration','per-row oracle comparisons','final state checks'],
       'order':'cyclic Latin-square across serial/fixed/dynamic; each task gets every position in three rounds',
       'state_and_policy':'fresh state copy and independent runtime estimator for every run',
       'rate_prior':'separate calibration prompts, conditional estimates with strength 4, lagged online updates',
       'draft':'fixed four-row block; skip drafter only when zero proposals will be verified',
       'budget_cost':'verify+commit plus drafter only for rows>1; update total runtime without double-counting sampler',
       'limits':'single request; no sliding-window eviction beyond 2048; no stochastic sampling or full engine'}
    def emit(data):print(json.dumps(data,ensure_ascii=False),flush=True)
    def prompt_state(text,thinking):
        ids=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=thinking,return_dict=False)
        assert 0<len(ids)<256
        st=target.state();features=[]
        for token in ids:
            lg,feat=target.step(st,[token]);features.append(feat)
        return ids,st,torch.cat(features),int(lg[-1].argmax())
    # Use actual target captures and a longer-position synthetic diagnostic.
    _,parity_state,actual_features,first=prompt_state('Answer briefly: what is two plus two?',False)
    torch.manual_seed(109)
    cases=[(actual_features[:1],first,'actual-first-row'),(actual_features,first,'actual-prompt'),
           (torch.randn(255,12800,device='cuda',dtype=torch.bfloat16),first,'synthetic-position-255')]
    report['stage']='draft_parity';report['draft_parity']=[];save()
    report['draft_parity'].append(compare(draft,cases,root/'reference/dflash_official.py'));save()
    emit({'bf16_parity':report['draft_parity'][-1]['cases']})
    fp=Draft(root/'checkpoint',w.embed[0].float(),w.head.b.weight.float(),helpers,dtype=torch.float32,max_context=1536)
    report['draft_parity'].append(compare(fp,cases,root/'reference/dflash_official.py'));save()
    emit({'fp32_parity':report['draft_parity'][-1]['cases']})
    del fp,parity_state,actual_features,cases
    gc.collect();torch.cuda.empty_cache()
    report['parity_peak']={'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved()}
    torch.cuda.reset_peak_memory_stats()

    def generate(mode,start,context,first,limit,curves,prior,draft_cost,*,oracle=None,capture_oracle=False,detail=False):
        st=start.clone();ctx=context.clone() if mode!='serial' else None
        output=[first];steps=[];gold_logits=[];stopped=first in eos
        policy=LaggedAcceptance(prior)
        budget=spark_q_e3_varlen.BudgetProbe()
        budget.curve([(rows,ms+(draft_cost if rows>1 else 0.)) for rows,ms in curves])
        for _ in range(5):budget.runtime(1,1,curves[0][1],0.)
        torch.cuda.synchronize();t0=time.perf_counter()
        while len(output)<limit and not stopped:
            before=len(output)
            if mode=='serial':
                lg,_=target.step(st,[output[-1]],capture=False)
                token=int(lg[-1].argmax());output.append(token);stopped=token in eos
                if capture_oracle:gold_logits.append(lg.cpu())
                continue
            width=4 if mode=='fixed' else budget.schedule(params(),[entry(policy.current,policy.previous,process=len(steps)+2,schedule=len(steps)+2)])[0]
            k=min(width-1,limit-len(output)-1)
            t=time.perf_counter()
            proposals=draft.propose(ctx,output[-1])[0][:k] if k else []
            torch.cuda.synchronize();dm=(time.perf_counter()-t)*1000
            t=time.perf_counter();lg,features,b=target.run(st,[output[-1]]+proposals)
            vm=(time.perf_counter()-t)*1000
            pred=lg.argmax(-1).tolist()
            out,keep,processed_drafts,stopped=greedy_result(output[-1],proposals,pred,eos_ids=eos,max_new=limit-len(output))
            raw_accept=0
            for proposed,golden in zip(proposals,pred):
                if proposed!=golden:break
                raw_accept+=1
            observed_k=min(k,keep) if stopped else k
            accepted=min(raw_accept,observed_k)
            t=time.perf_counter();target.commit(st,b,k+1,keep);cm=(time.perf_counter()-t)*1000
            ctx=torch.cat([ctx,features[:keep]])
            output.extend(out)
            assert len(ctx)==st.pos==start.pos+len(output)-1
            if oracle is not None:
                assert out==oracle['tokens'][before:before+len(out)],('token divergence',mode,before,out)
                for j in range(keep):
                    assert torch.equal(lg[j:j+1].cpu(),oracle['logits'][before-1+j]),('logit divergence',mode,before-1+j)
            policy.observe(observed_k,accepted)
            budget.runtime(1,k+1,vm+cm+dm,0.)
            row={'width':width,'verified_rows':k+1,'observed_drafts':observed_k,'accepted_drafts':accepted,
                 'committed_drafts':processed_drafts,'draft_ms':dm,'verify_ms':vm,'commit_ms':cm}
            if detail:row.update(position=st.pos,proposals=proposals,emitted=out,rates=policy.current.copy())
            steps.append(row)
        torch.cuda.synchronize();elapsed=(time.perf_counter()-t0)*1000
        return {'tokens':output,'state':st,'logits':gold_logits,'steps':steps,'decode_ms':elapsed,
                'stopped_by_eos':stopped,'reached_limit':len(output)>=limit and not stopped}

    # Disjoint rate calibration; no measured-task outputs feed the initial prior.
    calibration=[('Return only the English word hello.',False),('只输出 8 加 9 的结果。',False)]
    success=[1.,1.,1.];trials=[2.,2.,2.];curves=None;draft_cost=0.
    from e3_prefill_control import probe as prefill_control
    report['prefill_control']=prefill_control(target);save();emit({'prefill_control':report['prefill_control']})
    report['stage']='calibration'
    report['calibration']={'prompts':[s for s,_ in calibration],'runs':[]};save()
    for text,thinking in calibration:
        ids,st,ctx,first=prompt_state(text,thinking)
        if curves is None:
            curves=[]
            proposals=draft.propose(ctx,first)[0]
            for rows in range(1,5):
                samples=[]
                for _ in range(5):
                    scratch=st.clone();torch.cuda.synchronize();t=time.perf_counter()
                    lg,ft,b=target.run(scratch,[first]+proposals[:rows-1]);target.commit(scratch,b,rows,rows)
                    samples.append((time.perf_counter()-t)*1000);del scratch
                curves.append((rows,statistics.median(samples)))
            costs=[]
            for _ in range(5):
                torch.cuda.synchronize();t=time.perf_counter();draft.propose(ctx,first);torch.cuda.synchronize()
                costs.append((time.perf_counter()-t)*1000)
            draft_cost=statistics.median(costs)
        run=generate('fixed',st,ctx,first,96,curves,(.5,.5,.5),draft_cost)
        for row in run['steps']:
            for j in range(row['observed_drafts']):
                if j>row['accepted_drafts']:break
                trials[j]+=1;success[j]+=int(j<row['accepted_drafts'])
        report['calibration']['runs'].append({'prompt':text,'tokens':run['tokens'],'eos':run['stopped_by_eos']})
        del st,ctx,run
    prior=[s/t for s,t in zip(success,trials)]
    report['calibration'].update(conditional_success=success,conditional_trials=trials,conditional_prior=prior,
                                verify_cost_ms=curves,draft_cost_ms=draft_cost,
                                effective_budget_cost_ms=[(rows,ms+(draft_cost if rows>1 else 0.)) for rows,ms in curves])
    report['stage']='balanced_decode';report['tasks']=[];save()
    for task_index,task in enumerate(TASKS):
        ids,start,ctx,first=prompt_state(task['prompt'],task['thinking'])
        golden=generate('serial',start,ctx,first,task['limit'],curves,prior,draft_cost,capture_oracle=True)
        record={**task,'prompt_tokens':len(ids),'token_ids':golden['tokens'],
                'text':tokenizer.decode(golden['tokens'],skip_special_tokens=False),
                'stopped_by_eos':golden['stopped_by_eos'],'reached_limit':golden['reached_limit'],
                'correctness':[],'timings':[]}
        record['has_final_answer']=bool(record['stopped_by_eos'] and (not task['thinking'] or '</think>' in record['text']))
        record['selected_property_check']=selected_check(task,record['text']) if record['has_final_answer'] else None
        report['tasks'].append(record);save()
        # These warm all modes and run exact numerical checks before timed repetitions.
        for mode in ('fixed','dynamic'):
            check=generate(mode,start,ctx,first,task['limit'],curves,prior,draft_cost,oracle=golden,detail=True)
            assert check['tokens']==golden['tokens'],(task['id'],mode,'complete output')
            differences=target.state_difference(check['state'],golden['state'])
            assert not differences,(task['id'],mode,differences)
            record['correctness'].append({'mode':mode,'tokens_equal':True,'logits_bit_exact':True,
                                          'live_state_bit_exact':True,'steps':check['steps']})
            del check
        for repetition in range(repetitions):
            for order,mode in enumerate(balanced_order(repetition,task_index)):
                result=generate(mode,start,ctx,first,task['limit'],curves,prior,draft_cost)
                assert result['tokens']==golden['tokens'],(task['id'],mode,'timed output')
                differences=target.state_difference(result['state'],golden['state'])
                assert not differences,(task['id'],mode,'timed live state',differences)
                row={'repetition':repetition,'order':order,'mode':mode,'decode_ms':result['decode_ms'],
                     'timed_output_tokens':len(result['tokens'])-1,'tokens_equal':True,'live_state_bit_exact':True,
                     'accepted_drafts':sum(s['accepted_drafts'] for s in result['steps']),
                     'observed_drafts':sum(s['observed_drafts'] for s in result['steps']),
                     'widths':[s['width'] for s in result['steps']],
                     'draft_ms':sum(s['draft_ms'] for s in result['steps']),
                     'verify_ms':sum(s['verify_ms'] for s in result['steps']),
                     'commit_ms':sum(s['commit_ms'] for s in result['steps'])}
                record['timings'].append(row);save();emit({'task':task['id'],**row})
                del result
        record['summary']={}
        baseline=statistics.median(x['decode_ms'] for x in record['timings'] if x['mode']=='serial')
        for mode in ('serial','fixed','dynamic'):
            values=[x['decode_ms'] for x in record['timings'] if x['mode']==mode]
            median=statistics.median(values)
            record['summary'][mode]={'median_ms':median,'min_ms':min(values),'max_ms':max(values),
                                     'ratio_of_medians':baseline/median,
                                     'median_decode_tokens_per_s':(len(golden['tokens'])-1)*1000/median}
        save();emit({'task_complete':task['id'],'eos':record['stopped_by_eos'],'summary':record['summary']})
        del start,ctx,golden
    report['summary']={'tasks':len(report['tasks']),'completed_by_eos':sum(t['stopped_by_eos'] for t in report['tasks']),
                       'all_outputs_and_live_states_equal':True,'all_correctness_logits_bit_exact':True}
    for mode in ('fixed','dynamic'):
        base=sum(t['summary']['serial']['median_ms'] for t in report['tasks'])
        candidate=sum(t['summary'][mode]['median_ms'] for t in report['tasks'])
        report['summary'][mode]={'sum_serial_medians_ms':base,'sum_candidate_medians_ms':candidate,'ratio':base/candidate,
                                'per_task_wins_vs_serial':sum(t['summary'][mode]['ratio_of_medians']>1 for t in report['tasks'])}
    report['summary']['dynamic_wins_vs_fixed']=sum(t['summary']['dynamic']['median_ms']<t['summary']['fixed']['median_ms'] for t in report['tasks'])
    report['quality_scope']='saved complete text and small literal/property checks; code not executed, no general quality benchmark'
    save()
