"""Eight-row state contracts, official draft controls, then five-mode timings."""
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

from e3_policy_tasks import CALIBRATION,TEST
from spark_infer_q.wide_budget import MODES,conditional_prior,order


def run(root,checkpoint,w,target,report,save,repetitions):
    import torch
    from transformers import AutoTokenizer
    from spark_infer_q.dflash2_reference import Draft,upstream_helpers
    from spark_infer_q.dflash2_block import BlockDraft
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.verification_budget import LaggedAcceptance
    from e3_mechanism_checks import entry,params
    from e3_wide_parity import compare
    import spark_q_e3_varlen
    budget_sha=hashlib.sha256(Path(spark_q_e3_varlen.__file__).read_bytes()).hexdigest()
    assert budget_sha=='722b6ea835fa3e7e349af7f22ac727045a65af487a35f6709f3838e066bd678e'
    h=hashlib.sha256()
    with (root/'checkpoint/model.safetensors').open('rb') as f:
        while chunk:=f.read(8<<20):h.update(chunk)
    assert h.hexdigest()=='d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f'
    helpers=upstream_helpers(root/'reference/dflash.py','b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b')
    draft8=BlockDraft(root/'checkpoint',w.embed[0],w.head.b.weight,helpers,block_size=8,max_context=1536)
    draft4=object.__new__(BlockDraft);draft4.__dict__=draft8.__dict__.copy();draft4.block_size=4
    legacy=object.__new__(Draft);legacy.__dict__=draft4.__dict__.copy()
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    raw=json.loads((checkpoint/'generation_config.json').read_text()).get('eos_token_id',[])
    eos=set(raw if isinstance(raw,list) else [raw]);eos.add(tokenizer.eos_token_id);eos.discard(None)
    report['reference']={'draft_weights_sha256':h.hexdigest(),'draft_revision':'a22a1744e85c17364dcc54a6a420af7135d4c7f6',
        'budget_binary_sha256':budget_sha,'deepinfer_commit':'774ce244b87cc0eb8d4b94ab9f5de2ab22049f01'}
    report['measurement_contract']={'modes':MODES,'repetitions':repetitions,'order':'five-mode cyclic Latin square',
        'scope':'single-request bounded greedy, reused eight regression tasks, no new quality/generalization claim',
        'fixed3':'four-row proposer, three verified drafts',
        'fixed3_wide':'eight-row proposer, only its first three proposals verified',
        'fixed7':'eight-row proposer, all seven proposals verified',
        'dynamic7':'eight-row proposer, original DeepInfer budget selects zero through seven drafts; skip drafter at zero',
        'disabled':'no EOS verification crop, cached draft constants, learned local controller or stochastic sampling',
        'calibration':'six separate prompts, Beta(1,1) depth counts with rejection censoring, online prior strength four and lagged updates',
        'costs':'all target row widths 1..8 measured; full eight-row drafter cost added only for rows>1; runtime total counted once',
        'timing':'synchronized decode host wall-clock including drafter/verification/commit/control',
        'excluded':['load','prompt','first pending output','calibration','state initialization','oracle comparisons','final state checks'],
        'numerical_limits':'same BF16 arithmetic/official draft and target prefill discrepancies remain separate'}
    def emit(value):print(json.dumps(value,ensure_ascii=False),flush=True)
    def prompt_state(task):
        ids=tokenizer.apply_chat_template([{'role':'user','content':task['prompt']}],tokenize=True,
            add_generation_prompt=True,enable_thinking=task['thinking'],return_dict=False)
        assert 0<len(ids)<256
        st=target.state();features=[]
        for token in ids:
            lg,ft=target.step(st,[token]);features.append(ft)
        return ids,st,torch.cat(features),int(lg[-1].argmax())

    report['stage']='controlled_budget';save()
    probe=spark_q_e3_varlen.BudgetProbe();probe.curve([(rows,1.+.1*(rows-1)) for rows in range(1,9)])
    controlled={}
    for name,rates,expected in [('full',[1.]*7,8),('zero',[0.]*7,1),('partial',[.9,.1,0.,0.,0.,0.,0.],2)]:
        got=probe.schedule(params(),[entry(rates)])[0];assert got==expected,(name,got);controlled[name]=got
    report['controlled_budget']=controlled;save()
    diagnostic={'prompt':'Answer briefly: what is two plus two?','thinking':False}
    ids,start,ctx,first=prompt_state(diagnostic)
    report['stage']='eight_row_state_contract';report['state_contract']=[];save()
    tokens=[100,101,102,103,104,105,106,107]
    for rows in range(1,9):
        oracle=start.clone();gold_logits=[];gold_features=[]
        for token in tokens[:rows]:
            lg,ft=target.step(oracle,[token]);gold_logits.append(lg.cpu());gold_features.append(ft)
        expected=start.clone();gold=torch.cat(gold_logits);features=torch.cat(gold_features)
        for keep in range(rows+1):
            scratch=start.clone();lg,ft,b=target.run(scratch,tokens[:rows])
            assert torch.equal(lg.cpu(),gold),(rows,keep,'window logits')
            assert torch.equal(ft,features),(rows,keep,'window features')
            if keep:
                target.commit(scratch,b,rows,keep)
            else:
                # The upstream commit contract is 1..R. Keeping no input means
                # discarding this speculative state, never commit(..., keep=0).
                del scratch
                scratch=start.clone()
            assert not target.state_difference(scratch,expected),(rows,keep,'prefix state')
            # Rejected KV/scratch must not affect a subsequent real input.
            continuation=scratch.clone();reference=expected.clone()
            a,fa=target.step(continuation,[199]);b,fb=target.step(reference,[199])
            assert torch.equal(a,b) and torch.equal(fa,fb),(rows,keep,'continuation logits/features')
            assert not target.state_difference(continuation,reference),(rows,keep,'continuation state')
            report['state_contract'].append({'rows':rows,'keep':keep,'zero_keep_discard':keep==0,'logits_features_exact':True,
                'live_state_exact':True,'continuation_exact':True})
            del scratch,continuation,reference
            if keep<rows:target.step(expected,[tokens[keep]])
        del oracle,expected
        save();emit({'state_rows_pass':rows})
    assert len(report['state_contract'])==44
    report['stage']='draft_parity';save()
    assert torch.equal(legacy.forward_hidden(ctx,first)[0],draft4.forward_hidden(ctx,first)[0])
    assert legacy.propose(ctx,first)[0]==draft4.propose(ctx,first)[0]
    report['legacy_four_row_exact']=True
    torch.manual_seed(719)
    cases=[(ctx[:1],first,'actual-first-row'),(ctx,first,'actual-prompt'),
        (torch.randn(255,12800,device='cuda',dtype=torch.bfloat16),first,'synthetic-position-255')]
    report['draft_parity']=[compare(draft8,cases,root/'reference/dflash_official.py')];save()
    fp=BlockDraft(root/'checkpoint',w.embed[0].float(),w.head.b.weight.float(),helpers,
        dtype=torch.float32,block_size=8,max_context=1536)
    report['draft_parity'].append(compare(fp,cases,root/'reference/dflash_official.py'));save()
    report['diagnostic_peak']={'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved()}
    del fp,cases,start,ctx,features,gold,lg,ft,ids,a,b,fa,fb,gold_features,gold_logits
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    emit({'official_draft_controls':report['draft_parity']})

    def generate(mode,task,start,context,first,curves,prior,draft_cost,*,oracle=None,capture_gold=False):
        st=start.clone();ctx=context.clone() if mode!='serial' else None
        output=[first];logits_gold=[];steps=[];stopped=first in eos
        policy=LaggedAcceptance(prior);budget=spark_q_e3_varlen.BudgetProbe()
        budget.curve([(rows,ms+(draft_cost if rows>1 else 0.)) for rows,ms in curves])
        for _ in range(5):budget.runtime(1,1,curves[0][1],0.)
        torch.cuda.synchronize();t0=time.perf_counter()
        while len(output)<task['limit'] and not stopped:
            before=len(output)
            if mode=='serial':
                lg,_=target.step(st,[output[-1]],capture=False)
                token=int(lg[-1].argmax());output.append(token);stopped=token in eos
                if capture_gold:logits_gold.append(lg.cpu())
                continue
            if mode=='fixed_calibrated':width=calibrated_width
            elif mode in ('fixed3','fixed3_wide'):width=4
            elif mode=='fixed7':width=8
            else:width=budget.schedule(params(),[entry(policy.current,policy.previous,process=len(steps)+2,schedule=len(steps)+2)])[0]
            k=min(width-1,task['limit']-len(output)-1)
            engine=draft4 if mode=='fixed3' else draft8
            t=time.perf_counter();proposals=engine.propose(ctx,output[-1])[0][:k] if k else []
            torch.cuda.synchronize();dm=(time.perf_counter()-t)*1000
            t=time.perf_counter();lg,features,b=target.run(st,[output[-1]]+proposals);vm=(time.perf_counter()-t)*1000
            pred=lg.argmax(-1).tolist()
            out,keep,_,stopped=greedy_result(output[-1],proposals,pred,eos_ids=eos,max_new=task['limit']-len(output))
            raw_accept=0
            for proposed,next_token in zip(proposals,pred):
                if proposed!=next_token:break
                raw_accept+=1
            observed=min(k,keep) if stopped else k;accepted=min(raw_accept,observed)
            t=time.perf_counter();target.commit(st,b,k+1,keep);cm=(time.perf_counter()-t)*1000
            ctx=torch.cat([ctx,features[:keep]]);output.extend(out)
            assert len(ctx)==st.pos==start.pos+len(output)-1
            if oracle is not None:
                assert out==oracle['tokens'][before:before+len(out)],(task['id'],mode,before,'tokens')
                for j in range(keep):assert torch.equal(lg[j:j+1].cpu(),oracle['logits'][before-1+j]),(task['id'],mode,before,j,'logits')
            policy.observe(observed,accepted);budget.runtime(1,k+1,vm+cm+dm,0.)
            steps.append({'width':width,'verified_rows':k+1,'observed_drafts':observed,'accepted_drafts':accepted,
                'emitted_tokens':len(out),'draft_ms':dm,'verify_ms':vm,'commit_ms':cm,
                'lagged_rates':policy.previous.copy()})
        torch.cuda.synchronize()
        return {'tokens':output,'state':st,'logits':logits_gold,'steps':steps,
            'decode_ms':(time.perf_counter()-t0)*1000,'stopped_by_eos':stopped}

    samples={rows:[] for rows in range(1,9)};draft_samples={4:[],8:[]};all_cal_steps=[]
    report['stage']='calibration';report['calibration']={'prompts':[],'cost_samples':[]};save()
    for ci,task in enumerate(CALIBRATION):
        ids,st,ctx,first=prompt_state(task)
        proposals=draft8.propose(ctx,first)[0]
        for rows in range(1,9):
            scratch=st.clone();_,_,b=target.run(scratch,[first]+proposals[:rows-1]);target.commit(scratch,b,rows,rows);del scratch
        for rep in range(3):
            for block,engine in ((4,draft4),(8,draft8)):
                torch.cuda.synchronize();t=time.perf_counter();engine.propose(ctx,first);torch.cuda.synchronize()
                draft_samples[block].append((time.perf_counter()-t)*1000)
            for wi in range(8):
                rows=(wi+rep+ci)%8+1;scratch=st.clone();torch.cuda.synchronize();t=time.perf_counter()
                _,_,b=target.run(scratch,[first]+proposals[:rows-1]);target.commit(scratch,b,rows,rows)
                ms=(time.perf_counter()-t)*1000;samples[rows].append(ms)
                report['calibration']['cost_samples'].append({'prompt_index':ci,'prefix_rows':len(ids),'rows':rows,'verify_commit_ms':ms})
                del scratch
        curves=[(rows,statistics.median(samples[rows])) for rows in range(1,9)]
        run=generate('fixed7',task,st,ctx,first,curves,(.5,)*7,statistics.median(draft_samples[8]))
        all_cal_steps.extend(run['steps'])
        report['calibration']['prompts'].append({**task,'token_ids':run['tokens'],'stopped_by_eos':run['stopped_by_eos']})
        save();emit({'calibrated':task['id'],'tokens':len(run['tokens']),'eos':run['stopped_by_eos']})
        del st,ctx,run
    prior,success,trials=conditional_prior(all_cal_steps)
    draft_cost=statistics.median(draft_samples[8])
    report['calibration'].update(conditional_prior=prior,conditional_success=success,conditional_trials=trials,
        verify_commit_curve_ms=curves,draft_samples_ms=draft_samples,
        draft_median_ms={block:statistics.median(values) for block,values in draft_samples.items()},
        effective_native_curve_ms=[(rows,ms+(draft_cost if rows>1 else 0.)) for rows,ms in curves])
    cumulative=[];product=1.
    for rate in prior:product*=rate;cumulative.append(product)
    static_probe=spark_q_e3_varlen.BudgetProbe()
    static_probe.curve(report['calibration']['effective_native_curve_ms'])
    for _ in range(5):static_probe.runtime(1,1,curves[0][1],0.)
    calibrated_width=static_probe.schedule(params(),[entry(cumulative)])[0]
    report['calibration']['frozen_static_width']=calibrated_width
    save();emit({'calibration_prior':prior,'verify_curve':curves,'draft_ms':report['calibration']['draft_median_ms']})
    previous=json.loads((root/'artifacts/local-policy.json').read_text())
    previous_tokens={t['id']:t['token_ids'] for t in previous['tasks']}
    report['stage']='balanced_generation';report['tasks']=[];save()
    for ti,task in enumerate(TEST):
        ids,start,ctx,first=prompt_state(task)
        golden=generate('serial',task,start,ctx,first,curves,prior,draft_cost,capture_gold=True)
        assert golden['stopped_by_eos'] and golden['tokens']==previous_tokens[task['id']],(task['id'],'serial/EOS regression')
        record={**task,'prompt_tokens':len(ids),'token_ids':golden['tokens'],
            'text':tokenizer.decode(golden['tokens'],skip_special_tokens=False),'stopped_by_eos':True,'correctness':[],'timings':[]}
        report['tasks'].append(record);save()
        for mode in MODES[1:]:
            check=generate(mode,task,start,ctx,first,curves,prior,draft_cost,oracle=golden)
            assert check['tokens']==golden['tokens'] and not target.state_difference(check['state'],golden['state'])
            record['correctness'].append({'mode':mode,'tokens_equal':True,'kept_logits_bit_exact':True,
                'live_state_bit_exact':True,'steps':check['steps']});del check
        for rep in range(repetitions):
            for slot,mode in enumerate(order(rep,ti)):
                result=generate(mode,task,start,ctx,first,curves,prior,draft_cost)
                assert result['tokens']==golden['tokens']
                assert not target.state_difference(result['state'],golden['state']),(task['id'],mode,'timed state')
                row={'mode':mode,'repetition':rep,'order':slot,'decode_ms':result['decode_ms'],
                    'tokens_equal':True,'live_state_bit_exact':True,'steps':result['steps']}
                record['timings'].append(row);save();emit({'task':task['id'],'mode':mode,'repetition':rep,
                    'decode_ms':row['decode_ms'],'widths':sorted({s['width'] for s in row['steps']})});del result
        base=statistics.median(r['decode_ms'] for r in record['timings'] if r['mode']=='serial')
        record['summary']={}
        for mode in MODES:
            values=[r['decode_ms'] for r in record['timings'] if r['mode']==mode];ms=statistics.median(values)
            record['summary'][mode]={'median_ms':ms,'min_ms':min(values),'max_ms':max(values),'ratio':base/ms,
                'decode_tokens_per_s':(len(golden['tokens'])-1)*1000/ms}
        save();emit({'task_complete':task['id'],'summary':record['summary']});del start,ctx,golden
        gc.collect()
    # Matched secondary control: distinguish online adaptation from selecting
    # one globally better fixed width using only the independent calibration.
    report['stage']='calibrated_static_control';report['static_control']=[];save()
    control_modes=('fixed_calibrated','fixed7','dynamic7')
    for ti,task in enumerate(TEST):
        ids,start,ctx,first=prompt_state(task)
        golden=generate('serial',task,start,ctx,first,curves,prior,draft_cost,capture_gold=True)
        assert golden['tokens']==previous_tokens[task['id']]
        record={'id':task['id'],'correctness':[],'timings':[]}
        report['static_control'].append(record)
        for mode in control_modes:
            check=generate(mode,task,start,ctx,first,curves,prior,draft_cost,oracle=golden)
            assert check['tokens']==golden['tokens'] and not target.state_difference(check['state'],golden['state'])
            record['correctness'].append({'mode':mode,'tokens_equal':True,'kept_logits_bit_exact':True,'live_state_bit_exact':True})
            del check
        for rep in range(3):
            offset=(rep+ti)%3
            for slot,mode in enumerate(control_modes[offset:]+control_modes[:offset]):
                result=generate(mode,task,start,ctx,first,curves,prior,draft_cost)
                assert result['tokens']==golden['tokens'] and not target.state_difference(result['state'],golden['state'])
                record['timings'].append({'mode':mode,'repetition':rep,'order':slot,'decode_ms':result['decode_ms'],
                    'tokens_equal':True,'live_state_bit_exact':True,'steps':result['steps']})
                del result
        record['summary']={mode:statistics.median(r['decode_ms'] for r in record['timings'] if r['mode']==mode) for mode in control_modes}
        save();emit({'static_control':task['id'],'frozen_width':calibrated_width,'summary_ms':record['summary']})
        del start,ctx,golden
        gc.collect()
    report['static_control_summary']={mode:sum(t['summary'][mode] for t in report['static_control']) for mode in control_modes}
    base=sum(t['summary']['serial']['median_ms'] for t in report['tasks'])
    tokens=sum(len(t['token_ids'])-1 for t in report['tasks'])
    report['summary']={'tasks':len(TEST),'timing_runs':len(TEST)*repetitions*len(MODES),'timed_tokens':tokens,
        'static_control_timing_runs':72,'all_outputs_and_live_states_exact':True}
    for mode in MODES:
        ms=sum(t['summary'][mode]['median_ms'] for t in report['tasks'])
        report['summary'][mode]={'sum_medians_ms':ms,'ratio':base/ms,'decode_tokens_per_s':tokens*1000/ms}
    report['stage']='complete';save();emit({'summary':report['summary']})
