"""Separate immutable-draft-math reuse and greedy EOS verification ablations."""
import gc
import hashlib
import json
import statistics
import time

from e3_policy_tasks import TEST

MODES=('serial','original','cached','eos')


def order(repetition,task_index):
    offset=(repetition+task_index)%4
    return MODES[offset:]+MODES[:offset]


def run(root,checkpoint,w,target,report,save,repetitions):
    import torch
    from transformers import AutoTokenizer
    from spark_infer_q.dflash2_reference import Draft,upstream_helpers
    from spark_infer_q.draft_fastpath import CachedDraft,before_eos
    from spark_infer_q.speculative import greedy_result
    previous=json.loads((root/'artifacts/local-policy.json').read_text())
    assert previous['status']=='PASS_BOUNDED_LOCAL_POLICY'
    previous_tokens={t['id']:t['token_ids'] for t in previous['tasks']}
    h=hashlib.sha256()
    with (root/'checkpoint/model.safetensors').open('rb') as f:
        while chunk:=f.read(8<<20):h.update(chunk)
    assert h.hexdigest()=='d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f'
    cached=CachedDraft(root/'checkpoint',w.embed[0],w.head.b.weight,
        upstream_helpers(root/'reference/dflash.py','b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b'),max_context=1536)
    # One weight set, two implementations of norm/RoPE; no duplicate model load.
    original=object.__new__(Draft);original.__dict__=cached.__dict__.copy()
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    raw=json.loads((checkpoint/'generation_config.json').read_text()).get('eos_token_id',[])
    eos=set(raw if isinstance(raw,list) else [raw]);eos.add(tokenizer.eos_token_id);eos.discard(None)
    report['reference']={'draft_weights_sha256':h.hexdigest(),'previous_policy_model_sha256':previous['frozen_model_sha256']}
    report['measurement_contract']={'modes':MODES,'repetitions':repetitions,
        'order':'four-mode cyclic Latin square','workload':'eight policy-study prompts reused as implementation regression; no new unseen-data claim',
        'timing':'synchronized decode host wall-clock; includes draft, verification and control',
        'excluded':['load','prompt','first pending output','state setup','oracle comparisons'],
        'cached':'only immutable RoPE table and FP32 norm weight copies; hidden/proposals checked bit-exact',
        'eos':'cached drafter plus removal of proposed EOS and all following verifier inputs; greedy only',
        'limits':'same BF16 fidelity and prefill numerical limits; single request, bounded context, no stochastic/full engine claim'}
    # Pointwise math checks include positions outside this short workload.
    torch.manual_seed(319)
    positions=torch.tensor([0,1,31,255,1023,1535],device='cuda')
    x=torch.randn(6,32,128,device='cuda',dtype=torch.bfloat16)
    assert torch.equal(original.rope(x,positions),cached.rope(x,positions))
    for name,value in cached._norm_fp32.items():
        x=torch.randn(3,len(value),device='cuda',dtype=torch.bfloat16);residual=torch.randn_like(x)
        assert torch.equal(original.norm(x,name),cached.norm(x,name))
        for a,b in zip(original.norm(x,name,residual),cached.norm(x,name,residual)):assert torch.equal(a,b)
    report['math_preflight']={'rope_positions':positions.tolist(),'rope_bit_exact':True,'all_norms_bit_exact':True,
        'cache_bytes':cached._cos.numel()*cached._cos.element_size()+cached._sin.numel()*cached._sin.element_size()+sum(v.numel()*v.element_size() for v in cached._norm_fp32.values())}
    def emit(value):print(json.dumps(value,ensure_ascii=False),flush=True)
    def prompt_state(task):
        ids=tokenizer.apply_chat_template([{'role':'user','content':task['prompt']}],tokenize=True,
            add_generation_prompt=True,enable_thinking=task['thinking'],return_dict=False)
        st=target.state();features=[]
        for token in ids:
            lg,ft=target.step(st,[token]);features.append(ft)
        return ids,st,torch.cat(features),int(lg[-1].argmax())
    def generate(mode,task,start,context,first,*,gold=None,capture_gold=False):
        st=start.clone();ctx=context.clone() if mode!='serial' else None
        output=[first];logits_gold=[];steps=[];stopped=first in eos
        torch.cuda.synchronize();t0=time.perf_counter()
        while len(output)<task['limit'] and not stopped:
            before=len(output)
            if mode=='serial':
                lg,_=target.step(st,[output[-1]],capture=False)
                token=int(lg[-1].argmax());output.append(token);stopped=token in eos
                if capture_gold:logits_gold.append(lg.cpu())
                continue
            engine=original if mode=='original' else cached
            if gold is not None:
                base_hidden,_=original.forward_hidden(ctx,output[-1]);cached_hidden,_=cached.forward_hidden(ctx,output[-1])
                assert torch.equal(base_hidden,cached_hidden),(task['id'],before,'hidden')
                assert original.propose(ctx,output[-1])[0]==cached.propose(ctx,output[-1])[0],(task['id'],before,'proposal')
            maximum=min(3,task['limit']-len(output)-1)
            t=time.perf_counter();proposals=engine.propose(ctx,output[-1])[0][:maximum] if maximum else []
            torch.cuda.synchronize();draft_ms=(time.perf_counter()-t)*1000
            original_rows=len(proposals)+1
            if mode=='eos':proposals=before_eos(proposals,eos)
            rows=len(proposals)+1
            t=time.perf_counter();lg,ft,b=target.run(st,[output[-1]]+proposals)
            pred=lg.argmax(-1).tolist()
            out,keep,_,stopped=greedy_result(output[-1],proposals,pred,eos_ids=eos,max_new=task['limit']-len(output))
            target.commit(st,b,rows,keep);verify_ms=(time.perf_counter()-t)*1000
            ctx=torch.cat([ctx,ft[:keep]]);output.extend(out)
            assert len(ctx)==st.pos==start.pos+len(output)-1
            if gold is not None:
                assert out==gold['tokens'][before:before+len(out)],(task['id'],mode,before,'tokens')
                for j in range(keep):assert torch.equal(lg[j:j+1].cpu(),gold['logits'][before-1+j]),(task['id'],mode,before,j,'logits')
            steps.append({'rows':rows,'uncropped_rows':original_rows,'draft_ms':draft_ms,'verify_commit_select_ms':verify_ms,'emitted_tokens':len(out)})
        torch.cuda.synchronize()
        return {'tokens':output,'state':st,'logits':logits_gold,'steps':steps,
            'decode_ms':(time.perf_counter()-t0)*1000,'stopped_by_eos':stopped}

    report['stage']='generation';report['tasks']=[];save()
    for ti,task in enumerate(TEST):
        ids,start,ctx,first=prompt_state(task)
        golden=generate('serial',task,start,ctx,first,capture_gold=True)
        assert golden['stopped_by_eos'] and golden['tokens']==previous_tokens[task['id']]
        # Independent paired microtiming at identical initial features.
        micro={'original':[],'cached':[]}
        original.propose(ctx,first);cached.propose(ctx,first)
        for rep in range(6):
            for name in (('original','cached') if rep%2==0 else ('cached','original')):
                torch.cuda.synchronize();t=time.perf_counter();(original if name=='original' else cached).propose(ctx,first)
                torch.cuda.synchronize();micro[name].append((time.perf_counter()-t)*1000)
        record={**task,'prompt_tokens':len(ids),'token_ids':golden['tokens'],
            'text':tokenizer.decode(golden['tokens'],skip_special_tokens=False),'stopped_by_eos':True,
            'micro_draft_ms':micro,'correctness':[],'timings':[]}
        report['tasks'].append(record);save()
        for mode in MODES[1:]:
            result=generate(mode,task,start,ctx,first,gold=golden)
            assert result['tokens']==golden['tokens']
            assert not target.state_difference(result['state'],golden['state'])
            record['correctness'].append({'mode':mode,'hidden_and_proposals_exact':True,'kept_logits_bit_exact':True,
                'tokens_equal':True,'live_state_bit_exact':True,'steps':result['steps']})
            del result
        for rep in range(repetitions):
            for slot,mode in enumerate(order(rep,ti)):
                result=generate(mode,task,start,ctx,first)
                assert result['tokens']==golden['tokens']
                assert not target.state_difference(result['state'],golden['state'])
                row={'mode':mode,'repetition':rep,'order':slot,'decode_ms':result['decode_ms'],
                    'tokens_equal':True,'live_state_bit_exact':True,'steps':result['steps']}
                record['timings'].append(row);save();emit({'task':task['id'],'mode':mode,'repetition':rep,'decode_ms':row['decode_ms']})
                del result
        baseline=statistics.median(r['decode_ms'] for r in record['timings'] if r['mode']=='serial')
        record['summary']={}
        for mode in MODES:
            values=[r['decode_ms'] for r in record['timings'] if r['mode']==mode];median=statistics.median(values)
            record['summary'][mode]={'median_ms':median,'min_ms':min(values),'max_ms':max(values),
                'ratio':baseline/median,'decode_tokens_per_s':(len(golden['tokens'])-1)*1000/median}
        save();emit({'task_complete':task['id'],'summary':record['summary']})
        del start,ctx,golden
        gc.collect()
    tokens=sum(len(t['token_ids'])-1 for t in report['tasks'])
    baseline=sum(t['summary']['serial']['median_ms'] for t in report['tasks'])
    report['summary']={'tasks':len(TEST),'timing_runs':len(TEST)*repetitions*4,'timed_tokens':tokens,'all_outputs_and_live_states_exact':True}
    for mode in MODES:
        ms=sum(t['summary'][mode]['median_ms'] for t in report['tasks'])
        report['summary'][mode]={'sum_medians_ms':ms,'ratio':baseline/ms,'decode_tokens_per_s':tokens*1000/ms}
    report['stage']='complete';save();emit({'summary':report['summary']})
