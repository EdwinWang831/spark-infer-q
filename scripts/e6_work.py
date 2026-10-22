"""Controlled draft-head study on the unchanged E3/E4 single-request executor."""
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

from e6_tasks import CALIBRATION,TASKS,HEADS,POLICIES,order


def run(root,checkpoint,w,target,report,save,repetitions,draft_vocab):
    import torch
    from transformers import AutoTokenizer
    from spark_infer_q.dflash2_reference import upstream_helpers
    from spark_infer_q.dflash2_block import BlockDraft
    import importlib
    import spark_infer_q.draft_head as head_module
    importlib.reload(head_module)  # preflight may import it before final study staging
    from spark_infer_q.draft_head import HeadDraft,Q4Backend,make_projection,vocabulary_ids
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.verification_budget import LaggedAcceptance
    from spark_infer_q.wide_budget import conditional_prior
    from e3_mechanism_checks import entry,params
    from tensorfold.families.qwen4_exp.cuda.bf16 import quantize4
    from tensorfold.families.qwen4_exp.cuda.weight_types import draft_token_ids
    from tensorfold.cuda.kernels import qmm
    import spark_q_e3_varlen
    def emit(value):print(json.dumps(value,ensure_ascii=False),flush=True)
    def file_sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            while chunk:=f.read(8<<20):h.update(chunk)
        return h.hexdigest()
    def head_sha():return hashlib.sha256(w.head.b.weight.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    def clock(fn):
        torch.cuda.synchronize();t=time.perf_counter();result=fn();torch.cuda.synchronize()
        return (time.perf_counter()-t)*1000,result
    assert file_sha(Path(spark_q_e3_varlen.__file__))=='722b6ea835fa3e7e349af7f22ac727045a65af487a35f6709f3838e066bd678e'
    assert file_sha(root/'checkpoint/model.safetensors')=='d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f'
    before_sha=head_sha()
    helpers=upstream_helpers(root/'reference/dflash.py','b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b')
    draft=BlockDraft(root/'checkpoint',w.embed[0],w.head.b.weight,helpers,block_size=8,max_context=1024)
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    eos=set(report['preflight']['eos_ids'])
    ids_list=vocabulary_ids(draft_token_ids(str(draft_vocab)),eos,len(w.head.b.weight))
    ids_tensor=torch.tensor(ids_list,device=w.head.b.weight.device,dtype=torch.int64)
    backend=Q4Backend('tensorfold.group32.sm121',
        lambda weight:weight.is_cuda and weight.dtype==torch.bfloat16 and weight.shape[1]%32==0 and torch.cuda.get_device_capability(weight.device)==(12,1),
        quantize4,qmm.matmul)
    report['stage']='building_heads';report['heads']={};save()
    engines={}
    for name in HEADS:
        projection=make_projection(w.head.b.weight,ids=ids_tensor if name.endswith('subset') else None,
            precision='q4' if name.startswith('q4') else 'bf16',backend=backend)
        engines[name]=HeadDraft.attach(draft,projection)
        report['heads'][name]=dict(rows=len(ids_list) if name.endswith('subset') else len(w.head.b.weight),
            incremental_payload_bytes=projection.payload_bytes,requested_precision=projection.requested_precision,
            effective_precision=projection.effective_precision,backend=projection.backend)
        save()
    report['reference']=dict(draft_revision='a22a1744e85c17364dcc54a6a420af7135d4c7f6',
        draft_weights_sha256=file_sha(root/'checkpoint/model.safetensors'),target_head_sha256_before=before_sha,
        vocabulary_sha256=file_sha(draft_vocab),subset_ids_sha256=hashlib.sha256(','.join(map(str,ids_list)).encode()).hexdigest(),
        subset_eos_union=True,budget_binary_sha256=file_sha(Path(spark_q_e3_varlen.__file__)))
    report['measurement_contract']=dict(heads=HEADS,policies=POLICIES,repetitions=repetitions,
        order='four head groups cyclic positions; paired policies alternate, not eight-way Latin square',
        calibration='two independent prompts; per-head acceptance and cost frozen before tests',
        timing='synchronized decode host wall time; excludes load, prompt, initial clones and oracle checks',
        cold_control='first two tasks, fixed7, two repetitions; inner executor prompt plus decode, no API TTFT',
        memory='all four heads resident; compressed head is extra payload while target BF16 head remains',
        scope='single request, greedy, BF16 activations; unchanged target/backbone/selector; no cross-GPU claim',
        inherited_limits='target prefill and eager BF16 DFlash fidelity limitations from E3/E4 remain')
    report['resident']=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
    save()

    def prompt_state(task):
        prompt=tokenizer.apply_chat_template([{'role':'user','content':task['prompt']}],tokenize=True,
            add_generation_prompt=True,enable_thinking=task['thinking'],return_dict=False)
        assert len(prompt)+task['limit']<=1017
        st=target.state();features=[]
        for token in prompt:
            lg,ft=target.step(st,[token]);features.append(ft)
        return prompt,st,torch.cat(features),int(lg[-1].argmax())

    def generate(head,mode,task,start,context,first,curves,prior,draft_cost,*,oracle=None,capture=False):
        st=start.clone();ctx=context.clone() if mode!='serial' else None
        output=[first];stopped=first in eos;steps=[];gold=[]
        policy=LaggedAcceptance(prior);budget=spark_q_e3_varlen.BudgetProbe()
        budget.curve([(r,ms+(draft_cost if r>1 else 0.)) for r,ms in curves])
        for _ in range(5):budget.runtime(1,1,curves[0][1],0.)
        torch.cuda.synchronize();t0=time.perf_counter()
        while len(output)<task['limit'] and not stopped:
            before=len(output)
            if mode=='serial':
                lg,_=target.step(st,[output[-1]],capture=False)
                token=int(lg[-1].argmax());output.append(token);stopped=token in eos
                if capture:gold.append(lg.cpu())
                continue
            width=8 if mode=='fixed7' else budget.schedule(params(),[entry(policy.current,policy.previous,
                process=len(steps)+2,schedule=len(steps)+2)])[0]
            k=min(width-1,task['limit']-len(output)-1)
            dm,proposed=clock(lambda:engines[head].propose(ctx,output[-1])[0][:k]) if k else (0.,[])
            vm,(lg,features,b)=clock(lambda:target.run(st,[output[-1]]+proposed))
            pred=lg.argmax(-1).tolist()
            out,keep,_,stopped=greedy_result(output[-1],proposed,pred,eos_ids=eos,max_new=task['limit']-len(output))
            raw=0
            for p,n in zip(proposed,pred):
                if p!=n:break
                raw+=1
            observed=min(k,keep) if stopped else k;accepted=min(raw,observed)
            cm,_=clock(lambda:target.commit(st,b,k+1,keep))
            ctx=torch.cat([ctx,features[:keep]]);output.extend(out)
            assert len(ctx)==st.pos==start.pos+len(output)-1
            if oracle is not None:
                assert out==oracle['tokens'][before:before+len(out)],(task['id'],head,mode,before,'output')
                for j in range(keep):
                    assert torch.equal(lg[j:j+1].cpu(),oracle['logits'][before-1+j]),(task['id'],head,mode,before,j,'committed logits')
            policy.observe(observed,accepted);budget.runtime(1,k+1,dm+vm+cm,0.)
            steps.append(dict(width=width,verified_rows=k+1,observed_drafts=observed,accepted_drafts=accepted,
                emitted_tokens=len(out),draft_ms=dm,verify_ms=vm,commit_ms=cm))
        torch.cuda.synchronize()
        return dict(tokens=output,state=st,steps=steps,logits=gold,decode_ms=(time.perf_counter()-t0)*1000,stopped_by_eos=stopped)

    report['stage']='calibration';report['profile']=[];report['calibration']={'prompts':[]};save()
    samples={r:[] for r in range(1,9)};draft_samples={h:[] for h in HEADS};cal_steps={h:[] for h in HEADS}
    for ci,task in enumerate(CALIBRATION):
        prompt,st,ctx,first=prompt_state(task)
        # Profile real prompt and a real 16-token serial continuation, never test data.
        profile_contexts=[(ctx,first)]
        continuation=st.clone();cont_ctx=ctx.clone();anchor=first
        for _ in range(16):
            lg,ft=target.step(continuation,[anchor]);cont_ctx=torch.cat([cont_ctx,ft]);anchor=int(lg[-1].argmax())
            if anchor in eos:break
        profile_contexts.append((cont_ctx,anchor))
        for pi,(context,anchor) in enumerate(profile_contexts):
            for engine in engines.values():engine.propose(context,anchor)
            hidden,_=draft.forward_hidden(context,anchor);hidden=hidden[1:]
            assert torch.equal(hidden,engines['bf16_full'].forward_hidden(context,anchor)[0][1:])
            assert draft.propose(context,anchor)==engines['bf16_full'].propose(context,anchor)
            full_logits=engines['bf16_full'].projection.logits(hidden)
            full_values,full_candidates=engines['bf16_full'].projection.topk(hidden)
            base_proposals=engines['bf16_full'].propose(context,anchor)[0]
            metrics={}
            for head in HEADS:
                projection=engines[head].projection
                logits=projection.logits(hidden);values,candidates=projection.topk(hidden)
                reference=full_logits if projection.ids is None else full_logits.index_select(-1,projection.ids)
                difference=logits.float()-reference.float()
                candidates_list=candidates.tolist();base_list=full_candidates.tolist()
                proposals=engines[head].propose(context,anchor)[0]
                metrics[head]=dict(logits_relative_l2=float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(reference.float())),
                    logits_max_abs=float(difference.abs().max()),
                    full_top16_overlap=sum(len(set(a)&set(b)) for a,b in zip(candidates_list,base_list))/(7*16),
                    full_top1_matches=sum(a[0]==b[0] for a,b in zip(candidates_list,base_list)),
                    full_proposal_matches=sum(a==b for a,b in zip(proposals,base_proposals)),proposals=proposals,samples=[])
                if head=='bf16_full':assert torch.equal(values,full_values) and torch.equal(candidates,full_candidates)
            for rep in range(3):
                for head in HEADS[rep:]+HEADS[:rep]:
                    engine=engines[head]
                    backbone_ms,_=clock(lambda:engine.forward_hidden(context,anchor))
                    head_ms,(values,candidates)=clock(lambda:engine.projection.topk(hidden))
                    selector_ms,_=clock(lambda:engine.select(hidden,anchor,values,candidates))
                    proposer_ms,_=clock(lambda:engine.propose(context,anchor))
                    metrics[head]['samples'].append(dict(backbone_ms=backbone_ms,head_topk_ms=head_ms,
                        selector_host_ms=selector_ms,proposer_ms=proposer_ms))
                    draft_samples[head].append(proposer_ms)
            report['profile'].append(dict(calibration_id=task['id'],context_index=pi,prefix_rows=len(context),heads=metrics,baseline_adapter_exact=True))
            save()
        proposals=engines['bf16_full'].propose(ctx,first)[0]
        for rows in range(1,9):
            scratch=st.clone();_,_,b=target.run(scratch,[first]+proposals[:rows-1]);target.commit(scratch,b,rows,rows);del scratch
        for rep in range(3):
            for wi in range(8):
                rows=(wi+rep+ci)%8+1;scratch=st.clone()
                def verify():
                    _,_,b=target.run(scratch,[first]+proposals[:rows-1]);target.commit(scratch,b,rows,rows)
                ms,_=clock(verify);samples[rows].append(ms);del scratch
        curves=[(r,statistics.median(samples[r])) for r in range(1,9)]
        row=dict(id=task['id'],outputs={})
        for head in HEADS:
            result=generate(head,'fixed7',task,st,ctx,first,curves,[.5]*7,statistics.median(draft_samples[head]))
            cal_steps[head].extend(result['steps'])
            row['outputs'][head]=dict(tokens=result['tokens'],stopped_by_eos=result['stopped_by_eos'],steps=result['steps']);del result
        report['calibration']['prompts'].append(row);save();emit(dict(calibrated=task['id']))
        del st,ctx,continuation,cont_ctx,profile_contexts,hidden,full_logits,logits,reference,difference
        gc.collect()
    priors={};costs={}
    report['calibration'].update(verify_commit_curve_ms=curves,verify_samples_ms=samples,heads={})
    for head in HEADS:
        prior,success,trials=conditional_prior(cal_steps[head]);priors[head]=prior;costs[head]=statistics.median(draft_samples[head])
        report['calibration']['heads'][head]=dict(conditional_prior=prior,success=success,trials=trials,
            proposer_samples_ms=draft_samples[head],proposer_median_ms=costs[head],
            native_curve_ms=[(r,ms+(costs[head] if r>1 else 0.)) for r,ms in curves])
    save();emit(dict(frozen_calibration=report['calibration']['heads']))

    report['stage']='balanced_generation';report['tasks']=[];save()
    for ti,task in enumerate(TASKS):
        prompt,start,ctx,first=prompt_state(task)
        gold=generate('bf16_full','serial',task,start,ctx,first,curves,priors['bf16_full'],costs['bf16_full'],capture=True)
        record={**task,'prompt_ids':prompt,'token_ids':gold['tokens'],'text':tokenizer.decode(gold['tokens'],skip_special_tokens=False),
            'stopped_by_eos':gold['stopped_by_eos'],'correctness':[],'timings':[]}
        report['tasks'].append(record);save()
        # Each mode's complete oracle pass also provides a full warm-up before timing.
        for head in HEADS:
            for policy in POLICIES:
                result=generate(head,policy,task,start,ctx,first,curves,priors[head],costs[head],oracle=gold)
                assert result['tokens']==gold['tokens'] and not target.state_difference(result['state'],gold['state']),(task['id'],head,policy,'state')
                record['correctness'].append(dict(head=head,policy=policy,tokens_equal=True,committed_logits_exact=True,
                    live_state_exact=True,steps=result['steps']));del result
        for rep in range(repetitions):
            for slot,(head,policy) in enumerate(order(rep,ti)):
                result=generate(head,policy,task,start,ctx,first,curves,priors[head],costs[head])
                assert result['tokens']==gold['tokens'] and not target.state_difference(result['state'],gold['state']),(task['id'],head,policy,'timed state')
                record['timings'].append(dict(head=head,policy=policy,repetition=rep,order=slot,decode_ms=result['decode_ms'],
                    tokens_equal=True,live_state_exact=True,steps=result['steps']))
                save();emit(dict(task=task['id'],head=head,policy=policy,rep=rep,decode_ms=result['decode_ms']))
                del result
        del start,ctx,gold;gc.collect()
    report['stage']='cold_prompt_decode';report['cold_controls']=[];save()
    for ti,task in enumerate(TASKS[:2]):
        expected=report['tasks'][ti]['token_ids'];control=dict(id=task['id'],timings=[]);report['cold_controls'].append(control)
        for rep in range(2):
            heads=HEADS[rep:]+HEADS[:rep]
            for head in heads:
                torch.cuda.synchronize();t0=time.perf_counter()
                prompt,st,ctx,first=prompt_state(task)
                result=generate(head,'fixed7',task,st,ctx,first,curves,priors[head],costs[head])
                torch.cuda.synchronize();total_ms=(time.perf_counter()-t0)*1000
                assert result['tokens']==expected
                control['timings'].append(dict(head=head,repetition=rep,total_ms=total_ms,decode_ms=result['decode_ms'],tokens_equal=True))
                del st,ctx,result;save()
    report['reference']['target_head_sha256_after']=head_sha()
    assert report['reference']['target_head_sha256_after']==before_sha
    report['stage']='complete';save();emit(dict(study_complete=True))
