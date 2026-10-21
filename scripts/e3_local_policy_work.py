"""Train/calibrate frozen local progress models, then measure seven modes."""
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from e3_policy_tasks import TRAIN, CALIBRATION, TEST
from spark_infer_q.local_policy import MODES, brier, choose_width, fit_model, mode_order


def run(root,checkpoint,w,target,report,save,repetitions):
    import torch
    from transformers import AutoTokenizer
    from spark_infer_q.dflash2_reference import upstream_helpers
    from spark_infer_q.dflash2_scored import ScoredDraft
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.verification_budget import LaggedAcceptance
    from e3_mechanism_checks import entry,params
    import spark_q_e3_varlen

    budget_sha=hashlib.sha256(Path(spark_q_e3_varlen.__file__).read_bytes()).hexdigest()
    assert budget_sha=='722b6ea835fa3e7e349af7f22ac727045a65af487a35f6709f3838e066bd678e'
    h=hashlib.sha256()
    with (root/'checkpoint/model.safetensors').open('rb') as f:
        while chunk:=f.read(8<<20):h.update(chunk)
    assert h.hexdigest()=='d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f'
    draft=ScoredDraft(root/'checkpoint',w.embed[0],w.head.b.weight,
        upstream_helpers(root/'reference/dflash.py','b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b'),max_context=1536)
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    raw=json.loads((checkpoint/'generation_config.json').read_text()).get('eos_token_id',[])
    eos=set(raw if isinstance(raw,list) else [raw]);eos.add(tokenizer.eos_token_id);eos.discard(None)
    generator=torch.Generator(device='cpu').manual_seed(80321)
    feature_projection=(torch.randn(12800,64,generator=generator)/math.sqrt(12800)).cuda()
    embedding_projection=(torch.randn(2560,32,generator=generator)/math.sqrt(2560)).cuda()
    report['reference']={'budget_binary_sha256':budget_sha,'deepinfer_commit':'774ce244b87cc0eb8d4b94ab9f5de2ab22049f01',
        'draft_revision':'a22a1744e85c17364dcc54a6a420af7135d4c7f6','draft_weights_sha256':h.hexdigest()}
    report['environment']={'torch':torch.__version__,'capacity':1536,'eos_ids':sorted(eos),'tf32':False,
        'known_limits':'same E3 BF16 draft arithmetic and unresolved official BF16 / target prefill differences'}
    report['measurement_contract']={'modes':MODES,'repetitions':repetitions,'order':'seven-mode cyclic Latin square',
        'scope':'new eight prompts; single-request greedy; no stochastic or full engine claim',
        'splits':'16 training / 6 independent calibration / 8 test prompts fixed before labels',
        'model':'conditional logistic progress heads, frozen positive-slope Platt calibration; no test fitting',
        'labels':'probability of emitting >1/>2/>3 tokens, including EOS; not raw token softmax probabilities',
        'pre_features':'64 fixed random projections of last target capture, 32 anchor embedding projections, target top2 gap, four known scalars; seed 80321',
        'post_features':'pre features plus nine unary/selected-gap/edge-margin features from current draft',
        'feature_privacy':'raw target captures and projected training rows stay private; public model coefficients are newly fitted small predictors',
        'gate':'pre-model decides whether to pay draft cost; if enabled, verify full supported block',
        'local':'pre-gate then post-model width; once draft is paid its cost applies to every width including zero drafts',
        'timing':'synchronized decode host wall time; all prediction, projection, confidence and control overhead included',
        'excluded':['load','prompt','first pending output','state/model setup','training/calibration','oracle and final state checks'],
        'correctness':'separate per-kept-logit checks; every timed output and final live state checked outside timer',
        'zero_draft':'maintain target features for possible re-enabling; pure serial captures no draft features'}
    def emit(value):print(json.dumps(value,ensure_ascii=False),flush=True)
    def prompt_state(task):
        ids=tokenizer.apply_chat_template([{'role':'user','content':task['prompt']}],tokenize=True,
            add_generation_prompt=True,enable_thinking=task['thinking'],return_dict=False)
        assert 0<len(ids)<256
        st=target.state();features=[]
        for token in ids:
            logits,ft=target.step(st,[token]);features.append(ft)
        return ids,st,torch.cat(features),int(logits[-1].argmax()),logits[-1:]
    def pre_features(ctx,anchor,anchor_logits,offset,thinking,remaining):
        feature=ctx[-1].float();feature=feature*torch.rsqrt(feature.square().mean()+1e-6)
        embedding=w.embed[0][anchor].float();embedding=embedding*torch.rsqrt(embedding.square().mean()+1e-6)
        vals=anchor_logits[-1].float().topk(2).values
        gap=(vals[0]-vals[1]).clamp(0,30).log1p().reshape(1)
        scalar=torch.tensor([math.log1p(len(ctx)),math.log1p(offset),float(thinking),math.log1p(remaining)],device='cuda')
        return torch.cat([feature@feature_projection,embedding@embedding_projection,gap,scalar]).tolist()
    def serial(start,first,limit,*,capture_logits=False):
        st=start.clone();output=[first];gold=[];stopped=first in eos
        torch.cuda.synchronize();t=time.perf_counter()
        while len(output)<limit and not stopped:
            logits,_=target.step(st,[output[-1]],capture=False)
            token=int(logits[-1].argmax());output.append(token);stopped=token in eos
            if capture_logits:gold.append(logits.cpu())
        torch.cuda.synchronize()
        return {'tokens':output,'state':st,'logits':gold,'decode_ms':(time.perf_counter()-t)*1000,'stopped_by_eos':stopped}

    costs={width:[] for width in range(1,5)};base_draft=[];scored_draft=[];feature_cost=[]
    def sample_costs(task,st,ctx,first,anchor_logits):
        proposals=draft.propose(ctx,first)[0]
        scored,meta=draft.propose(ctx,first,confidence=True)
        assert scored==proposals,'confidence sidecar changed proposals'
        for width in range(1,5):
            scratch=st.clone();_,_,b=target.run(scratch,[first]+proposals[:width-1]);target.commit(scratch,b,width,width)
            del scratch
        for rep in range(3):
            for confidence,samples in ((False,base_draft),(True,scored_draft)):
                torch.cuda.synchronize();t=time.perf_counter();got=draft.propose(ctx,first,confidence=confidence)[0]
                torch.cuda.synchronize();samples.append((time.perf_counter()-t)*1000);assert got==proposals
            t=time.perf_counter();pre_features(ctx,first,anchor_logits,0,task['thinking'],task['limit']-1)
            torch.cuda.synchronize();feature_cost.append((time.perf_counter()-t)*1000)
            for wi in range(4):
                width=(wi+rep)%4+1;scratch=st.clone();torch.cuda.synchronize();t=time.perf_counter()
                _,_,b=target.run(scratch,[first]+proposals[:width-1]);target.commit(scratch,b,width,width)
                costs[width].append((time.perf_counter()-t)*1000);del scratch

    def collect(task,*,measure=False):
        ids,start,ctx,first,anchor_logits=prompt_state(task)
        golden=serial(start,first,task['limit'],capture_logits=True)
        st=start.clone();pre=[];post=[];counts=[]
        if measure:sample_costs(task,st,ctx,first,anchor_logits)
        for offset,anchor in enumerate(golden['tokens'][:-1]):
            x=pre_features(ctx,anchor,anchor_logits,offset,task['thinking'],task['limit']-offset-1)
            proposals,meta=draft.propose(ctx,anchor,confidence=True)
            if offset==0:assert draft.propose(ctx,anchor)[0]==proposals,'scored/base proposal mismatch'
            k=min(3,task['limit']-offset-2)
            scratch=st.clone();logits,features,b=target.run(scratch,[anchor]+proposals[:k])
            pred=logits.argmax(-1).tolist()
            out,keep,_,stopped=greedy_result(anchor,proposals[:k],pred,eos_ids=eos,max_new=task['limit']-offset-1)
            assert out==golden['tokens'][offset+1:offset+1+len(out)],(task['id'],offset,'label tokens')
            for j in range(keep):assert torch.equal(logits[j:j+1].cpu(),golden['logits'][offset+j]),(task['id'],offset,j,'label logits')
            raw_accept=0
            for proposed,next_token in zip(proposals[:k],pred):
                if proposed!=next_token:break
                raw_accept+=1
            observed=min(k,keep) if stopped else k
            counts.append((observed,min(raw_accept,observed)))
            labels=[int(len(out)>depth+1) for depth in range(3)]
            pre.append({'x':x,'y':labels});post.append({'x':x+meta['confidence_features'],'y':labels})
            lg,ft=target.step(st,[anchor]);ctx=torch.cat([ctx,ft]);anchor_logits=lg
            assert torch.equal(lg.cpu(),golden['logits'][offset])
            assert torch.equal(ft,features[:1]),(task['id'],offset,'captured feature mismatch')
            del scratch
        assert not target.state_difference(st,golden['state'])
        metadata={**task,'prompt_tokens':len(ids),'token_ids':golden['tokens'],
            'text':tokenizer.decode(golden['tokens'],skip_special_tokens=False),
            'stopped_by_eos':golden['stopped_by_eos'],'rows':len(pre),'kept_logits_and_serial_features_exact':True}
        del st,start,golden,ctx
        return pre,post,counts,metadata

    report['stage']='training_data';report['dataset']={'train':[],'calibration':[]};save()
    training_pre=[];training_post=[];cal_pre=[];cal_post=[];cal_counts=[]
    for split,tasks in (('train',TRAIN),('calibration',CALIBRATION)):
        report['stage']=split+'_data';save()
        for task in tasks:
            pre,post,counts,metadata=collect(task,measure=split=='calibration')
            (training_pre if split=='train' else cal_pre).extend(pre)
            (training_post if split=='train' else cal_post).extend(post)
            if split=='calibration':cal_counts.extend(counts)
            report['dataset'][split].append(metadata);save();emit({'collected':task['id'],'rows':len(pre),'eos':metadata['stopped_by_eos']})
            gc.collect()
    # Projected rows are private, allowing failure diagnosis without publishing target features.
    (root/'artifacts/local-policy-dataset.private.json').write_text(json.dumps({
        'training_pre':training_pre,'training_post':training_post,'cal_pre':cal_pre,'cal_post':cal_post}))
    report['stage']='fit';save()
    pre_model=fit_model(training_pre,cal_pre);post_model=fit_model(training_post,cal_post)
    report['models']={'pre':pre_model.metadata,'post':post_model.metadata,
        'calibration_brier':{'pre':brier(pre_model,cal_pre),'post':brier(post_model,cal_post)}}
    model_bytes=json.dumps(report['models'],sort_keys=True).encode();model_sha=hashlib.sha256(model_bytes).hexdigest()
    report['frozen_model_sha256']=model_sha
    verify=[statistics.median(costs[width]) for width in range(1,5)]
    dm=statistics.median(base_draft);sdm=statistics.median(scored_draft);fm=statistics.median(feature_cost)
    success=[1.,1.,1.];trials=[2.,2.,2.]
    for observed,accepted in cal_counts:
        for j in range(observed):
            if j>accepted:break
            trials[j]+=1;success[j]+=int(j<accepted)
    prior=[s/t for s,t in zip(success,trials)]
    native_curve=[(width,verify[width-1]+(dm if width>1 else 0.)) for width in range(1,5)]
    report['calibration_costs']={'verify_commit_samples_ms':costs,'base_draft_samples_ms':base_draft,
        'scored_draft_samples_ms':scored_draft,'feature_samples_ms':feature_cost,'verify_ms':verify,
        'base_draft_ms':dm,'scored_draft_ms':sdm,'pre_feature_ms':fm,'native_conditional_prior':prior,
        'native_conditional_success':success,'native_conditional_trials':trials}
    save();emit({'frozen_models':model_sha,'training_rows':len(training_pre),'calibration_rows':len(cal_pre),
        'calibration_brier':report['models']['calibration_brier'],'costs':{'verify':verify,'draft':dm,'scored':sdm,'features':fm}})

    def generate(mode,task,start,context,first,initial_logits,*,oracle=None):
        if mode=='serial':return serial(start,first,task['limit'])
        st=start.clone();ctx=context.clone();output=[first];anchor_logits=initial_logits
        steps=[];stopped=first in eos;policy=LaggedAcceptance(prior)
        budget=spark_q_e3_varlen.BudgetProbe();budget.curve(native_curve)
        for _ in range(5):budget.runtime(1,1,verify[0],0.)
        torch.cuda.synchronize();t0=time.perf_counter()
        while len(output)<task['limit'] and not stopped:
            before=len(output);maximum=min(4,task['limit']-len(output));pre_rates=None;post_rates=None
            t=time.perf_counter()
            if mode in ('gate','local'):
                x=pre_features(ctx,output[-1],anchor_logits,before-1,task['thinking'],task['limit']-before)
                pre_rates=pre_model.predict(x)
                draft_estimate=sdm if mode=='local' else dm
                pre_costs=[cost+fm+(draft_estimate if index else 0.) for index,cost in enumerate(verify)]
                width=choose_width(pre_rates,pre_costs,maximum)
                if mode=='gate' and width>1:width=maximum
            elif mode=='dynamic':
                width=budget.schedule(params(),[entry(policy.current,policy.previous,process=len(steps)+2,schedule=len(steps)+2)])[0]
                width=min(width,maximum)
            else:width=min(int(mode[-1])+1,maximum)
            pre_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            if width>1:
                proposals,meta=draft.propose(ctx,output[-1],confidence=mode=='local')
            else:proposals=[];meta=None
            torch.cuda.synchronize();draft_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            if mode=='local' and proposals:
                post_rates=post_model.predict(x+meta['confidence_features'])
                # The drafter and feature extraction have already been paid, even for width1.
                width=choose_width(post_rates,[cost+draft_ms+pre_ms for cost in verify],maximum)
            post_ms=(time.perf_counter()-t)*1000
            k=width-1;proposals=proposals[:k]
            t=time.perf_counter();logits,features,b=target.run(st,[output[-1]]+proposals)
            verify_ms=(time.perf_counter()-t)*1000
            pred=logits.argmax(-1).tolist()
            out,keep,_,stopped=greedy_result(output[-1],proposals,pred,eos_ids=eos,max_new=task['limit']-len(output))
            raw_accept=0
            for proposed,next_token in zip(proposals,pred):
                if proposed!=next_token:break
                raw_accept+=1
            observed=min(k,keep) if stopped else k;accepted=min(raw_accept,observed)
            t=time.perf_counter();target.commit(st,b,width,keep);commit_ms=(time.perf_counter()-t)*1000
            ctx=torch.cat([ctx,features[:keep]]);anchor_logits=logits[keep-1:keep];output.extend(out)
            assert len(ctx)==st.pos==start.pos+len(output)-1
            if oracle is not None:
                assert out==oracle['tokens'][before:before+len(out)],(task['id'],mode,before,'tokens')
                for j in range(keep):assert torch.equal(logits[j:j+1].cpu(),oracle['logits'][before-1+j]),(task['id'],mode,before,j,'logits')
            policy.observe(observed,accepted);budget.runtime(1,width,verify_ms+commit_ms+draft_ms,0.)
            steps.append({'width':width,'draft_invoked':meta is not None,'emitted_tokens':len(out),'accepted_drafts':accepted,
                'observed_drafts':observed,'pre_rates':pre_rates,'post_rates':post_rates,
                'pre_ms':pre_ms,'post_ms':post_ms,'draft_ms':draft_ms,'verify_ms':verify_ms,'commit_ms':commit_ms})
        torch.cuda.synchronize()
        return {'tokens':output,'state':st,'steps':steps,'decode_ms':(time.perf_counter()-t0)*1000,'stopped_by_eos':stopped}

    del training_pre,training_post,cal_pre,cal_post
    report['stage']='heldout_generation';report['tasks']=[];save()
    for ti,task in enumerate(TEST):
        ids,start,ctx,first,initial_logits=prompt_state(task)
        golden=serial(start,first,task['limit'],capture_logits=True)
        record={**task,'prompt_tokens':len(ids),'token_ids':golden['tokens'],
            'text':tokenizer.decode(golden['tokens'],skip_special_tokens=False),
            'stopped_by_eos':golden['stopped_by_eos'],'correctness':[],'timings':[]}
        report['tasks'].append(record);save()
        assert golden['stopped_by_eos'],(task['id'],'test truncated')
        for mode in MODES[1:]:
            check=generate(mode,task,start,ctx,first,initial_logits,oracle=golden)
            assert check['tokens']==golden['tokens']
            assert not target.state_difference(check['state'],golden['state']),(task['id'],mode,'state')
            record['correctness'].append({'mode':mode,'tokens_equal':True,'kept_logits_bit_exact':True,
                'live_state_bit_exact':True,'widths':[s['width'] for s in check['steps']]})
            del check
        for repetition in range(repetitions):
            for order,mode in enumerate(mode_order(repetition,ti)):
                result=generate(mode,task,start,ctx,first,initial_logits)
                assert result['tokens']==golden['tokens'],(task['id'],mode,'timed tokens')
                assert not target.state_difference(result['state'],golden['state']),(task['id'],mode,'timed state')
                row={'mode':mode,'repetition':repetition,'order':order,'decode_ms':result['decode_ms'],
                    'tokens_equal':True,'live_state_bit_exact':True,'steps':result.get('steps',[])}
                record['timings'].append(row);save();emit({'task':task['id'],'mode':mode,'repetition':repetition,
                    'decode_ms':row['decode_ms'],'widths':sorted({s['width'] for s in row['steps']})})
                del result
        baseline=statistics.median(r['decode_ms'] for r in record['timings'] if r['mode']=='serial')
        record['summary']={}
        for mode in MODES:
            values=[r['decode_ms'] for r in record['timings'] if r['mode']==mode];median=statistics.median(values)
            record['summary'][mode]={'median_ms':median,'min_ms':min(values),'max_ms':max(values),
                'ratio_of_medians':baseline/median,'decode_tokens_per_s':(len(golden['tokens'])-1)*1000/median}
        assert hashlib.sha256(json.dumps(report['models'],sort_keys=True).encode()).hexdigest()==model_sha
        save();emit({'task_complete':task['id'],'summary':record['summary']})
        del start,ctx,golden
        gc.collect()
    # Inspect new test prediction accuracy only after completing all timings; no fitting or retuning follows.
    report['stage']='heldout_prediction_metrics';report['heldout_prediction']=[];save()
    for task in TEST:
        pre,post,_,metadata=collect(task)
        report['heldout_prediction'].append({'task':task['id'],'rows':len(pre),
            'pre_brier':brier(pre_model,pre),'post_brier':brier(post_model,post)})
        save()
    base=sum(t['summary']['serial']['median_ms'] for t in report['tasks'])
    timed_tokens=sum(len(t['token_ids'])-1 for t in report['tasks'])
    report['summary']={'tasks':len(TEST),'timing_runs':sum(len(t['timings']) for t in report['tasks']),
        'completed_by_eos':sum(t['stopped_by_eos'] for t in report['tasks']),'timed_tokens':timed_tokens,
        'all_outputs_and_live_states_exact':True,'frozen_model_unchanged':True}
    for mode in MODES:
        ms=sum(t['summary'][mode]['median_ms'] for t in report['tasks'])
        report['summary'][mode]={'sum_medians_ms':ms,'ratio':base/ms,'decode_tokens_per_s':timed_tokens*1000/ms}
    report['stage']='complete';save();emit({'summary':report['summary']})
