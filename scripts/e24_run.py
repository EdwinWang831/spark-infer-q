"""Independent coverage producer and fixed-recipe expanded-data fit."""
from collections import Counter
import copy
import hashlib
import json
import time

from e24_contract import CONTRACT,check_seal,sha,write_new,ConversationTokenizer


def tasks(root,phase='all'):
    name={'all':'e24_all_tasks.json','learning':'e24_learning_tasks.json','test':'e24_test_tasks.json'}[phase]
    return json.loads((root/name).read_text())['tasks']


def proxy(session,task):
    view=copy.copy(session);view.tokenizer=ConversationTokenizer(session.tokenizer,task)
    return view


def expected_prompt(session,task):
    messages=task.get('messages',[dict(role='user',content=task['prompt'])])
    ids=session.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                              enable_thinking=False,return_dict=False)
    if 'target_prompt_tokens' in task:
        n=task['target_prompt_tokens'];assert len(ids)>n;ids=ids[:n-128]+ids[-128:]
    return ids,hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def tools(session):
    from e3_wide_target import WideTarget
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_context import CachedContextDraft
    from spark_infer_q.confidence_features import ConfidenceDraft
    from spark_infer_q.local_policy import ProgressModel
    from spark_infer_q.collection_budget import BudgetedTrajectoryDraft
    target=WideTarget(session.weights,capacity=2048)
    base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    native=CachedContextDraft.attach(base);featured=ConfidenceDraft.attach(base)
    prior=json.loads((session.root/'artifacts/e16-selected-model.json').read_text())
    trajectory=BudgetedTrajectoryDraft(ConfidenceDraft.attach(base),ProgressModel(prior['metadata']),
        prior['costs'],prior['cpu_predict_ms'],session.calibration['conditional_prior'])
    return target,native,featured,trajectory


def records(root,phase):
    result=[];files={}
    for shard in range(9 if phase=='learning' else 3):
        meta=json.loads((root/f'artifacts/e24-{phase}-{shard:02d}.json').read_text())
        assert meta['status']=='PASS_E24_COLLECTION' and meta['phase']==phase
        assert meta['seal_sha256']==sha(root/'artifacts/e24-seal.json')
        if phase=='test':assert meta['fit_sha256']==sha(root/'artifacts/e24-fit.json')
        p=root/'artifacts'/meta['dataset'];assert sha(p)==meta['dataset_sha256']
        files[p.name]=sha(p);result.extend(json.loads(line) for line in p.read_text().splitlines())
    assert [r['id'] for r in result]==[t['id'] for t in tasks(root,phase)]
    return result,files


def run(session,arguments):
    root=session.root;stage=arguments['stage'];seal=check_seal(root)
    tag='e24-'+stage
    if stage=='collect':
        phase=arguments['phase'];shard=arguments['shard']
        assert phase in ('learning','test') and type(shard) is int and 0<=shard<(9 if phase=='learning' else 3)
        tag=f'e24-{phase}-{shard:02d}'
    path=root/f'artifacts/{tag}.json';assert not path.exists()
    report=dict(status='RUNNING',stage=stage,contract=CONTRACT,started_unix=time.time(),
        seal_sha256=sha(root/'artifacts/e24-seal.json'),target_object_id=id(session.weights),draft_object_id=id(session.draft))
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(path)
    save();old_threads=session.torch.get_num_threads()
    try:
        session.torch.set_num_threads(2)
        if stage=='preflight':
            all_tasks=tasks(root);assert len(all_tasks)==504 and len({t['group'] for t in all_tasks})==504
            used={t['group'] for n in ('e17_new_tasks.json','e23_test_tasks.json')
                  for t in json.loads((root/n).read_text())['tasks']}
            assert not used&{t['group'] for t in all_tasks}
            lengths=[]
            for t in all_tasks:
                proxy(session,t)  # Validate the role/history contract as well.
                ids,digest=expected_prompt(session,t)
                assert len(ids)+t['output_cap']+8<=2048,(t['id'],len(ids))
                lengths.append(dict(id=t['id'],family=t['family'],split=t['split'],prompt_tokens=len(ids),
                                    message_count=len(t.get('messages',[])) or 1,prompt_sha256=digest))
            report.update(status='PASS_E24_PREFLIGHT',lengths=lengths,new_target_calls=0)
        elif stage=='validate':
            from spark_infer_q.collector import collect_record
            from e20_validate import compatible_record
            target,native,featured,trajectory=tools(session)
            eligible=[t for t in tasks(root,'learning') if t['split']=='train']
            cases=[next(t for t in eligible if t['family']=='external_qa' and t['language']==l) for l in ('en','zh')]
            cases.extend(next(t for t in eligible if t['family']==f) for f in ('code','math','multi_turn'))
            cases.append(next(t for t in eligible if t['family']=='long_context' and t['target_prompt_tokens']==1536))
            report['cases']=[]
            for original in cases:
                task=dict(original,output_cap=16);view=proxy(session,task)
                trajectory.configure(task,calibrate=True)
                a,ga=collect_record(view,target,native,featured,trajectory,task,mode='serial',calibrate=True,trace=True)
                trajectory.configure(task,calibrate=True)
                b,gb=collect_record(view,target,native,featured,trajectory,task,mode='speculative',calibrate=True,trace=True)
                assert a['prompt_sha256']==b['prompt_sha256']==expected_prompt(session,task)[1]
                compatible_record(a,b)
                assert ga['tokens']==gb['tokens'] and session.torch.equal(ga['features'],gb['features'])
                assert not target.state_difference(ga['state'],gb['state'])
                assert (ga['logits'] is None and gb['logits'] is None) or session.torch.equal(ga['logits'],gb['logits'])
                report['cases'].append(dict(id=task['id'],family=task['family'],prompt_tokens=a['prompt_tokens'],
                    output_cap=16,positions=len(a['rows']),exact_tokens=True,exact_features=True,
                    exact_logits=True,exact_state=True,exact_rows=True))
                del ga,gb;save()
            report['status']='PASS_E24_PRODUCER_VALIDATION'
        elif stage=='collect':
            from spark_infer_q.collector import collect_record
            validation=root/'artifacts/e24-validate.json';v=json.loads(validation.read_text())
            assert v['status']=='PASS_E24_PRODUCER_VALIDATION' and len(v['cases'])==6
            if phase=='test':
                fit=root/'artifacts/e24-fit.json';f=json.loads(fit.read_text())
                assert f['eligible_fresh_test'] and sha(root/'artifacts/e24-selected-model.json')==f['selected_model_sha256']
                report['fit_sha256']=sha(fit)
            todo=tasks(root,phase);size=48 if phase=='learning' else 24;todo=todo[shard*size:(shard+1)*size]
            preflight=json.loads((root/'artifacts/e24-preflight.json').read_text())
            assert preflight['status']=='PASS_E24_PREFLIGHT'
            expected={r['id']:r['prompt_sha256'] for r in preflight['lengths']}
            data=root/f'artifacts/{tag}-rows.jsonl';assert not data.exists();data.touch()
            target,native,featured,trajectory=tools(session)
            report.update(phase=phase,shard=shard,tasks=[],dataset=data.name,validation_sha256=sha(validation),producer='e24-role-aware-collection-v1')
            for task in todo:
                trajectory.configure(task,calibrate=task['split']=='calibration')
                rec,transient=collect_record(proxy(session,task),target,native,featured,trajectory,task,
                    mode='speculative',calibrate=task['split']=='calibration')
                assert rec['prompt_sha256']==expected[task['id']]
                with data.open('a') as handle:handle.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n')
                report['tasks'].append({k:v for k,v in rec.items() if k not in ('rows','cost_samples')})
                report['tasks'][-1]['positions']=len(rec['rows']);del transient
                save();print(json.dumps(dict(e24_collected=task['id'],done=len(report['tasks']),total=len(todo))),flush=True)
            report.update(status='PASS_E24_COLLECTION',dataset_sha256=sha(data),positions=sum(t['positions'] for t in report['tasks']))
        elif stage=='fit':
            from e23_train import training_rows,curves,timing
            from e15_metrics import oracle_rate
            from e17_train import expand,replay_positions
            from spark_infer_q.mlp_runtime import NumpyMLPProgressModel
            from spark_infer_q.mlp_head import MLPProgressModel,fit_mlp
            from spark_infer_q.decision_head import fit_decision_mlp
            from spark_infer_q.local_policy import ProgressModel
            from spark_infer_q.confidence_head import headroom_capture,choose_rows
            assert not list((root/'artifacts').glob('e24-test-??.json'))
            old=json.loads((root/'artifacts/e17-fit.json').read_text());train,cal=training_rows(root,old)
            fresh,files=records(root,'learning');new=expand(fresh,'e24_coverage')
            train=train+[r for r in new if r['split']=='train'];cal=cal+[r for r in new if r['split']=='calibration']
            assert not {r['group'] for r in train}&{r['group'] for r in cal}
            reference=json.loads((root/'artifacts/e23-selected-model.json').read_text())
            objective=seal['fixed_objective'];assert objective==reference['candidate']
            started=time.perf_counter()
            if objective=='decision_regret':
                c=curves(train,old['costs'],seal['reference_numpy_fee_ms'])
                patterns=Counter(tuple(zip(r['emitted'][:r['max_rows']],v[:r['max_rows']])) for r,v in zip(train,c))
                q=oracle_rate(patterns)
                model=fit_decision_mlp(train,cal,c,q,alpha=1.,temperature=.1)
                report['train_only_reference_rate']=q
            else:model=fit_mlp(train,cal)
            fit_seconds=time.perf_counter()-started
            models=dict(expanded=NumpyMLPProgressModel(model.metadata),frozen_reference=NumpyMLPProgressModel(reference['metadata']),
                        linear_reference=ProgressModel(old['models'][old['selected_model']]['metadata']))
            times=timing(models,cal);scores={}
            for name in ('expanded','frozen_reference'):
                ps=replay_positions(cal,models[name],'f1',old['costs'],times[name]['cpu_predict_ms'])
                scores[name]=headroom_capture(ps,ps[0]['costs_ms'],old['fixed_rows'])
            selected=max(scores,key=lambda n:scores[n]['policy']);artifacts={}
            scalar=MLPProgressModel(model.metadata);error=0.
            for row,c in zip(cal,curves(cal,old['costs'],times['expanded']['cpu_predict_ms'])):
                p=scalar.predict(row['x']);q=models['expanded'].predict(row['x'])
                error=max(error,max(abs(a-b) for a,b in zip(p,q)))
                assert error<1e-12 and choose_rows(p,c,row['max_rows'])==choose_rows(q,c,row['max_rows'])
            for name,metadata in [('expanded',model.metadata),('frozen_reference',reference['metadata'])]:
                artifact=dict(candidate=name,metadata=metadata,cpu_predict_ms=times[name]['cpu_predict_ms'],
                    feature_group='f1',costs=old['costs'],fixed_rows=old['fixed_rows'],experimental=True)
                p=root/f'artifacts/e24-{name}-model.json';write_new(p,artifact);artifacts[name]=sha(p)
                if name==selected:write_new(root/'artifacts/e24-selected-model.json',artifact)
            report.update(status='PASS_E24_FIT',selected_model=selected,eligible_fresh_test=selected=='expanded',
                model_sha256=artifacts,selected_model_sha256=sha(root/'artifacts/e24-selected-model.json'),
                training_rows=len(train),calibration_rows=len(cal),added_datasets=files,fixed_objective=objective,
                fit_seconds=fit_seconds,timings=times,calibration=scores,costs=old['costs'],fixed_rows=old['fixed_rows'],runtime_max_error=error)
        elif stage=='evaluate':
            from e17_train import expand,replay_positions,prior_comparison
            from e15_metrics import grouped_metrics
            from spark_infer_q.mlp_runtime import NumpyMLPProgressModel
            from spark_infer_q.local_policy import ProgressModel
            fit=json.loads((root/'artifacts/e24-fit.json').read_text());assert fit['eligible_fresh_test']
            fresh,files=records(root,'test');rows=expand(fresh,'e24_fresh_test');outputs={}
            for name in ('expanded','frozen_reference'):
                p=root/f'artifacts/e24-{name}-model.json';assert sha(p)==fit['model_sha256'][name]
                outputs[name]=replay_positions(rows,NumpyMLPProgressModel(json.loads(p.read_text())['metadata']),
                    'f1',fit['costs'],fit['timings'][name]['cpu_predict_ms'])
            old=json.loads((root/'artifacts/e17-fit.json').read_text())
            linear=replay_positions(rows,ProgressModel(old['models'][old['selected_model']]['metadata']),
                'f1',fit['costs'],fit['timings']['linear_reference']['cpu_predict_ms'])
            ps=outputs['expanded'];point=grouped_metrics(ps,fit['fixed_rows'])
            versus=prior_comparison(ps,outputs['frozen_reference']);vs_linear=prior_comparison(ps,linear)
            language={l:grouped_metrics([p for p,r in zip(ps,rows) if r['language']==l],fit['fixed_rows']) for l in ('en','zh')}
            gate=(point['captured'] is not None and point['captured']>=.30 and point['group_bootstrap_gain_95'][0]>0
                and point['group_bootstrap_capture_95'][0]>0 and point['different_from_fixed']>0
                and versus['group_bootstrap_gain_95'][0]>0 and vs_linear['group_bootstrap_gain_95'][0]>0
                and all(language[l]['group_bootstrap_gain_95'][0]>0 for l in ('en','zh')))
            report.update(status='PASS_E24_EVALUATION',gate_pass=bool(gate),test=point,versus_frozen=versus,
                versus_linear=vs_linear,test_by_language=language,datasets=files,
                fit_sha256=sha(root/'artifacts/e24-fit.json'),positions=len(rows),roots=len(fresh))
        else:raise ValueError(stage)
        check_seal(root);assert session.draft.max_context==1024
        report['finished_unix']=time.time();save()
        return dict(status=report['status'],artifact=path.name,
                    **{k:report[k] for k in ('eligible_fresh_test','gate_pass') if k in report},
                    target_object_id=id(session.weights),draft_object_id=id(session.draft))
    finally:session.torch.set_num_threads(old_threads)
