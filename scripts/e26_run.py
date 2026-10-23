"""Fit the frozen-base residual and matched controls using train/calibration only."""
from collections import Counter
import copy
import json
import time

from e26_contract import CONTRACT,check_seal,sha,write_new


def run(session,arguments):
    from e23_train import training_rows,timing
    from e24_run import records,expected_prompt
    from e17_train import expand,replay_positions
    from spark_infer_q.local_policy import ProgressModel
    from spark_infer_q.residual_head import NumpyLinearProgressModel,ResidualProgressModel,fit_residual
    from spark_infer_q.mlp_head import MLPProgressModel,fit_mlp
    from spark_infer_q.mlp_runtime import NumpyMLPProgressModel
    from spark_infer_q.confidence_head import choose_rows,headroom_capture
    assert not arguments
    root=session.root;torch=session.torch;seal=check_seal(root)
    p=root/'artifacts/e26-fit.json';assert not p.exists()
    report=dict(status='RUNNING',seal_sha256=sha(root/'artifacts/e26-seal.json'),contract=CONTRACT,
        started_unix=time.time(),target_object_id=id(session.weights),draft_object_id=id(session.draft))

    def save():
        temp=p.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(p)

    save();threads=torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        tasks=json.loads((root/'e26_test_tasks.json').read_text())['tasks']
        assert len(tasks)==len({t['group'] for t in tasks})==12
        excluded={t['group'] for n in ('e17_new_tasks.json','e23_test_tasks.json','e24_all_tasks.json') for t in json.loads((root/n).read_text())['tasks']}
        assert not excluded&{t['group'] for t in tasks}
        lengths=[]
        for t in tasks:
            ids,h=expected_prompt(session,t)
            assert len(ids)+t['output_cap']+8<=2048
            lengths.append(dict(id=t['id'],tokens=len(ids),prompt_sha256=h))
        report['preflight']=dict(prompts=12,lengths=lengths,new_target_calls=0)
        save()
        old=json.loads((root/'artifacts/e17-fit.json').read_text());prior24=json.loads((root/'artifacts/e24-fit.json').read_text())
        train,cal=training_rows(root,old);fresh,files=records(root,'learning');new=expand(fresh,'e24_coverage')
        train += [r for r in new if r['split']=='train'];cal += [r for r in new if r['split']=='calibration']
        assert len(train)==prior24['training_rows']==67277 and len(cal)==prior24['calibration_rows']==10054
        assert not {r['group'] for r in train}&{r['group'] for r in cal}
        assert not ({r['group'] for r in train}|{r['group'] for r in cal})&{t['group'] for t in tasks}
        base=copy.deepcopy(old['models'][old['selected_model']]['metadata'])
        original=ProgressModel(base);fast=NumpyLinearProgressModel(base)
        error=0.;changed=0
        for r in train+cal:
            a,b=original.predict(r['x']),fast.predict(r['x']);error=max(error,max(abs(x-y) for x,y in zip(a,b)))
            bucket=old['costs']['long' if r['context_rows']>=768 else 'short']
            paid=bucket['native_draft_ms']+bucket['feature_increment_ms'] if r['max_rows']>1 else 0.
            c=[v+paid for v in bucket['verify_commit_ms']]
            changed+=choose_rows(a,c,r['max_rows'])!=choose_rows(b,c,r['max_rows'])
        assert error<1e-12 and changed==0
        started=time.perf_counter();hybrid=fit_residual(train,base,hidden=16,seed=CONTRACT['seed'],l2=CONTRACT['l2'],max_iter=80)
        residual_seconds=time.perf_counter()-started
        assert hybrid.residual_parameter_count==2327 and hybrid.metadata['base']==base
        started=time.perf_counter();pure=fit_mlp(train,cal,hidden=(64,32),seed=CONTRACT['seed'],l2=CONTRACT['l2'],max_iter=80)
        mlp_seconds=time.perf_counter()-started
        models=dict(linear_original=original,linear_fast=fast,mlp=NumpyMLPProgressModel(pure.metadata))
        mlp_error=0.
        for r in cal:
            a,b=pure.predict(r['x']),models['mlp'].predict(r['x']);mlp_error=max(mlp_error,max(abs(x-y) for x,y in zip(a,b)))
            bucket=old['costs']['long' if r['context_rows']>=768 else 'short']
            paid=bucket['native_draft_ms']+bucket['feature_increment_ms'] if r['max_rows']>1 else 0.
            c=[v+paid for v in bucket['verify_commit_ms']]
            assert choose_rows(a,c,r['max_rows'])==choose_rows(b,c,r['max_rows'])
        assert mlp_error<1e-12
        for alpha in CONTRACT['alphas']:
            m=copy.deepcopy(hybrid.metadata);m['alpha']=alpha;models['alpha_'+str(alpha)]=ResidualProgressModel(m)
        times=timing(models,cal);scores={};choices={}
        for name,model in models.items():
            ps=replay_positions(cal,model,'f1',old['costs'],times[name]['cpu_predict_ms'])
            scores[name]=headroom_capture(ps,ps[0]['costs_ms'],old['fixed_rows'])
            choices[name]=[choose_rows(q['rates'],q['costs_ms'],q['max_rows']) for q in ps]
        chosen=max(('alpha_'+str(a) for a in CONTRACT['alphas']),key=lambda n:scores[n]['policy'])
        selected=models[chosen];alpha=selected.metadata['alpha']
        different=sum(a!=b for a,b in zip(choices[chosen],choices['linear_fast']))
        eligible=alpha>0 and scores[chosen]['policy']>scores['linear_fast']['policy'] and different>0
        artifacts={}
        for name,metadata in [('hybrid',selected.metadata),('linear_fast',base),('mlp',pure.metadata)]:
            dest=root/f'artifacts/e26-{name}-model.json'
            write_new(dest,dict(candidate=name,metadata=metadata,feature_group='f1',experimental=True))
            artifacts[name]=sha(dest)
        runtime_times={n:times[chosen if n=='hybrid' else n] for n in ('hybrid','linear_fast','mlp')}
        report.update(status='PASS_E26_FIT',selected_alpha=alpha,eligible_online=bool(eligible),
            training_rows=len(train),calibration_rows=len(cal),added_datasets=files,
            residual_parameter_count=selected.residual_parameter_count,residual_fit_seconds=residual_seconds,mlp_fit_seconds=mlp_seconds,
            base_unchanged=selected.metadata['base']==base,fast_linear_max_error=error,fast_linear_changed_actions=changed,
            mlp_numpy_max_error=mlp_error,calibration=scores,calibration_changed_widths=different,
            model_sha256=artifacts,timings=runtime_times,all_cpu_timings=times,costs=old['costs'],fixed_rows=old['fixed_rows'])
        check_seal(root);assert session.draft.max_context==1024
        report['finished_unix']=time.time();save()
        return dict(status='PASS_E26_FIT',eligible_online=bool(eligible),artifact=p.name,fit_sha256=sha(p),
            target_object_id=id(session.weights),draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');save();raise
    finally:torch.set_num_threads(threads)
