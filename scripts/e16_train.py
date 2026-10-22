"""CPU-only seven-depth fitting and frozen held-out gate in the resident worker."""
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from e16_tasks import TASKS,CONTRACT


def run(session,arguments):
    from spark_infer_q.confidence_head import fit_heads,choose_rows,headroom_capture
    from spark_infer_q.local_policy import ProgressModel
    from e15_metrics import grouped_metrics
    root=session.root;torch=session.torch
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    collections=[];records=[];datasets={}
    for shard in range(12):
        collection=json.loads((root/f'artifacts/e16-collection-{shard:02d}.json').read_text())
        assert collection['status']=='PASS_E16_COLLECTION_SHARD' and collection['contract']==CONTRACT
        datafile=root/'artifacts'/collection['dataset']
        assert sha(datafile)==collection['dataset_sha256']
        assert collection['source_sha256']==(collections[0]['source_sha256'] if collections else collection['source_sha256'])
        collections.append(collection);datasets[datafile.name]=collection['dataset_sha256']
        records.extend(json.loads(line) for line in datafile.read_text().splitlines())
    collection=dict(source_sha256=collections[0]['source_sha256'],
                    dataset_sha256=hashlib.sha256(json.dumps(datasets,sort_keys=True).encode()).hexdigest())
    assert [r['id'] for r in records]==[t['id'] for t in TASKS]
    group_splits=defaultdict(set)
    rows=[]
    for rec in records:
        group_splits[rec['group']].add(rec['split'])
        for r in rec['rows']:
            assert len(r['x'])==137 and len(r['y'])==7 and len(r['emitted'])==8
            assert r['remaining']<=rec['output_cap']-1
            rows.append(dict(r,task=rec['id'],group=rec['group'],split=rec['split'],language=rec['language'],family=rec['family'],eos=rec['eos']))
    assert all(len(v)==1 for v in group_splits.values())
    splits={s:[r for r in rows if r['split']==s] for s in ('train','calibration','test','ood')}
    bucket=lambda r:'long' if r['context_rows']>=768 else 'short'
    costs={}
    for name in ('short','long'):
        rs=[r for r in splits['calibration'] if bucket(r)==name]
        samples=[s for rec in records if rec['split']=='calibration' and ('long' if rec['prompt_tokens']>=768 else 'short')==name for s in rec['cost_samples']]
        curve=[statistics.median(s['ms'] for s in samples if s['width']==w) for w in range(1,9)]
        native=statistics.median(r['native_draft_ms'] for r in rs)
        featured=statistics.median(r['feature_draft_ms'] for r in rs)
        costs[name]=dict(verify_commit_ms=curve,native_draft_ms=native,feature_draft_ms=featured,
                         feature_increment_ms=max(0.,featured-native),rows=len(rs),timings=len(samples))
    def baseline_cost(r):
        c=costs[bucket(r)];draft=c['native_draft_ms'] if r['max_rows']>1 else 0.
        return [v+draft for v in c['verify_commit_ms']]
    def fixed_score(width,rs):
        return sum(r['emitted'][min(width,r['max_rows'])-1] for r in rs)/sum(baseline_cost(r)[min(width,r['max_rows'])-1] for r in rs)
    fixed=max(range(1,9),key=lambda w:fixed_score(w,splits['calibration']))
    report=dict(status='RUNNING',contract=CONTRACT,started_unix=time.time(),dataset_sha256=collection['dataset_sha256'],
                collection_source_sha256=collection['source_sha256'],datasets=datasets,costs=costs,fixed_rows=fixed,models={},
                split_rows={s:len(v) for s,v in splits.items()},
                split_groups={s:len({r['group'] for r in v}) for s,v in splits.items()},
                source_sha256={n:sha(root/n) for n in ['e16_train.py','e15_metrics.py','e16_tasks.py',
                    'spark_infer_q/confidence_head.py','spark_infer_q/local_policy.py','spark_infer_q/confidence_features.py']},
                target_object_id=id(session.weights),draft_object_id=id(session.draft))
    output=root/'artifacts/e16-training.json'
    def save():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def vector(r,name):return r['x'][:25] if name=='f0' else r['x']
    models={};predictions={};cpu={}
    def positions(name,split):
        result=[]
        for r,rates in zip(splits[split],predictions[name][split]):
            base=baseline_cost(r);c=costs[bucket(r)]
            extra=c['feature_increment_ms']+cpu[name] if r['max_rows']>1 else 0.
            result.append(dict(group=r['group'],emitted=r['emitted'],rates=rates,max_rows=r['max_rows'],
                               baseline_costs_ms=base,costs_ms=[v+extra for v in base]))
        return result
    save();old_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        for name in ('f0','f1'):
            t=time.perf_counter()
            training=[dict(x=vector(r,name),y=r['y']) for r in splits['train']]
            calibration=[dict(x=vector(r,name),y=r['y']) for r in splits['calibration']]
            with torch.device('cpu'):
                model=fit_heads(training,calibration,l2=CONTRACT['l2'],max_iter=CONTRACT['max_iter'])
            models[name]=model
            samples=[];probe=calibration[:256]
            for _ in range(3):
                t0=time.perf_counter()
                for r in probe:model.predict(r['x'])
                samples.append((time.perf_counter()-t0)*1000/len(probe))
            cpu[name]=statistics.median(samples)
            # Select model using calibration only. Test predictions are deferred.
            predictions[name]={'calibration':[model.predict(vector(r,name)) for r in splits['calibration']]}
            ps=positions(name,'calibration');cal=headroom_capture(ps,ps[0]['costs_ms'],fixed)
            report['models'][name]=dict(metadata=model.metadata,cpu_predict_ms=cpu[name],fit_seconds=time.perf_counter()-t,
                                        calibration_point=cal)
            save();print(json.dumps(dict(e16_fitted=name,calibration=cal)),flush=True)
        selected=max(models,key=lambda name:report['models'][name]['calibration_point']['policy'])
        report['selected_model']=selected;report['selection']='calibration policy tokens/ms only; frozen before test predictions'
        save()
        for name,model in models.items():
            diagnostics={}
            for split,rs in splits.items():
                if split not in predictions[name]:predictions[name][split]=[model.predict(vector(r,name)) for r in rs]
                pp=predictions[name][split]
                brier=[statistics.mean((p[d]-r['y'][d])**2 for r,p in zip(rs,pp)) for d in range(7)]
                conditional=[]
                for d in range(7):
                    observed=[(r,p) for r,p in zip(rs,pp) if d==0 or r['y'][d-1]]
                    conditional.append(dict(trials=len(observed),success=sum(r['y'][d] for r,p in observed),
                        mean_prediction=statistics.mean(p[d]/(p[d-1] if d else 1.) for r,p in observed) if observed else None))
                ps=positions(name,split)
                point=headroom_capture(ps,ps[0]['costs_ms'],fixed)
                diagnostics[split]=dict(cumulative_brier=brier,conditional=conditional,point=point,
                    width_counts=dict(Counter(choose_rows(p['rates'],p['costs_ms'],p['max_rows']) for p in ps)))
            report['models'][name]['diagnostics']=diagnostics
        evaluations={s:grouped_metrics(positions(selected,s),fixed,CONTRACT['bootstrap_samples']) for s in ('calibration','test','ood')}
        test=evaluations['test'];cal=evaluations['calibration']
        passed=(test['captured'] is not None and test['captured']>=CONTRACT['gate_capture']
                and test['group_bootstrap_gain_95'][0]>0 and test['group_bootstrap_capture_95'] is not None
                and test['group_bootstrap_capture_95'][0]>0 and test['undefined_capture_draws']<=100
                and test['different_from_fixed']>0 and cal['different_from_fixed']>0 and cal['relative_gain']>0)
        language_evaluations={}
        for language in ('en','zh','mixed'):
            ps=[p for p,r in zip(positions(selected,'test'),splits['test']) if r['language']==language]
            language_evaluations[language]=grouped_metrics(ps,fixed,CONTRACT['bootstrap_samples'])
        passed=passed and all(language_evaluations[lang]['group_bootstrap_gain_95'][0]>0 for lang in ('en','zh'))
        report['test_by_language']=language_evaluations
        report['sampling']='at most 64 uniformly spaced positions per prompt; reported replay is sample-weighted'
        report.update(evaluations=evaluations,gate_pass=passed,status='PASS_E16_TRAINING',
                      online_decision='ELIGIBLE_FOR_NEW_PROMPT_AB' if passed else 'STOP_NO_ONLINE_SPEED_CLAIM',finished_unix=time.time())
        for name,digest in report['source_sha256'].items():assert sha(root/name)==digest,name
        save()
        model_artifact=dict(feature_group=selected,metadata=models[selected].metadata,costs=costs,
                            cpu_predict_ms=cpu[selected],fixed_rows=fixed,contract=CONTRACT,
                            dataset_sha256=collection['dataset_sha256'],gate_pass=passed)
        (root/'artifacts/e16-selected-model.json').write_text(json.dumps(model_artifact,indent=2,allow_nan=False)+'\n')
        for name in models:
            artifact=dict(model_artifact,feature_group=name,metadata=models[name].metadata,cpu_predict_ms=cpu[name],selected=name==selected)
            (root/f'artifacts/e16-{name}-model.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
        return dict(status=report['status'],artifact=output.name,selected_model=selected,gate_pass=passed,
                    online_decision=report['online_decision'],target_object_id=id(session.weights),draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());save();raise
    finally:torch.set_num_threads(old_threads)
