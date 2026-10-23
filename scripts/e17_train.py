"""Two stages: fit on old train + fresh learning set; evaluate only after test collection."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time
from e17_run_contract import CONTRACT,load_tasks,sha


def load_collection(root,phase):
    records=[];datasets={};hashes=None
    for i in range(10 if phase=='learning' else 2):
        meta=json.loads((root/f'artifacts/e17-{phase}-{i:02d}.json').read_text())
        assert meta['status']=='PASS_E17_COLLECTION_SHARD' and meta['phase']==phase and meta['contract']==CONTRACT
        path=root/'artifacts'/meta['dataset'];assert sha(path)==meta['dataset_sha256']
        assert hashes is None or hashes==meta['source_sha256'];hashes=meta['source_sha256']
        datasets[path.name]=sha(path);records.extend(json.loads(line) for line in path.read_text().splitlines())
    expected=[t for t in load_tasks(root) if (t['split']=='test')==(phase=='test')]
    assert [r['id'] for r in records]==[t['id'] for t in expected]
    return records,datasets


def expand(records,origin):
    rows=[]
    for rec in records:
        for r in rec['rows']:
            assert len(r['x'])==137 and len(r['y'])==7 and len(r['emitted'])==8
            rows.append(dict(r,task=rec['id'],group=rec['group'],language=rec['language'],split=rec['split'],origin=origin))
    return rows


def load_costs(root):
    records=[];overrides={};datasets={}
    for i in range(8):
        p=root/f'artifacts/e17-cost-{i:02d}.json';j=json.loads(p.read_text())
        assert j['status']=='PASS_E17_COST_SHARD' and not j['smoke']
        datasets[p.name]=sha(p)
        for task in j['tasks']:
            for row in task['positions']:
                assert all(s['exact'] for s in row['samples'])
                overrides[(task['id'],row['position'])]=row['weight'];records.append(row)
    assert len(overrides)==512
    return records,overrides,datasets


def costs_for(root,calibration,old_train,paired,prior):
    costs={}
    for bucket in ('short','long'):
        islong=bucket=='long'
        samples=[s for rec in calibration if (rec['prompt_tokens']>=768)==islong for s in rec['cost_samples']]
        source='new_calibration'
        if not samples:
            samples=[dict(width=s['width'],ms=s['verify_commit_ms']) for r in paired if (r['context_rows']>=768)==islong for s in r['samples']]
            source='paired_old_train'
        if all(any(s['width']==w for s in samples) for w in range(1,9)):
            curve=[statistics.median(s['ms'] for s in samples if s['width']==w) for w in range(1,9)]
        else:curve=prior['costs'][bucket]['verify_commit_ms'];source='frozen_e16_fallback'
        rows=[r for rec in calibration for r in rec['rows'] if (r['context_rows']>=768)==islong]
        if not rows:rows=[r for r in old_train if (r['context_rows']>=768)==islong]
        assert rows,bucket
        native=statistics.median(r['native_draft_ms'] for r in rows);featured=statistics.median(r['feature_draft_ms'] for r in rows)
        costs[bucket]=dict(verify_commit_ms=curve,native_draft_ms=native,feature_draft_ms=featured,
            feature_increment_ms=max(0.,featured-native),source=source,rows=len(rows),timings=len(samples))
    return costs


def vector(row,group):return row['x'][:25] if group=='f0' else row['x']


def replay_positions(rows,model,group,costs,cpu_ms):
    result=[]
    for r in rows:
        c=costs['long' if r['context_rows']>=768 else 'short'];paid=r['max_rows']>1
        base=[v+(c['native_draft_ms'] if paid else 0.) for v in c['verify_commit_ms']]
        extra=c['feature_increment_ms']+cpu_ms if paid else 0.
        result.append(dict(group=r['group'],emitted=r['emitted'],rates=model.predict(vector(r,group)),
                           max_rows=r['max_rows'],baseline_costs_ms=base,costs_ms=[v+extra for v in base]))
    return result


def prior_comparison(policy,reference,samples=2000):
    import random
    from spark_infer_q.confidence_head import choose_rows
    groups={}
    for a,b in zip(policy,reference):
        assert a['group']==b['group'] and a['emitted']==b['emitted']
        sums=groups.setdefault(a['group'],[0.,0.,0.,0.])
        for offset,p in ((0,a),(2,b)):
            w=choose_rows(p['rates'],p['costs_ms'],p['max_rows'])
            sums[offset]+=p['emitted'][w-1];sums[offset+1]+=p['costs_ms'][w-1]
    def gain(keys):
        sums=[sum(groups[k][i] for k in keys) for i in range(4)]
        return (sums[0]/sums[1])/(sums[2]/sums[3])-1
    keys=sorted(groups);rng=random.Random(20261022);draws=sorted(gain(rng.choices(keys,k=len(keys))) for _ in range(samples))
    return dict(relative_gain=gain(keys),group_bootstrap_gain_95=[draws[int(.025*(samples-1))],draws[int(.975*(samples-1))]],groups=len(keys))


def run(session,arguments):
    from spark_infer_q.confidence_head import fit_heads,choose_rows,headroom_capture
    from spark_infer_q.regret_head import fit_regret_heads
    from spark_infer_q.local_policy import ProgressModel
    from spark_infer_q.budget_regret import budget_regret,regret_weight
    from e15_metrics import grouped_metrics
    root=session.root;torch=session.torch;stage=arguments.get('stage','fit');assert stage in ('fit','evaluate')
    names=['e17_train.py','e17_run_contract.py','e15_metrics.py','spark_infer_q/regret_head.py',
        'spark_infer_q/confidence_head.py','spark_infer_q/confidence_features.py','spark_infer_q/budget_regret.py','spark_infer_q/local_policy.py']
    hashes={n:sha(root/n) for n in names};hashes['e17_new_tasks.json']=sha(root/'e17_new_tasks.json')
    prior_path=root/'artifacts/e16-selected-model.json';prior=json.loads(prior_path.read_text());old_model=ProgressModel(prior['metadata'])
    output=root/('artifacts/e17-fit.json' if stage=='fit' else 'artifacts/e17-evaluation.json');assert not output.exists()
    report=dict(status='RUNNING',stage=stage,contract=CONTRACT,started_unix=time.time(),source_sha256=hashes,
        prior_sha256=sha(prior_path),target_object_id=id(session.weights),draft_object_id=id(session.draft))
    def save():
        p=output.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(output)
    save();threads=torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        if stage=='fit':
            assert not list((root/'artifacts').glob('e17-test-??.json')),'test must remain uncollected before selection'
            paired,overrides,cost_files=load_costs(root)
            old_records=[];old_files={}
            for i in range(12):
                meta=json.loads((root/f'artifacts/e16-collection-{i:02d}.json').read_text());p=root/'artifacts'/meta['dataset']
                assert sha(p)==meta['dataset_sha256'];old_files[p.name]=sha(p)
                old_records.extend(r for r in map(json.loads,p.read_text().splitlines()) if r['split']=='train')
            assert len(old_records)==300
            fresh,new_files=load_collection(root,'learning');assert len(fresh)==510
            old_train=expand(old_records,'e16_train');new_rows=expand(fresh,'external')
            training=old_train+[r for r in new_rows if r['split']=='train'];calibration=[r for r in new_rows if r['split']=='calibration']
            costs=costs_for(root,[r for r in fresh if r['split']=='calibration'],old_train,paired,prior)
            for r in training:
                if r['origin']=='e16_train' and (r['task'],r['position']) in overrides:r['weight']=overrides[(r['task'],r['position'])]
                else:
                    p=replay_positions([r],old_model,'f1',costs,prior['cpu_predict_ms'])[0]
                    width=choose_rows(p['rates'],p['costs_ms'],p['max_rows'])
                    regret=budget_regret(r['emitted'],p['costs_ms'],width,CONTRACT['reference_rate'],r['max_rows'])
                    r['weight']=regret_weight(regret['regret_ms'],CONTRACT['old_weight_scale_ms'])
            cp=replay_positions(calibration,old_model,'f1',costs,prior['cpu_predict_ms'])
            def fixed_score(w):
                return sum(p['emitted'][min(w,p['max_rows'])-1] for p in cp)/sum(p['baseline_costs_ms'][min(w,p['max_rows'])-1] for p in cp)
            fixed=max(range(1,9),key=fixed_score)
            files={**old_files,**new_files,**cost_files};data_sha=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
            report.update(costs=costs,fixed_rows=fixed,datasets=files,dataset_sha256=data_sha,models={},
                training_rows=len(training),calibration_rows=len(calibration),
                training_groups=len({r['group'] for r in training}),calibration_groups=len({r['group'] for r in calibration}),
                measured_weight_overrides=512,weight_summary=dict(min=min(r['weight'] for r in training),max=max(r['weight'] for r in training),mean=statistics.mean(r['weight'] for r in training)),
                selection='calibration policy tokens/ms only; test labels collected after sealed selection')
            models={};save()
            for candidate in CONTRACT['candidates']:
                group=candidate.rsplit('_',1)[-1];weighted=candidate.startswith('regret_weighted')
                train=[dict(x=vector(r,group),y=r['y'],**({'weight':r['weight']} if weighted else {})) for r in training]
                cal=[dict(x=vector(r,group),y=r['y']) for r in calibration];t=time.perf_counter()
                with torch.device('cpu'):model=(fit_regret_heads if weighted else fit_heads)(train,cal,l2=CONTRACT['l2'],max_iter=CONTRACT['max_iter'])
                models[candidate]=model;timings=[]
                for _ in range(3):
                    start=time.perf_counter()
                    for r in cal[:256]:model.predict(r['x'])
                    timings.append((time.perf_counter()-start)*1000/min(256,len(cal)))
                cpu=statistics.median(timings);ps=replay_positions(calibration,model,group,costs,cpu)
                report['models'][candidate]=dict(feature_group=group,weighted=weighted,metadata=model.metadata,cpu_predict_ms=cpu,
                    fit_seconds=time.perf_counter()-t,calibration_point=headroom_capture(ps,ps[0]['costs_ms'],fixed))
                save();print(json.dumps(dict(e17_fitted=candidate,calibration=report['models'][candidate]['calibration_point'])),flush=True)
            selected=max(report['models'],key=lambda k:report['models'][k]['calibration_point']['policy'])
            report['selected_model']=selected
            report['paired_model']=('unweighted_' if selected.startswith('regret_weighted') else 'regret_weighted_')+selected.rsplit('_',1)[-1]
            for candidate,entry in report['models'].items():
                artifact=dict(candidate=candidate,feature_group=entry['feature_group'],metadata=entry['metadata'],cpu_predict_ms=entry['cpu_predict_ms'],
                    costs=costs,fixed_rows=fixed,contract=CONTRACT,dataset_sha256=data_sha,experimental=True)
                p=root/f'artifacts/e17-{candidate}-model.json';assert not p.exists();p.write_text(json.dumps(artifact,indent=2)+'\n')
                if candidate==selected:
                    target=root/'artifacts/e17-selected-model.json';assert not target.exists();target.write_text(p.read_text())
            report.update(selected_model_sha256=sha(root/'artifacts/e17-selected-model.json'),status='PASS_E17_FIT',finished_unix=time.time())
        else:
            fit_path=root/'artifacts/e17-fit.json';fit=json.loads(fit_path.read_text());assert fit['status']=='PASS_E17_FIT'
            artifact=root/'artifacts/e17-selected-model.json';assert sha(artifact)==fit['selected_model_sha256']
            fresh,files=load_collection(root,'test');assert len(fresh)==90
            rs=expand(fresh,'external_test');costs=fit['costs'];fixed=fit['fixed_rows'];selected=fit['selected_model']
            report.update(fit_sha256=sha(fit_path),selected_model=selected,selected_model_sha256=sha(artifact),
                test_datasets=files,test_rows=len(rs),test_groups=len({r['group'] for r in rs}),models={})
            replay={}
            for candidate,entry in fit['models'].items():
                model=ProgressModel(entry['metadata']);ps=replay_positions(rs,model,entry['feature_group'],costs,entry['cpu_predict_ms']);replay[candidate]=ps
                report['models'][candidate]=dict(point=headroom_capture(ps,ps[0]['costs_ms'],fixed),
                    cumulative_brier=[statistics.mean((p['rates'][d]-r['y'][d])**2 for p,r in zip(ps,rs)) for d in range(7)])
            point=grouped_metrics(replay[selected],fixed,CONTRACT['bootstrap_samples'])
            languages={lang:grouped_metrics([p for p,r in zip(replay[selected],rs) if r['language']==lang],fixed,CONTRACT['bootstrap_samples']) for lang in ('en','zh','mixed')}
            prior_ps=replay_positions(rs,old_model,'f1',costs,prior['cpu_predict_ms'])
            passed=point['captured'] is not None and point['captured']>=CONTRACT['gate_capture'] and point['group_bootstrap_gain_95'][0]>0
            passed=passed and point['group_bootstrap_capture_95'] is not None and point['group_bootstrap_capture_95'][0]>0
            passed=passed and point['different_from_fixed']>0 and all(languages[lang]['group_bootstrap_gain_95'][0]>0 for lang in ('en','zh'))
            report.update(test=point,test_by_language=languages,versus_frozen_e16=prior_comparison(replay[selected],prior_ps,CONTRACT['bootstrap_samples']),
                gate_pass=bool(passed),online_decision='ELIGIBLE_FOR_NEW_PROMPT_AB' if passed else 'STOP_NO_ONLINE_SPEED_CLAIM',
                status='PASS_E17_EVALUATION',finished_unix=time.time())
        for n,h in hashes.items():assert sha(root/n)==h,n
        assert sha(prior_path)==report['prior_sha256'];save()
        return dict(status=report['status'],artifact=output.name,selected_model=report['selected_model'],
            **({'gate_pass':report['gate_pass']} if stage=='evaluate' else {}),
            target_object_id=id(session.weights),draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());save();raise
    finally:torch.set_num_threads(threads)
