"""Train-only action loss versus the frozen same-structure BCE MLP."""
from collections import Counter
import json
import statistics
import time

from e23_contract import CONTRACT, check_seal, sha, write_new
from e17_train import load_collection, expand, replay_positions, prior_comparison
from e15_metrics import oracle_rate, grouped_metrics
from spark_infer_q.confidence_head import headroom_capture, choose_rows
from spark_infer_q.local_policy import ProgressModel
from spark_infer_q.mlp_head import MLPProgressModel
from spark_infer_q.mlp_runtime import NumpyMLPProgressModel
from spark_infer_q.decision_head import fit_decision_mlp


def curves(rows, costs, fee):
    result=[]
    for row in rows:
        c=costs['long' if row['context_rows']>=768 else 'short']
        paid=row['max_rows']>1
        extra=c['native_draft_ms']+c['feature_increment_ms']+fee if paid else 0.
        result.append([v+extra for v in c['verify_commit_ms']])
    return result


def training_rows(root, e17):
    # This helper deliberately never opens E17 test collections.
    for name,digest in e17['datasets'].items():
        assert sha(root/'artifacts'/name)==digest,name
    old=[]
    for i in range(12):
        meta=json.loads((root/f'artifacts/e16-collection-{i:02d}.json').read_text())
        old.extend(r for r in map(json.loads,(root/'artifacts'/meta['dataset']).read_text().splitlines()) if r['split']=='train')
    fresh,_=load_collection(root,'learning')
    rows=expand(fresh,'external')
    train=expand(old,'e16_train')+[r for r in rows if r['split']=='train']
    calibration=[r for r in rows if r['split']=='calibration']
    assert len(train)==e17['training_rows'] and len(calibration)==e17['calibration_rows']
    assert not {r['group'] for r in train}&{r['group'] for r in calibration}
    return train,calibration


def timing(models, rows):
    rows=rows[:256]
    names=list(models);values={n:[] for n in names}
    for model in models.values():
        for r in rows[:32]:model.predict(r['x'])
    for rep in range(9):
        order=names[rep%len(names):]+names[:rep%len(names)]
        if rep%2:order.reverse()
        for name in order:
            start=time.perf_counter()
            for r in rows:models[name].predict(r['x'])
            values[name].append((time.perf_counter()-start)*1000/len(rows))
    return {n:dict(cpu_predict_ms=statistics.median(v),repeats_ms=v) for n,v in values.items()}


def run(session, arguments):
    root=session.root;stage=arguments['stage'];assert stage in ('preflight','fit','evaluate')
    seal=check_seal(root)
    path=root/f'artifacts/e23-{stage}.json';assert not path.exists()
    result=dict(status='RUNNING',stage=stage,contract=CONTRACT,started_unix=time.time(),
                seal_sha256=sha(root/'artifacts/e23-seal.json'),
                target_object_id=id(session.weights),draft_object_id=id(session.draft))
    threads=session.torch.get_num_threads()
    try:
        session.torch.set_num_threads(2)
        if stage=='preflight':
            tasks=json.loads((root/'e23_test_tasks.json').read_text())['tasks']
            before={t['group'] for t in json.loads((root/'e17_new_tasks.json').read_text())['tasks']}
            assert not before&{t['group'] for t in tasks}
            lengths=[]
            for t in tasks:
                assert t['split']=='test'
                ids=session.tokenizer.apply_chat_template([dict(role='user',content=t['prompt'])],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False,return_dict=False)
                assert len(ids)+t['output_cap']+8<=2048
                lengths.append(len(ids))
            result.update(status='PASS_E23_PREFLIGHT',prompts=len(tasks),roots=len({t['group'] for t in tasks}),
                          max_prompt_tokens=max(lengths),new_target_calls=0)
        elif stage=='fit':
            assert not list((root/'artifacts').glob('e23-test-??.json'))
            e17=json.loads((root/'artifacts/e17-fit.json').read_text())
            bce=json.loads((root/'artifacts/e19-selected-model.json').read_text())
            assert bce['candidate']=='unweighted_f1' and bce['dataset_sha256']==e17['dataset_sha256']
            train,cal=training_rows(root,e17)
            # Calibrated E22 timing is frozen input; not chosen from test outcomes.
            fee=seal['reference_numpy_fee_ms']
            cost=curves(train,e17['costs'],fee)
            patterns=Counter(tuple(zip(r['emitted'][:r['max_rows']],c[:r['max_rows']])) for r,c in zip(train,cost))
            reference=oracle_rate(patterns)
            started=time.perf_counter()
            model=fit_decision_mlp(train,cal,cost,reference,alpha=CONTRACT['alpha'],temperature=CONTRACT['temperature'],
                hidden=(64,32),seed=CONTRACT['seed'],l2=CONTRACT['l2'],max_iter=CONTRACT['max_iter'])
            fit_seconds=time.perf_counter()-started
            metadata=dict(bce_reference=bce['metadata'],decision_regret=model.metadata)
            models={n:NumpyMLPProgressModel(m) for n,m in metadata.items()}
            models['linear_reference']=ProgressModel(e17['models'][e17['selected_model']]['metadata'])
            times=timing(models,cal)
            evaluations={}
            for n in CONTRACT['candidates']:
                ps=replay_positions(cal,models[n],'f1',e17['costs'],times[n]['cpu_predict_ms'])
                evaluations[n]=headroom_capture(ps,ps[0]['costs_ms'],e17['fixed_rows'])
            selected=max(CONTRACT['candidates'],key=lambda n:evaluations[n]['policy'])
            # Validate the new runtime on every calibration input, including actions.
            scalar=MLPProgressModel(model.metadata);maximum=0.
            for r,c in zip(cal,curves(cal,e17['costs'],times['decision_regret']['cpu_predict_ms'])):
                p=scalar.predict(r['x']);q=models['decision_regret'].predict(r['x'])
                maximum=max(maximum,max(abs(a-b) for a,b in zip(p,q)))
                assert maximum<1e-12 and choose_rows(p,c,r['max_rows'])==choose_rows(q,c,r['max_rows'])
            artifacts={}
            for n,m in metadata.items():
                artifact=dict(candidate=n,metadata=m,cpu_predict_ms=times[n]['cpu_predict_ms'],
                    dataset_sha256=e17['dataset_sha256'],costs=e17['costs'],fixed_rows=e17['fixed_rows'],
                    architecture='mlp',feature_group='f1',experimental=True,contract=CONTRACT)
                p=root/f'artifacts/e23-{n}-model.json';write_new(p,artifact);artifacts[n]=sha(p)
                if n==selected:write_new(root/'artifacts/e23-selected-model.json',artifact)
            result.update(status='PASS_E23_FIT',selected_model=selected,eligible_fresh_test=selected=='decision_regret',
                training_rows=len(train),calibration_rows=len(cal),dataset_sha256=e17['dataset_sha256'],
                fit_seconds=fit_seconds,train_only_reference_rate=reference,timings=times,
                calibration=evaluations,model_sha256=artifacts,
                selected_model_sha256=sha(root/'artifacts/e23-selected-model.json'),
                calibration_runtime_max_error=maximum,costs=e17['costs'],fixed_rows=e17['fixed_rows'])
        else:
            fit=json.loads((root/'artifacts/e23-fit.json').read_text());assert fit['eligible_fresh_test']
            rows=[];datasets={}
            expected=json.loads((root/'e23_test_tasks.json').read_text())['tasks'];ids=[]
            for i in range(3):
                meta=json.loads((root/f'artifacts/e23-test-{i:02d}.json').read_text())
                assert meta['status']=='PASS_E23_TEST_COLLECTION' and meta['fit_sha256']==sha(root/'artifacts/e23-fit.json')
                p=root/'artifacts'/meta['dataset'];assert sha(p)==meta['dataset_sha256']
                records=[json.loads(line) for line in p.read_text().splitlines()];ids.extend(r['id'] for r in records)
                rows.extend(expand(records,'e23_fresh_test'));datasets[p.name]=sha(p)
            assert ids==[t['id'] for t in expected]
            e17=json.loads((root/'artifacts/e17-fit.json').read_text())
            outputs={};models={}
            for n in CONTRACT['candidates']:
                p=root/f'artifacts/e23-{n}-model.json';assert sha(p)==fit['model_sha256'][n]
                models[n]=NumpyMLPProgressModel(json.loads(p.read_text())['metadata'])
                outputs[n]=replay_positions(rows,models[n],'f1',fit['costs'],fit['timings'][n]['cpu_predict_ms'])
            linear=replay_positions(rows,ProgressModel(e17['models'][e17['selected_model']]['metadata']),
                                    'f1',fit['costs'],fit['timings']['linear_reference']['cpu_predict_ms'])
            ps=outputs['decision_regret'];point=grouped_metrics(ps,fit['fixed_rows'])
            vs_bce=prior_comparison(ps,outputs['bce_reference']);vs_linear=prior_comparison(ps,linear)
            languages={lang:grouped_metrics([p for p,r in zip(ps,rows) if r['language']==lang],fit['fixed_rows']) for lang in ('en','zh','mixed')}
            gate=(point['captured']>=CONTRACT['capture'] and point['group_bootstrap_gain_95'][0]>0
                  and point['group_bootstrap_capture_95'][0]>0 and point['different_from_fixed']>0
                  and vs_bce['group_bootstrap_gain_95'][0]>0 and vs_linear['group_bootstrap_gain_95'][0]>0
                  and all(languages[l]['group_bootstrap_gain_95'][0]>0 for l in ('en','zh')))
            result.update(status='PASS_E23_EVALUATION',gate_pass=bool(gate),test=point,
                versus_bce=vs_bce,versus_linear=vs_linear,test_by_language=languages,
                datasets=datasets,positions=len(rows),roots=len({r['group'] for r in rows}),
                fit_sha256=sha(root/'artifacts/e23-fit.json'),
                online='ELIGIBLE_FOR_SEPARATE_ONLINE_AB' if gate else 'NOT_ELIGIBLE')
        check_seal(root)
        result['finished_unix']=time.time();write_new(path,result)
        return dict(status=result['status'],artifact=path.name,
                    **{k:result[k] for k in ('eligible_fresh_test','gate_pass') if k in result},
                    target_object_id=id(session.weights),draft_object_id=id(session.draft))
    finally:
        session.torch.set_num_threads(threads)
