"""Independent CPU MLP stages; E17 inputs and outputs remain immutable."""
import hashlib
import importlib.util
import io
import json
import statistics
import time
import unittest

from e19_run_contract import CONTRACT, sha, verify_seal, write_new
from e17_run_contract import CONTRACT as E17_CONTRACT
from e17_train import (load_collection, load_costs, expand, replay_positions,
                       prior_comparison)
from spark_infer_q.confidence_head import choose_rows, headroom_capture
from spark_infer_q.local_policy import ProgressModel
from spark_infer_q.mlp_head import MLPProgressModel, fit_mlp
from spark_infer_q.budget_regret import budget_regret, regret_weight
from e15_metrics import grouped_metrics


def load_training(root, e17):
    """Reconstruct E17 rows, weights and order; never open new test collections."""
    _, overrides, cost_files = load_costs(root)
    old, old_files = [], {}
    for index in range(12):
        meta = json.loads((root/f'artifacts/e16-collection-{index:02d}.json').read_text())
        path = root/'artifacts'/meta['dataset']
        assert sha(path) == meta['dataset_sha256']
        old_files[path.name] = sha(path)
        old.extend(r for r in map(json.loads, path.read_text().splitlines()) if r['split'] == 'train')
    assert len(old) == 300
    fresh, new_files = load_collection(root, 'learning')
    rows = expand(fresh, 'external')
    train = expand(old, 'e16_train')+[r for r in rows if r['split'] == 'train']
    calibration = [r for r in rows if r['split'] == 'calibration']
    files = {**old_files, **new_files, **cost_files}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    assert digest == e17['dataset_sha256'] and files == e17['datasets']
    assert len(train) == e17['training_rows'] and len(calibration) == e17['calibration_rows']
    assert not {r['group'] for r in train} & {r['group'] for r in calibration}
    prior_path = root/'artifacts/e16-selected-model.json'
    assert sha(prior_path) == e17['prior_sha256']
    prior = json.loads(prior_path.read_text())
    old_model = ProgressModel(prior['metadata'])
    for row in train:
        if row['origin'] == 'e16_train' and (row['task'], row['position']) in overrides:
            row['weight'] = overrides[row['task'], row['position']]
        else:
            p = replay_positions([row], old_model, 'f1', e17['costs'], prior['cpu_predict_ms'])[0]
            width = choose_rows(p['rates'], p['costs_ms'], p['max_rows'])
            regret = budget_regret(row['emitted'], p['costs_ms'], width,
                                   E17_CONTRACT['reference_rate'], row['max_rows'])
            row['weight'] = regret_weight(regret['regret_ms'], E17_CONTRACT['old_weight_scale_ms'])
    assert min(r['weight'] for r in train) == e17['weight_summary']['min']
    assert max(r['weight'] for r in train) == e17['weight_summary']['max']
    assert abs(statistics.mean(r['weight'] for r in train)-e17['weight_summary']['mean']) < 1e-12
    return train, calibration


def timed_models(models, calibration):
    """Same backend/input/order controls for new and frozen linear heads."""
    rows = calibration[:256]
    assert rows
    names = list(models)
    for model in models.values():
        for row in rows[:32]:
            model.predict(row['x'][:len(model.metadata['mean'])])
    timings = {name: [] for name in names}
    for repetition in range(9):
        order = names[repetition % len(names):]+names[:repetition % len(names)]
        if repetition % 2:
            order.reverse()
        for name in order:
            model = models[name]
            size = len(model.metadata['mean'])
            start = time.perf_counter()
            for row in rows:
                model.predict(row['x'][:size])
            timings[name].append((time.perf_counter()-start)*1000/len(rows))
    return {name: dict(cpu_predict_ms=statistics.median(values), cpu_predict_repeats_ms=values)
            for name, values in timings.items()}


def brier(rows, ps):
    return [statistics.mean((p['rates'][depth]-row['y'][depth])**2 for p, row in zip(ps, rows))
            for depth in range(7)]


def run(session, arguments):
    root, torch = session.root, session.torch
    stage = arguments['stage']
    assert stage in ('smoke', 'fit', 'evaluate')
    verify_seal(root)
    output = root/f'artifacts/e19-{stage}.json'
    assert not output.exists()
    report = dict(status='RUNNING', stage=stage, contract=CONTRACT, started_unix=time.time(),
                  source_seal_sha256=sha(root/'artifacts/e19-source-seal.json'),
                  target_object_id=id(session.weights), draft_object_id=id(session.draft))

    def save():
        tmp = output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        tmp.replace(output)

    save()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(CONTRACT['cpu_threads'])
        if stage == 'smoke':
            suite = unittest.TestSuite()
            for name in ('test_mlp_head', 'test_mlp_pipeline'):
                path = root/f'tests/{name}.py'
                spec = importlib.util.spec_from_file_location('_e19_'+name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
            log = io.StringIO()
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
            report.update(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                          skipped=len(result.skipped), test_log=log.getvalue())
            assert result.wasSuccessful() and not result.skipped, log.getvalue()
            report['status'] = 'PASS_E19_SMOKE'
        else:
            path = root/'artifacts/e17-fit.json'
            e17 = json.loads(path.read_text())
            assert e17['status'] == 'PASS_E17_FIT'
            report['e17_fit_sha256'] = sha(path)
            report['dataset_sha256'] = e17['dataset_sha256']
            if stage == 'fit':
                train, calibration = load_training(root, e17)
                report.update(training_rows=len(train), calibration_rows=len(calibration), models={},
                              costs=e17['costs'], fixed_rows=e17['fixed_rows'])
                models = {}
                for candidate in CONTRACT['candidates']:
                    weighted = candidate.startswith('regret_weighted')
                    started = time.perf_counter()
                    rows = [dict(x=r['x'], y=r['y'], weight=r['weight']) for r in train]
                    cal = [dict(x=r['x'], y=r['y']) for r in calibration]
                    model = fit_mlp(rows, cal, weighted=weighted,
                                    hidden=tuple(CONTRACT['architecture'][1:3]), seed=CONTRACT['seed'],
                                    l2=CONTRACT['l2'], max_iter=CONTRACT['max_iter'])
                    assert model.parameter_count == CONTRACT['parameter_count']
                    models[candidate] = model
                    report['models'][candidate] = dict(feature_group='f1', weighted=weighted,
                        metadata=model.metadata, parameter_count=model.parameter_count,
                        fit_seconds=time.perf_counter()-started)
                    save()
                    print(json.dumps(dict(e19_fitted=candidate, seconds=report['models'][candidate]['fit_seconds'])), flush=True)
                controls = {'linear_'+name: ProgressModel(e17['models'][name]['metadata']) for name in CONTRACT['candidates']}
                controls['e17_selected'] = ProgressModel(e17['models'][e17['selected_model']]['metadata'])
                timing = timed_models({**models, **controls}, calibration)
                for candidate, model in models.items():
                    entry = report['models'][candidate]
                    entry.update(timing[candidate])
                    ps = replay_positions(calibration, model, 'f1', e17['costs'], entry['cpu_predict_ms'])
                    entry['calibration_point'] = headroom_capture(ps, ps[0]['costs_ms'], e17['fixed_rows'])
                    entry['calibration_brier'] = brier(calibration, ps)
                selected = max(CONTRACT['candidates'], key=lambda n: report['models'][n]['calibration_point']['policy'])
                report.update(selected_model=selected, control_timings={n: timing[n] for n in controls},
                              e17_selected_candidate=e17['selected_model'], selection=CONTRACT['selection'])
                for candidate, entry in report['models'].items():
                    artifact = dict(candidate=candidate, feature_group='f1', architecture='mlp',
                                    metadata=entry['metadata'], cpu_predict_ms=entry['cpu_predict_ms'],
                                    costs=e17['costs'], fixed_rows=e17['fixed_rows'], contract=CONTRACT,
                                    dataset_sha256=e17['dataset_sha256'], e17_fit_sha256=sha(path), experimental=True)
                    model_path = root/f'artifacts/e19-{candidate}-model.json'
                    write_new(model_path, artifact)
                    entry['artifact_sha256'] = sha(model_path)
                    if candidate == selected:
                        selected_path = root/'artifacts/e19-selected-model.json'
                        write_new(selected_path, artifact)
                        report['selected_model_sha256'] = sha(selected_path)
                report['status'] = 'PASS_E19_FIT'
                # No new test collection/evaluation file has been opened by this stage.
            else:
                fit_path = root/'artifacts/e19-fit.json'
                fit = json.loads(fit_path.read_text())
                assert fit['status'] == 'PASS_E19_FIT'
                assert fit['source_seal_sha256'] == report['source_seal_sha256']
                assert fit['e17_fit_sha256'] == sha(path)
                artifact = root/'artifacts/e19-selected-model.json'
                assert sha(artifact) == fit['selected_model_sha256']
                records, files = load_collection(root, 'test')
                rows = expand(records, 'external_test')
                assert len(records) == 90
                selected = fit['selected_model']
                replay = {}
                report.update(fit_sha256=sha(fit_path), selected_model=selected,
                              selected_model_sha256=sha(artifact), test_datasets=files,
                              test_rows=len(rows), test_groups=len({r['group'] for r in rows}), models={})
                for candidate, entry in fit['models'].items():
                    mp = root/f'artifacts/e19-{candidate}-model.json'
                    assert sha(mp) == entry['artifact_sha256']
                    model = MLPProgressModel(json.loads(mp.read_text())['metadata'])
                    ps = replay_positions(rows, model, 'f1', fit['costs'], entry['cpu_predict_ms'])
                    replay[candidate] = ps
                    linear = e17['models'][candidate]
                    ref = replay_positions(rows, ProgressModel(linear['metadata']), 'f1', fit['costs'],
                                           fit['control_timings']['linear_'+candidate]['cpu_predict_ms'])
                    report['models'][candidate] = dict(point=headroom_capture(ps, ps[0]['costs_ms'], fit['fixed_rows']),
                        cumulative_brier=brier(rows, ps), versus_matched_linear=prior_comparison(ps, ref, CONTRACT['bootstrap_samples']))
                ps = replay[selected]
                point = grouped_metrics(ps, fit['fixed_rows'], CONTRACT['bootstrap_samples'])
                languages = {lang: grouped_metrics([p for p, r in zip(ps, rows) if r['language'] == lang],
                             fit['fixed_rows'], CONTRACT['bootstrap_samples']) for lang in ('en', 'zh', 'mixed')}
                control = e17['models'][e17['selected_model']]
                ref = replay_positions(rows, ProgressModel(control['metadata']), control['feature_group'], fit['costs'],
                                       fit['control_timings']['e17_selected']['cpu_predict_ms'])
                vs_selected = prior_comparison(ps, ref, CONTRACT['bootstrap_samples'])
                matched = report['models'][selected]['versus_matched_linear']
                passed = (point['captured'] is not None and point['captured'] >= CONTRACT['gate_capture']
                          and point['group_bootstrap_gain_95'][0] > 0
                          and point['group_bootstrap_capture_95'] is not None
                          and point['group_bootstrap_capture_95'][0] > 0 and point['different_from_fixed'] > 0
                          and all(languages[n]['group_bootstrap_gain_95'][0] > 0 for n in ('en', 'zh'))
                          and matched['group_bootstrap_gain_95'][0] > 0
                          and vs_selected['group_bootstrap_gain_95'][0] > 0)
                report.update(test=point, test_by_language=languages, versus_e17_selected=vs_selected,
                              gate_pass=bool(passed), status='PASS_E19_EVALUATION',
                              online_decision='ELIGIBLE_FOR_ONLINE' if passed else 'STOP_NO_ONLINE_SPEED_CLAIM')
                assert sha(fit_path) == report['fit_sha256']
            assert sha(path) == report['e17_fit_sha256']
        verify_seal(root)
        report.update(finished_unix=time.time())
        save()
        return dict(status=report['status'], artifact=output.name,
                    **({'gate_pass': report['gate_pass']} if stage == 'evaluate' else {}),
                    target_object_id=id(session.weights), draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}', finished_unix=time.time())
        save()
        raise
    finally:
        torch.set_num_threads(threads)
