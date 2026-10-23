"""CPU-only Spark timing of frozen heads and an optional scalar runtime.

Does not import Torch, fit weights, call CUDA, or submit a resident worker job.
Runs only after the sealed chains have terminated; writes a separate artifact.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time


def main(root, runtime, output):
    sys.path.insert(0, str(root))
    from spark_infer_q.mlp_head import MLPProgressModel
    from spark_infer_q.local_policy import ProgressModel
    from spark_infer_q.confidence_head import choose_rows
    import numpy as np
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = json.loads((root/'worker-status.json').read_text())
    assert before['status'] == 'READY' and before['load_count'] == 1
    for name, status in [('e17-pipeline.json','PASS_E17_CHAIN'), ('e19-pipeline.json','COMPLETE_OFFLINE_GATE_FAILED')]:
        assert json.loads((root/'artifacts'/name).read_text())['status'] == status
    spec = importlib.util.spec_from_file_location('spark_infer_q._e22_numpy_runtime', runtime)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = root/'artifacts'
    linear = json.loads((a/'e17-selected-model.json').read_text())
    mlp = json.loads((a/'e19-selected-model.json').read_text())
    fit = json.loads((a/'e19-fit.json').read_text())
    assert digest(a/'e19-selected-model.json') == fit['selected_model_sha256']
    assert linear['dataset_sha256'] == mlp['dataset_sha256']
    models = dict(linear=ProgressModel(linear['metadata']),
                  mlp_python=MLPProgressModel(mlp['metadata']),
                  mlp_numpy=module.NumpyMLPProgressModel(mlp['metadata']))
    records = []
    input_sha = {n:digest(a/n) for n in ('e17-selected-model.json','e19-selected-model.json','e19-fit.json')}
    for prefix, count in [('e16-collection',12),('e17-learning',10),('e17-test',2)]:
        for index in range(count):
            meta_path = a/f'{prefix}-{index:02d}.json'
            meta = json.loads(meta_path.read_text())
            path = a/meta['dataset']
            assert digest(path) == meta['dataset_sha256']
            input_sha[path.name] = digest(path)
            records.extend(r for r in map(json.loads,path.read_text().splitlines())
                           if prefix != 'e16-collection' or r['split'] == 'train')
    calibration = [r for rec in records if rec['split']=='calibration' for r in rec['rows']][:256]
    assert len(calibration) == 256
    for model in models.values():
        for row in calibration[:32]:
            model.predict(row['x'])
    names = list(models)
    timings = {name:[] for name in names}
    for rep in range(9):
        order = names[rep%len(names):]+names[:rep%len(names)]
        if rep%2:
            order.reverse()
        for name in order:
            t = time.perf_counter()
            for row in calibration:
                models[name].predict(row['x'])
            timings[name].append((time.perf_counter()-t)*1000/len(calibration))
    medians = {k:statistics.median(v) for k,v in timings.items()}
    max_error = 0.
    by_split = {}
    changed_actions = dict(original_fee=0, numpy_fee=0)
    test = dict(tokens=0, ms=0., linear_tokens=0, linear_ms=0.)
    for rec in records:
        by_split[rec['split']] = by_split.get(rec['split'],0)+len(rec['rows'])
        for row in rec['rows']:
            ref = models['mlp_python'].predict(row['x'])
            fast = models['mlp_numpy'].predict(row['x'])
            max_error = max(max_error,max(abs(x-y) for x,y in zip(ref,fast)))
            assert max_error < 1e-12
            c = linear['costs']['long' if row['context_rows'] >= 768 else 'short']
            paid = row['max_rows']>1
            for label, fee in [('original_fee',mlp['cpu_predict_ms']),('numpy_fee',medians['mlp_numpy'])]:
                curve = [v+(c['native_draft_ms']+c['feature_increment_ms']+fee if paid else 0.) for v in c['verify_commit_ms']]
                w = choose_rows(fast,curve,row['max_rows'])
                changed_actions[label] += w != choose_rows(ref,curve,row['max_rows'])
                if label == 'numpy_fee' and rec['split']=='test':
                    test['tokens'] += row['emitted'][w-1]
                    test['ms'] += curve[w-1]
            if rec['split']=='test':
                curve = [v+(c['native_draft_ms']+c['feature_increment_ms']+medians['linear'] if paid else 0.) for v in c['verify_commit_ms']]
                w = choose_rows(models['linear'].predict(row['x']),curve,row['max_rows'])
                test['linear_tokens'] += row['emitted'][w-1]
                test['linear_ms'] += curve[w-1]
    assert not any(changed_actions.values())
    after = json.loads((root/'worker-status.json').read_text())
    assert after == before
    report = dict(status='PASS_CPU_RUNTIME_PROTOTYPE', cpu_predict_ms=medians,
        repeats_ms=timings, by_split_positions=by_split, max_probability_abs_error=max_error,
        changed_actions=changed_actions, original_mlp_cpu_ms=mlp['cpu_predict_ms'],
        head_only_speedup=medians['mlp_python']/medians['mlp_numpy'],
        diagnostic_test_replay=dict(mlp_tokens_per_ms=test['tokens']/test['ms'],
            linear_tokens_per_ms=test['linear_tokens']/test['linear_ms'],
            relative_gain=(test['tokens']/test['ms'])/(test['linear_tokens']/test['linear_ms'])-1),
        source_sha256={'benchmark':digest(Path(__file__)),'runtime':digest(runtime)},
        input_sha256=input_sha, numpy_version=np.__version__,
        worker_status_sha256=digest(root/'worker-status.json'), worker_unchanged=True,
        contract=dict(training=False, target_calls=0, cuda_calls=0, online_generation=False,
            backend='NumPy float64, one-vector calls including input conversion, validation and calibration',
            initialization_timed=False, timing_inputs='first 256 calibration positions; 9 rotated/reversed passes',
            validity='post-test exploratory runtime check; no new independent promotion evidence',
            interpretation='head-only timing and offline test replay; not measured model decode TPS'))
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('runtime',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    main(args.root,args.runtime,args.output)
