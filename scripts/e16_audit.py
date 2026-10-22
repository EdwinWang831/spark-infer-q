"""Read-only, CPU audit of the finished E16 raw archive and public predictors."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from e16_tasks import TASKS, CONTRACT, selected_positions
from e15_metrics import grouped_metrics
from spark_infer_q.confidence_head import cumulative_labels, headroom_capture
from spark_infer_q.local_policy import ProgressModel


def _only_trailing_newlines(executed, current):
    return executed.rstrip(b'\n')==current.rstrip(b'\n')


def _statements_kept(executed, current):
    """Every top-level statement of the executed file is unchanged in the current one."""
    def statements(source):
        text=source.decode()
        return {ast.get_source_segment(text,node) for node in ast.parse(text).body}
    return statements(executed)<=statements(current)


# Sources changed after the run; their executed bytes are read from history.
LEGACY_SOURCES={'spark_infer_q/draft_context.py':('ee7a725',_only_trailing_newlines),
                'spark_infer_q/draft_fidelity.py':('a3d0534',_statements_kept)}


def audit(root):
    repository=Path(__file__).resolve().parents[1]
    sha=lambda b:hashlib.sha256(b).hexdigest()
    load=lambda n:json.loads((root/n).read_text())
    pipeline=load('e16-pipeline.json');assert pipeline['status']=='PASS_E16_CHAIN'
    training=load('e16-training.json');online=load('e16-online.json');ablation=load('e16-ablation.json')
    stages=[pipeline,training,online,ablation]
    records=[];datasets={}
    for index in range(12):
        c=load(f'e16-collection-{index:02d}.json');stages.append(c)
        assert c['status']=='PASS_E16_COLLECTION_SHARD' and c['contract']==CONTRACT
        raw=(root/c['dataset']).read_bytes();assert sha(raw)==c['dataset_sha256']
        datasets[c['dataset']]=sha(raw);records.extend(json.loads(line) for line in raw.decode().splitlines())
    assert datasets==training['datasets']
    assert sha(json.dumps(datasets,sort_keys=True).encode())==training['dataset_sha256']
    assert [r['id'] for r in records]==[t['id'] for t in TASKS]
    sources={};legacy=[]
    for stage in stages:
        for name,expected in stage['source_sha256'].items():
            local=repository/(name if '/' in name else 'scripts/'+name)
            payload=local.read_bytes()
            if sha(payload)!=expected:
                assert name in LEGACY_SOURCES,name
                revision,compatible=LEGACY_SOURCES[name]
                payload=subprocess.check_output(['git','show',f'{revision}:{name}'],cwd=repository)
                assert compatible(payload,local.read_bytes()),name
                assert sha(payload)==expected,name
                legacy.append(name)
            if name in sources:assert sources[name]==expected,name
            sources[name]=expected
    rows=[]
    for rec,task in zip(records,TASKS):
        assert all(rec[k]==task[k] for k in ('id','group','family','split','language','thinking','output_cap'))
        assert rec['serial_positions']==rec['completion_tokens']-1
        assert rec['selected_positions']==selected_positions(rec['serial_positions'],64)
        assert [r['position'] for r in rec['rows']]==rec['selected_positions']
        for r in rec['rows']:
            pos=r['position'];remaining=task['output_cap']-pos-1
            assert r['remaining']==remaining and r['max_rows']==min(8,remaining)
            assert r['context_rows']==rec['prompt_tokens']+pos
            known=[math.log1p(rec['prompt_tokens']+pos),math.log1p(pos+1),math.log1p(remaining),float(task['thinking'])]
            assert all(math.isclose(a,b,rel_tol=0.,abs_tol=1e-13) for a,b in zip(r['x'][21:25],known))
            assert len(r['x'])==137 and all(math.isfinite(x) for x in r['x'])
            assert r['y']==cumulative_labels(r['emitted'])
            assert all(1<=v<=min(w,remaining) for w,v in enumerate(r['emitted'],1))
            assert r['emitted']==sorted(r['emitted'])
            rows.append(dict(r,group=rec['group'],language=rec['language'],split=rec['split']))
    selected=load('e16-selected-model.json');assert selected['feature_group']==training['selected_model']=='f1'
    for group in ('f0','f1'):
        artifact=load('e16-'+group+'-model.json')
        assert artifact['metadata']==training['models'][group]['metadata']
        assert artifact['dataset_sha256']==training['dataset_sha256']
    assert online['status']=='PASS_E16_ONLINE' and training['gate_pass']
    assert online['training_sha256']==sha((root/'e16-training.json').read_bytes())
    assert online['model_sha256']==sha((root/'e16-selected-model.json').read_bytes())
    old=repository/'results/raw/e15/e15-training.json'
    assert old.exists() and sha(old.read_bytes())==online['prior_sha256']
    model=ProgressModel(selected['metadata']);cpu=selected['cpu_predict_ms'];fixed=selected['fixed_rows']
    replays={}
    for split in ('test','ood'):
        ps=[]
        for r in rows:
            if r['split']!=split:continue
            c=selected['costs']['long' if r['context_rows']>=768 else 'short'];paid=r['max_rows']>1
            base=[v+(c['native_draft_ms'] if paid else 0.) for v in c['verify_commit_ms']]
            extra=c['feature_increment_ms']+cpu if paid else 0.
            ps.append(dict(group=r['group'],emitted=r['emitted'],rates=model.predict(r['x']),max_rows=r['max_rows'],
                           baseline_costs_ms=base,costs_ms=[v+extra for v in base]))
        replay=grouped_metrics(ps,fixed,CONTRACT['bootstrap_samples'])
        expected=training['evaluations'][split]
        def compare(a,b):
            if isinstance(a,dict):
                assert a.keys()==b.keys();[compare(a[k],b[k]) for k in a]
            elif isinstance(a,list):
                assert len(a)==len(b);[compare(x,y) for x,y in zip(a,b)]
            elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
            else:assert a==b,(a,b)
        compare(json.loads(json.dumps(replay)),expected)
        replays[split]=replay
    before=pipeline['worker_before'];after=pipeline['worker_after'];current=load('worker-status.json')
    assert before['pid']==after['pid']==current['pid']==319
    assert before['load_count']==after['load_count']==current['load_count']==1
    assert before['failed_jobs']==after['failed_jobs']==current['failed_jobs']
    assert current['status']==after['status']=='READY'
    assert len(pipeline['jobs'])==15 and all(j['result']['status']=='PASS' for j in pipeline['jobs'])
    ids={(s['target_object_id'],s['draft_object_id']) for s in stages if 'target_object_id' in s}
    assert len(ids)==1
    result=dict(status='PASS_E16_INDEPENDENT_CPU_AUDIT',source_sha256=sources,
                legacy_source=list(set(legacy)),dataset_sha256=training['dataset_sha256'],
                prompts=len(records),positions=len(rows),natural_eos=sum(r['eos'] for r in records),
                split_positions=dict(Counter(r['split'] for r in rows)),reproduced=replays,
                prior_training_sha256=online['prior_sha256'],resident=dict(pid=current['pid'],load_count=1,status='READY',
                    failed_jobs_unchanged=True,same_target_draft_objects=True),
                auditor_sha256=sha(Path(__file__).read_bytes()))
    (root/'e16-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('reproduced','source_sha256')}))
    return result


if __name__=='__main__':audit(Path(sys.argv[1]))
