"""Recompute the sealed draft-Graph service ablation and Target diagnostics."""
import argparse,hashlib,json,statistics
from pathlib import Path
from k4batch_progress_export import summarize

JOBS=dict(component='406-k4batch-draft-graph-component-v4.json',
    service='409-k4batch-draft-graph-service-v3.json',lifecycle='404-k4batch-draft-graph-life-v2.json',
    cpu='398-k4batch-draft-graph-cpu-v2.json',target='407-k4batch-target-breakdown-v2.json',
    audit='410-k4batch-draft-graph-audit-v2.json')


def export(raw):
    jobs={k:json.loads((raw/'jobs/done'/name).read_text()) for k,name in JOBS.items()}
    assert all(j['status']=='PASS' and j['load_count']==1 and not j['result']['model_reloaded'] for j in jobs.values())
    assert len({j['worker_pid'] for j in jobs.values()})==1
    values={k:j['result'] for k,j in jobs.items()}
    for path in raw.glob('k4batch-*/manifest.json'):
        for n,h in json.loads(path.read_text()).items():assert hashlib.sha256((path.parent/n).read_bytes()).hexdigest()==h,(path,n)
    checkout=Path(__file__).resolve().parents[1]
    for key in ('component','service','lifecycle','target'):
        for n,h in values[key]['source_sha256'].items():
            if n.startswith('spark_infer_q/') or n=='THIRD_PARTY_NOTICES.md':
                assert hashlib.sha256((checkout/n).read_bytes()).hexdigest()==h,(key,n)
    service,component,life,target,audit=(values[k] for k in ('service','component','lifecycle','target','audit'))
    assert len(service['timings'])==45 and len(service['correctness'])==9
    assert all(r['outputs_exact'] for r in service['correctness'])
    for group in ('short','cap128','staggered_prefill'):
        arms=[r for r in service['correctness'] if r['group']==group]
        assert len({tuple(r['widths']) for r in arms})==1,('diagnostic policy changed',group)
    assert all(not r['draft_captures'] and r['outputs_exact'] for r in service['timings'])
    assert all(not r['counters'].get('eager',0) and not r['counters'].get('evictions',0) for r in service['timings'])
    assert all(service['correctness_counts'][k]>0 for k in ('logits_rows','feature_rows','state_checks','prefill_checks'))
    assert component['failure_isolation']==dict(old_prefix_exact=True,rows_not_published=True,owner_quarantined=True)
    assert component['graph_storage_closed'] and component['graph_capabilities']['draft_graph_count']==20
    assert len(component['correctness'])==14 and all(r['hidden_exact'] for r in component['correctness'])
    assert len(component['timings'])==63
    assert len(life['cases'])==14 and life['gloo_restored'] and life['environment_restored']
    assert life['performance_cpu'] and life['main_affinity_unchanged']
    assert not audit['live_batch_objects'] and audit['stable_after_second_gc'] and audit['native_group_released']
    assert audit['native_environment_restored'] and not audit['empty_cache_called']
    worker=json.loads((raw/'worker-status.json').read_text())
    assert worker['status']=='READY' and worker['pid']==audit['worker_pid'] and worker['load_count']==1
    for name,digest in (('target-state.so','b63a8c527c56b2aec964fba81352bed9bb16b510db32cfe14b4d3f19ee2b64e2'),
                        ('batch-draft-w8.so',component['kernel']['binary_sha256'])):
        assert hashlib.sha256((raw/'libraries'/name).read_bytes()).hexdigest()==digest,name
    assert len(target['records'])==20 and target['numerical_checks']
    breakdown={}
    for shape in {(tuple(r['shape'])) for r in target['records']}:
        rows=[r for r in target['records'] if tuple(r['shape'])==shape]
        assert len(rows)==5 and all(r['exact'] for r in rows)
        total=statistics.mean(r['model_loop_ms'] for r in rows)
        categories={k:statistics.mean(r['category_ms'][k] for r in rows) for k in rows[0]['category_ms']}
        assert sum(categories.values())<=total+0.01
        breakdown[str(shape)]=dict(mean_model_loop_ms=total,category_mean_ms=categories,
            category_fraction_of_model_loop={k:v/total for k,v in categories.items()},
            unattributed_ms=total-sum(categories.values()),repetitions=5)
    component_summary={case:{mode:statistics.median(r['wall_ms'] for r in component['timings']
        if r['case']==case and r['mode']==mode) for mode in ('eager','buffers','graphs')}
        for case in ('short','long','survivor')}
    summary=summarize(service['timings'],('eager','buffers','graphs'))
    incremental=summarize([r for r in service['timings'] if r['mode']!='eager'],('buffers','graphs'))
    return dict(status='K4_DRAFT_QUERY_GRAPH_OPT_IN',default_promoted=False,max_service_requests=2,
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summary=summary,graphs_vs_buffers=incremental,component_median_ms=component_summary,
        component_contract=component['contract'],component_correctness=component['correctness'],
        component_failure_isolation=component['failure_isolation'],startup_warmup=component['startup_warmup'],
        graph_capabilities=component['graph_capabilities'],service_contract=service['contract'],
        target_capabilities=service['capabilities'],
        correctness_counts=service['correctness_counts'],correctness=service['correctness'],timings=service['timings'],
        warmups=service.get('warmups',[]),
        late_shape_warmups=service.get('late_shape_warmups',[]),component_timings=component['timings'],
        target_breakdown=breakdown,target_contract=target['contract'],target_records=target['records'],
        lifecycle=life['cases'],cpu_tests=values['cpu']['output'],resident=audit,jobs=JOBS,
        formal_source_sha256=service['source_sha256'],component_source_sha256=component['source_sha256'],
        target_source_sha256=target['source_sha256'],
        evidence_archive_sha256=hashlib.sha256((raw/'evidence.tar.gz').read_bytes()).hexdigest(),
        exclusions=['392/394/399 component probes are earlier distinct runs, never pooled',
            '393 CPU runner rejected a successful test because its name contained skipped; 398 reruns both suites',
            '396 service is an earlier snapshot, not pooled with 409',
            '403 rejected a formal cohort with a new Target capture; no successful final timing retained',
            '395 Target diagnostic and 397 lifecycle are earlier snapshots, not pooled',
            'Prior batch draft differences from original sequential drafting remain disclosed',
            'No single-stream or four-request full service result',
            'No joint budget cost recalibration; frozen E17/E29 remain controls',
            'Target categories are instrumented Graph stream intervals, not summed active kernel times'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=export(args.raw)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],summary=result['summary'],target=result['target_breakdown']),indent=2))


if __name__=='__main__':main()
