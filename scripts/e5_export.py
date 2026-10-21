"""Validate and sanitize the bounded native scheduler bridge evidence."""
import argparse
import hashlib
import json
from pathlib import Path


CASES = ['single_eos_and_plan_fences',
    'two_requests_chunked_prompt_and_interleaved_decode','length_cap_prefix_only',
    'cancel_waiting_and_generation_reuse','cancel_prepared_prompt_and_generation_reuse',
    'cancel_prepared_decode_and_generation_reuse','invalid_admission_and_full_slots']


def validate(report):
    assert report['status']=='PASS_BOUNDED_NATIVE_SCHEDULER_BRIDGE'
    assert [c['case'] for c in report['cases']]==CASES
    gpu=report['kind']=='full_target_native_scheduler'
    assert report['kind'] in ('full_target_native_scheduler','cpu_native_scheduler')
    assert report['cuda_initialized'] is gpu
    assert report['core_sha256']=='f6aa14b790eb7bc729251eea152fd9c83cf215bc5ab912e25ee19c57d8fc1768'
    events=rows=states=0
    for case in report['cases']:
        assert case['status']=='PASS'
        assert case['free_native_blocks']==case['total_native_blocks']>0
        assert case['completed']
        positions={}
        for event in case['events']:
            key=(event['slot'],event['generation'])
            assert event['start']==positions.get(key,0), 'state skipped or request reused without a new generation'
            assert event['rows']>0 and event['published'] in (0,1)
            assert not event['cancelled'] or event['published']==0
            positions[key]=event['start']+event['rows']
        if case['case'].startswith('cancel_'):
            old,new=case['completed']
            assert old['cancelled'] and not new['cancelled']
            assert old['slot']==new['slot'] and new['generation']>old['generation']
        if case['case']=='length_cap_prefix_only':
            assert len(case['completed'][0]['token_ids'])==3
        if case['case']=='two_requests_chunked_prompt_and_interleaved_decode':
            decoded=[e['request_id'] for e in case['events'] if not e['prompt']]
            assert len(set(decoded))==2
            assert sum(a!=b for a,b in zip(decoded,decoded[1:]))>=2, 'requests ran in separate complete batches'
        checks=case['backend_checks']
        if gpu:
            if case['case']=='invalid_admission_and_full_slots':
                assert not case['events'] and checks['full_logit_rows']==checks['live_state_checks']==0
            else:
                assert checks['full_logit_rows']>0 and checks['live_state_checks']>0
            assert len(checks['released_states'])==len(case['completed'])
            rows+=checks['full_logit_rows'];states+=checks['live_state_checks']
        else:assert checks['released_states']==len(case['completed'])
        events+=len(case['events'])
    return dict(case_groups=len(report['cases']),scheduled_request_segments=events,
        full_logit_rows=rows,live_state_checks=states,
        completed_or_cancelled_requests=sum(len(c['completed']) for c in report['cases']),
        all_native_logical_blocks_returned=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    reports={name:json.loads((a.directory/(name+'.json')).read_text()) for name in ('cpu','replay','gpu')}
    for report in reports.values():
        for name,sha in report['harness_sha256'].items():
            path=repo/name if name.startswith('spark_infer_q/') else repo/'scripts'/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,name
    assert reports['cpu']['harness_sha256']==reports['replay']['harness_sha256']==reports['gpu']['harness_sha256']
    for replay,gpu in zip(reports['replay']['cases'],reports['gpu']['cases'],strict=True):
        assert replay['completed']==gpu['completed'], 'GPU output/position differs from frozen E4 replay'
        assert replay['events']==gpu['events'], 'GPU dispatch differs from real-token replay'
    result=dict(status='PASS_BOUNDED_SYNCHRONOUS_NATIVE_SCHEDULER_BRIDGE',
        reports=reports,summary={name:validate(report) for name,report in reports.items()},
        replay_vs_gpu_dispatch_and_outputs_equal=True)
    text=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for private in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/'):assert private not in text,private
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
