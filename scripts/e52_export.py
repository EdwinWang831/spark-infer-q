"""Validate complete-MoE timing, exactness, fixed gates and resident provenance."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import require,sha,public
from scripts.prefill_kernel_evidence import read_stage
from scripts.e52_component import summary

PHASES=(('first','component-v1','588-e52-component-v1.json'),('first_audit','audit-v1','590-e52-audit-v1.json'),
        ('component','component-v2','592-e52-component-v2.json'),('audit','audit-v2','594-e52-audit-v2.json'))


def check_component(r):
    require(r['status']=='PASS_E52_COMPONENT' and r['full_decode_measured'] is False and r['default_promoted'] is False,'scope/status differs')
    expected={(rows,strided,dtype,pattern) for rows in range(1,9) for strided in (False,True)
              for dtype in ('torch.float32','torch.bfloat16') for pattern in ('same','distinct','mixed')}
    require(len(r['preflight'])==96 and {(x['rows'],x['strided'],x['dtype'],x['pattern']) for x in r['preflight']}==expected,'preflight coverage differs')
    require(r['separate_stream_exact'] and all(x['exact'] and x['guards_graph_exact'] for x in r['preflight']),'preflight gate failed')
    names={'english64':64,'chinese768':768,'code1024':1024,'mixed1536':1536}
    require(len(r['inputs'])==4 and {x['name']:x['prompt_tokens'] for x in r['inputs']}==names,'prompt scope differs')
    require(len(r['trajectory'])==8 and {(x['name'],x['step']) for x in r['trajectory']}=={(n,s) for n in names for s in (0,1)},'trajectory coverage differs')
    trajectory={(x['name'],x['step']):x for x in r['trajectory']}
    require(all(x['logits_features_state_exact'] and 1<=x['keep']<=x['width']<=8 for x in r['trajectory']),'diagnostic equality failed')
    cases=r['cases'];require(len(cases)==24 and {(x['name'],x['step'],x['layer']) for x in cases}=={(n,s,l) for n in names for s in (0,1) for l in (0,23,47)},'real MoE coverage differs')
    intervals=0;calls=0
    for c in cases:
        require(c['exact'] and not c['plan_prefill'] and c['plan_tile']==16 and c['rows']==trajectory[c['name'],c['step']]['width'],'decode plan/width differs')
        require(all(1<=n<=c['rows'] for n in c['routed_expert_member_counts']),'expert membership invalid')
        require(sum(c['routed_expert_member_counts'])==10*c['rows'],'routed pair work differs')
        times=c['timings'];require(len(times)==24,'timing coverage differs')
        expected=[]
        for regime in ('warm','evict_128MiB'):
            for rep in range(6):
                for mode in (('original','pipeline') if (rep+c['layer'])%2==0 else ('pipeline','original')):
                    expected.append((regime,rep,mode))
        require([(t['regime'],t['repetition'],t['mode']) for t in times]==expected,'AB/BA order differs')
        for t in times:
            require(t['calls']==(5 if t['regime']=='warm' else 1) and 0<t['ms']<100,'timing/calls invalid')
            intervals+=1;calls+=t['calls']
    recalculated=summary(cases);require(recalculated==r['summary'],'summary arithmetic differs')
    gate=all(v['time_reduction']>=.05 and v['root_bootstrap_gain_95'][0]>0 and min(p['speed_change'] for p in v['per_root'])>=-.02 for v in recalculated.values())
    require(gate==r['candidate_gate_passed'] and r['decision']==('ADVANCE_TO_FULL_DECODE' if gate else 'REJECT_DECODE_PIPELINE'),'frozen decision differs')
    return dict(summary=recalculated,candidate_gate_passed=gate,decision=r['decision'],preflight=96,real_cases=24,
        diagnostic_windows=8,cuda_event_intervals=intervals,timed_complete_moe_calls=calls,
        real_widths=sorted({c['rows'] for c in cases}),maximum_expert_members=max(n for c in cases for n in c['routed_expert_member_counts']),
        all_selected_moe_and_diagnostic_outputs_exact=True,full_candidate_generation_validated=False)


def export(raw):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text());reports={};manifests={};records={}
    require(worker['status']=='READY' and worker['pid']==46235 and worker['load_count']==1,'resident status differs')
    for label,version,job in PHASES:reports[label],manifests[label],records[label]=read_stage(raw,'e52',version,job,worker)
    base=reports['component'];out=check_component(base);first=check_component(reports['first'])
    for label,r in reports.items():
        require(r['target_object_id']==base['target_object_id'] and r['draft_object_id']==base['draft_object_id'] and
            r['baseline_bytes']==77393408000 and not r['model_reloaded'],'resident identity/baseline differs')
        for name,digest in r['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    for studied,audited in (('first','first_audit'),('component','audit')):
        a=reports[audited]
        require(a['status']=='PASS_E52_AUDIT' and a['audited_job_sha256']==records[studied]['sha256'] and
            a['audited_status']==reports[studied]['status'] and a['allocated_bytes']==77393408000,'postreturn audit differs')
        require(all(a[x] for x in ('original_operators_unchanged','same_resident_objects','owned_modules_released')),'audit flag failed')
    buildjob=raw/'references/430-e43-build-v1.json';built=json.loads(buildjob.read_text())['result']
    require(json.loads(buildjob.read_text())['status']=='PASS' and built['target_object_id']==base['target_object_id'],'existing binary identity differs')
    build=raw/'references/e43-build'
    require(sha(build/'build/binary/spark_q_e43_pipeline_v1.so')==built['binary_sha256']==base['binary_sha256'] and
        base['build_job_sha256']==sha(buildjob),'existing binary chain differs')
    for name in ('scripts/e43_build.py','scripts/e43_compute.cuh','scripts/e43_bindings.cpp'):
        require(manifests['component'][name]==built['source_sha256'][name],'pipeline build input differs')
    require(sha(build/'build/cuda/nvfp4/experts.cu')==built['generated_sha256'],'generated pipeline changed')
    for label in ('first','component'):
        for name,digest in manifests[label].items():
            if name.startswith('spark_infer_q/'):require(manifests['audit'][name]==digest,'executed library changed')
    require(worker['allocated_bytes']==77393408000 and not out['candidate_gate_passed'],'final residency or rejection differs')
    meta=json.loads((raw.parent/'archive-metadata.json').read_text());require(sha(raw.parent/'evidence.tar.gz')==meta['sha256'],'archive changed')
    return public(dict(status='PASS_E52_EVIDENCE_REJECTED_CANDIDATE',date='2026-10-26',**out,
        initial_tail_guard_run=first,provenance=records,archive=meta,
        resident=dict(pid=worker['pid'],load_count=1,allocated_bytes=worker['allocated_bytes']),
        candidate_binary_sha256=base['binary_sha256'],baseline_binary_sha256=base['baseline_binary_sha256'],
        final_library_sha256={k:v for k,v in manifests['component'].items() if k.startswith('spark_infer_q/')},
        default_promoted=False,full_decode_measured=False,
        scope='real short-context dynamic verification; complete MoE component, not full model candidate generation or256K/HTTP throughput'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    value=export(a.raw);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(value['status'])
