"""Validate Mia tile component, complete-model/request and public-hook evidence."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prefill_kernel_evidence import bundle,clean,require,sha
from scripts.e44_accounting import summarize
from scripts.e42_export import request_summary as paired_requests
from scripts.e44_build import MIA_SHA


def numeric_summary(study):
    expected={'english64':64,'code512':512,'code513':513,'mixed1024':1024,'chinese1536':1536}
    rows=study['numeric'];require(len(rows)==5 and {r['name'] for r in rows}==set(expected),'numeric coverage differs')
    seen=set()
    for r in rows:
        n=expected[r['name']];require(r['length']==n and r['chunks']==[1024]*(n//1024)+([n%1024] if n%1024 else []),'numeric input/chunks differ')
        require(r['logits_exact'] is True and r['features_exact'] is True and not r['state_differences'],'full-model numeric gate failed')
        require(len(r['windows'])==8,'handoff coverage differs')
        for width,w in enumerate(r['windows'],1):
            require(w['width']==width and w['keep']==(width if width%2 else width//2) and w['exact'] is True,'partial commit gate failed')
            seen.add((width,w['parity']))
    require(seen=={(r,p) for r in range(1,9) for p in (0,1)},'Graph parity coverage differs')
    graph=study['numeric_graph'];require(graph['graph_captures']==16 and graph['graph_replays']==40 and not graph['graph_fallbacks'],'numeric Graph fell back')
    return dict(prefill_cases=5,feature_rows=sum(expected.values()),full_logits_rows=185,verify_windows=40,buckets=[list(x) for x in sorted(seen)])


def prompt_summary(rows):
    require(len(rows)==24,'prompt timing coverage differs');result={}
    for name,n in (('english64',64),('code1024',1024),('mixed1536',1536)):
        group=[r for r in rows if r['name']==name]
        require(len(group)==8 and {(r['mode'],r['repetition']) for r in group}=={(m,i) for m in ('serial','candidate') for i in range(4)},'prompt pairs differ')
        require(all(r['exact'] is True and r['prompt_tokens']==n and type(r['wall_ms']) in (int,float) and math.isfinite(r['wall_ms']) and r['wall_ms']>0 for r in group),'invalid prompt numeric/timing/work')
        costs={m:sum(r['wall_ms'] for r in group if r['mode']==m) for m in ('serial','candidate')}
        result[name]=dict(prompt_tokens=n,cost_ms=costs,reduction=1-costs['candidate']/costs['serial'])
    longs={m:sum(result[n]['cost_ms'][m] for n in ('code1024','mixed1536')) for m in ('serial','candidate')}
    return dict(per_case=result,long_cost_ms=longs,long_reduction=1-longs['candidate']/longs['serial'],timed_prompts=24)


def check_http_lifecycle(value):
    require(len(value['lifecycle'])==3 and {r['case'] for r in value['lifecycle']}=={n+'_cancel_and_reuse' for n in ('waiting','prepared_prompt','prepared_decode')} and all(r['exact'] is True for r in value['lifecycle']),'cancellation/reuse failed')
    require(len(value['http'])==2 and all(r['exact'] is True for r in value['http']) and
        {(r['case'],r['output_tokens'],r['finish_reason']) for r in value['http']}=={('stream',24,'length'),('cancel',1,'cancelled')},'HTTP functional coverage differs')


def export(raw,build_version,build_job,component_version,component_job,study_version,study_job,integration_version,integration_job,audit_version,audit_job):
    raw=Path(raw)
    value=bundle(raw,'e44',build_version,build_job,component_version,component_job,audit_version,audit_job,
        study=(study_version,study_job),integration=(integration_version,integration_job))
    reports=value['reports'];built,component,study,integration=(reports[k] for k in ('build','component','study','integration'))
    summary=summarize(component['cases']);require(summary==component['summary'] and summary['cost_gate_passed'] is True,'component cost gate failed')
    require(built['mia_source_sha256']==MIA_SHA and sha(raw/'stages'/build_version/'results/raw/e44/reference/fn_experts_tile.cu')==MIA_SHA,'Mia source differs')
    require(built['policy']==dict(tile=64,gate_up='rt4_s4',down='rt2_s3',minimum_rows=1024),'Mia policy differs')
    require((study['physical_capacity'],study['logical_capacity'],study['token_budget'],study['chunk_rows'],study['output_cap'])==(2048,1792,1024,1024,64),'request configuration differs')
    numeric=numeric_summary(study);prompts=prompt_summary(study['timings'])
    # Reuse strict paired request accounting with explicit labels only.
    rows=json.loads(json.dumps(study['requests']))
    for r in rows:r['mode']={'serial':'chunk512','candidate':'chunk1024'}[r['mode']]
    requests=paired_requests(rows)
    for c in requests['per_case'].values():c['arms']={('serial' if k=='chunk512' else 'candidate'):v for k,v in c['arms'].items()}
    requests['total']={('serial' if k=='chunk512' else 'candidate'):v for k,v in requests['total'].items()}
    requests['comparison']='same chunk1024; only original versus Mia expert backend differs'
    for name in ('en','zh','code1024','mixed1536'):
        group=[r for r in study['requests'] if r['name']==name]
        require(all(r['trace']==group[0]['trace'] for r in group),'full prompt/decode trace differs')
    check_http_lifecycle(study);check_http_lifecycle(integration)
    require(integration['timing_claim'] is False and integration['candidate_calls']>0,'final hook not exercised')
    require(len(integration['numeric'])==2 and {r['name'] for r in integration['numeric']}=={'code1024','mixed1536'},'integrated prefix coverage differs')
    for r in integration['numeric']:
        require(all(r[k] is True for k in ('full_logits_exact','features_exact','state_exact')) and len(r['windows'])==8,'integrated numeric gate failed')
        require([w['width'] for w in r['windows']]==list(range(1,9)) and all(w['exact'] is True and w['keep']==(w['width'] if w['width']%2 else w['width']//2) for w in r['windows']),'integrated handoff failed')
    require(integration['numeric_graph']['graph_replays']==16 and not integration['numeric_graph']['graph_fallbacks'],'integrated Graph coverage differs')
    require(len(integration['requests'])==2 and {r['mode'] for r in integration['requests']}=={'serial','candidate'} and
        all(r['exact'] is True and r['prompt_tokens']==1024 and 1<=r['output_tokens']<=64 for r in integration['requests']) and
        len({r['output_tokens'] for r in integration['requests']})==1,'integrated request work differs')
    for key,name in (('model_sha256','e27-linear-model.json'),('cost_sha256','e29-frozen-costs.json'),('w8_source_sha256','draft_w8_kernel.cu'),('w8_binary_sha256','draft_w8.so')):
        require(sha(raw/'frozen'/name)==study['profile'][key],'optimized profile changed')
    root=Path(__file__).resolve().parents[1]
    for name,digest in value['stage_sources']['integration'].items():
        if name.startswith('spark_infer_q/'):require(sha(root/name)==digest,'current integrated library differs: '+name)
    value.update(summary=summary,numeric=numeric,prompts=prompts,requests=requests,HTTP_functional_only=True,
        public_hook_validated=True,expert_backend_opt_in=True,exporter_sha256=sha(__file__),
        evidence_helper_sha256=sha(Path(__file__).with_name('prefill_kernel_evidence.py')))
    return clean(value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    keys=('build-version','build-job','component-version','component-job','study-version','study-job','integration-version','integration-job','audit-version','audit-job')
    for k in keys:p.add_argument('--'+k,required=True)
    a=p.parse_args();v=export(a.raw,*[getattr(a,k.replace('-','_')) for k in keys])
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v[k] for k in ('status','numeric','prompts','requests','residency')},indent=2))
