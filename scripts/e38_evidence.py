"""Recompute geometry and numerical gates without a GPU."""
import math

NAMES=('code','english','mixed','new_json')
LENGTHS=dict(code=512,english=64,mixed=512,new_json=128)


def require(ok,message):
    if not ok:raise ValueError(message)


def geometry(result,seed):
    require(result.get('status')=='PASS_E38_GEOMETRY','geometry failed')
    require(result.get('upstream_functions_unchanged') is True,'global source modified')
    inputs=[next(c for c in seed['candidate_inputs'] if c['name']==n) for n in NAMES]
    require(result['inputs']==inputs,'frozen input identity differs')
    cases=result['cases'];require(len(cases)==4 and [c['name'] for c in cases]==list(NAMES),'missing/repeated geometry input')
    total=dict(layers=0,tiles16=0,tiles32=0,paired32_items=0,routed_pairs=0)
    for case in cases:
        n=LENGTHS[case['name']]
        require(case['length']==n and case['logits_features_exact'] is True,'geometry output gate failed')
        hashes={r['output_sha256'] for r in seed['timings'] if r['name']==case['name']}
        require(len(hashes)==1 and case['output_sha256'] in hashes,'E36 output hash differs')
        layers=case['layers'];require(len(layers)==48 and [l['layer'] for l in layers]==list(range(48)),'layer coverage differs')
        for layer in layers:
            hist=layer['rows_per_expert_histogram']
            require(hist and all(str(int(k))==k and 1<=int(k)<=n and type(v) is int and v>0 for k,v in hist.items()),'invalid expert occupancy')
            active=sum(hist.values());pairs=sum(int(k)*v for k,v in hist.items())
            a=sum(((int(k)+15)//16)*v for k,v in hist.items())
            b=sum(((int(k)+31)//32)*v for k,v in hist.items())
            paired=sum((int(k)//32+int(int(k)%32>16))*v for k,v in hist.items())
            require(active<=512 and pairs==n*10 and layer['routed_pairs']==pairs,'routed pair count differs')
            expect=dict(active_experts=active,tiles16=a,tiles32=b,capacity16=a*16,capacity32=b*32,
                paired32_items=paired,experts_above16=sum(v for k,v in hist.items() if int(k)>16))
            require(all(layer[k]==v for k,v in expect.items()),'tile arithmetic differs')
            for key in ('gate_up_ms','down_ms'):
                require(type(layer[key]) in (int,float) and math.isfinite(layer[key]) and layer[key]>0,'invalid kernel diagnostic')
            total['layers']+=1;total['tiles16']+=a;total['tiles32']+=b;total['paired32_items']+=paired;total['routed_pairs']+=pairs
    total['potential_weight_stage_load_reduction']=1-total['tiles32']/total['tiles16']
    require(total==result['summary'],'geometry summary differs')
    return total


def component(result,seed,summarize):
    require(result.get('status')=='PASS_E38_COMPONENT','component failed')
    for key in ('tf32_disabled','upstream_functions_unchanged','separate_caller_stream_exact'):
        require(result.get(key) is True,key+' gate failed')
    require(result.get('default_promoted') is False and result.get('timing_scope')=='whole_moe_graph_cuda_event'
            and result.get('candidate_min_rows')==256,'component scope differs')
    expected={(n,s,d) for n in (1,15,16,17,31,32,33,64,128,512) for s in (False,True)
              for d in ('torch.float32','torch.bfloat16')}
    checks=result['preflight']
    require(len(checks)==40 and {(c['rows'],c['strided'],c['dtype']) for c in checks}==expected,'preflight coverage differs')
    require(all(c.get(k) is True for c in checks for k in ('exact','guards_exact','graph_exact')),'preflight numerical gate failed')
    prompts=result['diagnostic_prompts']
    require(len(prompts)==4 and [p['name'] for p in prompts]==list(NAMES),'diagnostic prompt coverage differs')
    for p in prompts:
        hashes={r['output_sha256'] for r in seed['timings'] if r['name']==p['name']}
        require(p.get('exact') is True and len(hashes)==1 and p['output_sha256'] in hashes,'diagnostic output differs')
    require(all(c['rows']==LENGTHS[c['name']] for c in result['cases']),'real component input length differs')
    summary=summarize(result['cases']);require(summary==result['summary'],'component summary differs')
    return summary
