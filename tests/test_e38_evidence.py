import copy
import unittest

from scripts.e38_evidence import component,geometry,LENGTHS
from scripts.e38_accounting import summarize
from tests.test_e38_accounting import fixture as cost_fixture
from tests.test_e36_candidate_evidence import fixture as seed_fixture


def fixture():
    seed=seed_fixture();inputs=[next(c for c in seed['candidate_inputs'] if c['name']==n) for n in LENGTHS]
    cases=[];summary=dict(layers=192,tiles16=0,tiles32=0,paired32_items=0,routed_pairs=0)
    for name,n in LENGTHS.items():
        a=((n+15)//16)*10;b=((n+31)//32)*10;p=(n//32+int(n%32>16))*10
        layer=dict(routed_pairs=n*10,active_experts=10,rows_per_expert_histogram={str(n):10},
            tiles16=a,tiles32=b,capacity16=a*16,capacity32=b*32,experts_above16=10,
            paired32_items=p,gate_up_ms=1.,down_ms=1.)
        cases.append(dict(name=name,length=n,output_sha256='a'*64,logits_features_exact=True,
                          layers=[dict(layer,layer=i) for i in range(48)]))
        for k,v in [('tiles16',a),('tiles32',b),('paired32_items',p),('routed_pairs',n*10)]:summary[k]+=v*48
    summary['potential_weight_stage_load_reduction']=1-summary['tiles32']/summary['tiles16']
    return dict(status='PASS_E38_GEOMETRY',upstream_functions_unchanged=True,inputs=inputs,cases=cases,summary=summary),seed


class GeometryEvidenceTests(unittest.TestCase):
    def test_histogram_reconstructs_counts_without_speed_claim(self):
        data,seed=fixture();out=geometry(data,seed)
        self.assertEqual(out['routed_pairs'],583680)
        self.assertEqual(out['layers'],192)
        self.assertNotIn('prompt_tps',out)

    def test_incomplete_or_inconsistent_geometry_cannot_publish(self):
        for change in ('layer','hash','hist','pairs','tiles','summary','nan'):
            with self.subTest(change=change):
                d,s=fixture();layer=d['cases'][0]['layers'][0]
                if change=='layer':d['cases'][0]['layers'].pop()
                elif change=='hash':d['cases'][0]['output_sha256']='b'*64
                elif change=='hist':layer['rows_per_expert_histogram']={'513':10}
                elif change=='pairs':layer['routed_pairs']+=1
                elif change=='tiles':layer['tiles32']+=1
                elif change=='summary':d['summary']['tiles16']+=1
                else:layer['gate_up_ms']=float('nan')
                with self.assertRaises(ValueError):geometry(d,s)


def component_fixture():
    seed=seed_fixture();cases=cost_fixture()
    for c in cases:c['rows']=LENGTHS[c['name']]
    checks=[dict(rows=n,strided=s,dtype=d,exact=True,guards_exact=True,graph_exact=True)
        for n in (1,15,16,17,31,32,33,64,128,512) for s in (False,True)
        for d in ('torch.float32','torch.bfloat16')]
    result=dict(status='PASS_E38_COMPONENT',tf32_disabled=True,upstream_functions_unchanged=True,
        separate_caller_stream_exact=True,default_promoted=False,timing_scope='whole_moe_graph_cuda_event',
        candidate_min_rows=256,preflight=checks,cases=cases,summary=summarize(cases),
        diagnostic_prompts=[dict(name=n,output_sha256='a'*64,exact=True) for n in LENGTHS])
    return result,seed


class ComponentEvidenceTests(unittest.TestCase):
    def test_complete_gates_allow_component_cost_accounting(self):
        d,s=component_fixture()
        self.assertEqual(component(d,s,summarize),d['summary'])

    def test_cost_measurements_require_full_preflight_and_numerical_coverage(self):
        for change in ('guard','missing','stream','rows','hash','summary'):
            with self.subTest(change=change):
                d,s=component_fixture()
                if change=='guard':d['preflight'][0]['guards_exact']=False
                elif change=='missing':d['preflight'].pop()
                elif change=='stream':d['separate_caller_stream_exact']=False
                elif change=='rows':d['cases'][0]['rows']-=1
                elif change=='hash':d['diagnostic_prompts'][0]['output_sha256']='c'*64
                else:d['summary']['cost_gate_passed']=False
                with self.assertRaises(ValueError):component(d,s,summarize)


if __name__=='__main__':unittest.main()
