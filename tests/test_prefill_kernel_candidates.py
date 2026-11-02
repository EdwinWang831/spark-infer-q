"""CPU evidence gates and transform guards; fixtures are not GPU measurements."""
import copy
import unittest

from scripts.e43_build import transform as pipeline
from scripts.e43_accounting import summarize as pipeline_summary
from scripts.e44_accounting import summarize as tile_summary


class PrefillKernelCandidateTests(unittest.TestCase):
    def cases(self,ratio=.94):
        return [dict(name=name,layer=layer,rows=rows,exact=True,timings=[
            dict(regime=regime,mode=mode,repetition=rep,calls=3 if regime=='warm' else 1,
                 ms=1. if mode=='serial' else (ratio if rows>=1024 else 1.))
            for regime in ('warm','evict_128MiB') for rep in range(6) for mode in ('serial','candidate')])
            for name,rows in (('english64',64),('code512',512),('code1024',1024),('mixed1024',1024))
            for layer in (0,23,47)]

    def test_pipeline_transform_changes_only_stage_body_and_symbols(self):
        start='template <int M>\n__device__ __forceinline__ void compute_stage('
        end='template <int M>\n__device__ __forceinline__ void k_loop('
        source='prefix\n'+start+'OLD\n'+end+'SUFFIX nvfp4_expert_kernel void nvfp4_experts_cuda('
        result=pipeline(source,'NEW')
        self.assertEqual(result,'prefix\nNEW\n'+end+'SUFFIX e43_pipeline_kernel void e43_experts_cuda(')
        for changed in (source.replace(start,''),source+start,source.replace(end,'')):
            with self.assertRaises(AssertionError):pipeline(changed,'NEW')

    def test_complete_moe_rejects_missing_cases_state_and_timing_work(self):
        cases=self.cases(.8);self.assertTrue(pipeline_summary(cases)['cost_gate_passed'])
        for mutation in ('missing','exact','calls','duplicate','nan'):
            rows=copy.deepcopy(cases)
            if mutation=='missing':rows.pop()
            elif mutation=='exact':rows[0]['exact']=False
            elif mutation=='calls':rows[0]['timings'][0]['calls']=1
            elif mutation=='duplicate':rows[0]['timings'][0]=rows[0]['timings'][1]
            elif mutation=='nan':rows[0]['timings'][0]['ms']=float('nan')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):pipeline_summary(rows)

    def test_tile_gate_uses_only_eligible_cases_and_preserves_controls(self):
        cases=self.cases()
        for c in cases:
            if c['rows']<1024:
                for row in c['timings']:
                    if row['mode']=='candidate':row['ms']=2.
        result=tile_summary(cases)
        self.assertTrue(result['cost_gate_passed'])
        self.assertAlmostEqual(result['eligible']['time_reduction'],.06)
        self.assertLess(result['regimes']['evict_128MiB']['time_reduction'],0.)
        self.assertEqual(len(result['regimes']['evict_128MiB']['per_case']),12)

    def test_tile_gate_requires_five_percent_and_five_paired_wins(self):
        self.assertFalse(tile_summary(self.cases(.96))['cost_gate_passed'])
        cases=self.cases()
        for c in cases:
            if c['rows']>=1024:
                for row in c['timings']:
                    if row['mode']=='candidate' and row['regime']=='evict_128MiB':
                        row['ms']=.8 if row['repetition']<4 else 1.01
        self.assertFalse(tile_summary(cases)['cost_gate_passed'])
        cases=self.cases();cases[6]['rows']=512
        with self.assertRaises(ValueError):tile_summary(cases)


if __name__=='__main__':unittest.main()
