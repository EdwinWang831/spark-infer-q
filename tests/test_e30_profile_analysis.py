import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from e30_analyze import analyze_trace,union_us


class ChromeKernelAccountingTests(unittest.TestCase):
    def test_annotation_parents_and_outside_work_are_not_double_counted(self):
        trace=dict(traceEvents=[
            dict(cat='user_annotation',ph='X',name='e30.verify',ts=10.,dur=10.),
            dict(cat='gpu_user_annotation',ph='X',name='e30.verify',ts=11.,dur=8.),
            dict(cat='kernel',ph='X',name='_b16mm',ts=12.,dur=3.,args=dict(grid=[1,3880,1])),
            dict(cat='kernel',ph='X',name='qmmf_kernel',ts=16.,dur=2.),
            dict(cat='kernel',ph='X',name='draft',ts=1.,dur=4.),
            dict(cat='gpu_memcpy',ph='X',name='copy',ts=18.,dur=1.)])
        r=analyze_trace(trace)
        self.assertEqual(r['kernel_calls'],2);self.assertEqual(r['kernel_sum_us'],5.)
        self.assertEqual(r['kernel_union_us'],5.);self.assertEqual(r['activity_union_us'],6.)
        self.assertEqual(r['verify_wall_us'],10.);self.assertEqual(r['bf16_grids'][0]['calls'],1)

    def test_union_preserves_overlap_and_missing_range_rejected(self):
        self.assertEqual(union_us([(2.,6.),(4.,8.),(10.,12.)]),8.)
        with self.assertRaises(ValueError):union_us([(2.,1.)])
        with self.assertRaises(ValueError):analyze_trace(dict(traceEvents=[]))


if __name__=='__main__':unittest.main()
