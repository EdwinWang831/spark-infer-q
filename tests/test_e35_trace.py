import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e35_trace import intersection_us,analyze


class MoEOverlapTraceTests(unittest.TestCase):
    def test_overlap_does_not_double_count_in_family(self):
        self.assertEqual(intersection_us([(0,10),(5,15)],[(7,12),(8,11)]),5)
        self.assertEqual(intersection_us([(0,3)],[(3,8)]),0)

    def test_parent_annotations_and_outside_kernels_excluded(self):
        def event(name,ts,dur,stream,cat='kernel'):
            return dict(name=name,cat=cat,ph='X',ts=ts,dur=dur,args=dict(stream=stream))
        trace=dict(traceEvents=[event('e35.moe',0,100,0,'user_annotation'),
            event('nvfp4_expert_kernel',10,40,1),event('qmmf_kernel',30,30,2),
            event('nvfp4_expert_kernel',10,40,1,'gpu_user_annotation'),
            event('qmmf_kernel',110,20,2)])
        result=analyze(trace)
        self.assertEqual(result['kernel_calls'],2)
        self.assertEqual(result['routed_shared_overlap_us'],20)
        self.assertEqual(result['kernel_active_union_us'],50)
