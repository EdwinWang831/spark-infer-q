import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e15_metrics import grouped_metrics
from spark_infer_q.confidence_head import headroom_capture
from spark_infer_q.confidence_features import feature_vector


class ConfidenceTrainingMetricsTests(unittest.TestCase):
    def test_oracle_optimizes_the_sum_ratio_not_each_position_ratio(self):
        positions=[dict(group='a',emitted=[1,2],rates=[1.],max_rows=2,costs_ms=[2.,3.5],baseline_costs_ms=[2.,3.5]),
                   dict(group='b',emitted=[1,1],rates=[0.],max_rows=1,costs_ms=[.1,.2],baseline_costs_ms=[.1,.2])]
        r=headroom_capture(positions,[2.,3.5],2)
        self.assertAlmostEqual(r['oracle'],2/2.1)
        self.assertLess(r['policy'],r['oracle'])

    def test_group_bootstrap_keeps_paired_variants_together(self):
        p=dict(group='same_semantic_group',emitted=[1,1],rates=[0.],max_rows=2,
               costs_ms=[50.,60.],baseline_costs_ms=[50.,60.])
        r=grouped_metrics([p,p],fixed_rows=2,samples=32)
        self.assertEqual(r['groups'],1)
        self.assertAlmostEqual(r['captured'],1.)
        self.assertEqual(r['group_bootstrap_gain_95'][0],r['group_bootstrap_gain_95'][1])

    def test_feature_ablation_and_known_remaining_capacity(self):
        signals=list(range(133))
        f0=feature_vector(signals,1024,17,79,False,'f0')
        f1=feature_vector(signals,1024,17,79,False,'f1')
        self.assertEqual(len(f0),25);self.assertEqual(len(f1),137)
        self.assertEqual(f0,f1[:25])
        with self.assertRaises(ValueError):feature_vector(signals,1024,17,0,False)
