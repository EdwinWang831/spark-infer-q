import unittest
from spark_infer_q.local_policy import MODES, ProgressModel, choose_width, mode_order


class LocalPolicyTests(unittest.TestCase):
    def test_decision_costs_include_paid_drafter(self):
        # Once draft cost is paid it must be present in *every* post-draft choice.
        self.assertEqual(choose_width([.2,.1,0.],[1.,1.5,1.55,1.7]),1)
        self.assertEqual(choose_width([.2,.1,0.],[2.,2.5,2.55,2.7]),3)

    def test_rates_are_cumulative_and_prediction_is_frozen(self):
        metadata={'mean':[0.], 'scale':[1.],
            'heads':[{'weights':[0.], 'bias':0.} for _ in range(3)],
            'calibration':[{'slope':1.,'bias':0.} for _ in range(3)]}
        model=ProgressModel(metadata)
        self.assertEqual(model.predict([1.]),[.5,.25,.125])
        self.assertEqual(model.predict([-100.]),[.5,.25,.125])
        self.assertEqual(model.metadata,metadata)
        with self.assertRaises(ValueError):model.predict([float('nan')])
        with self.assertRaises(ValueError):choose_width([.2,.3,0.],[1,2,3,4])

    def test_output_cap_requires_no_unneeded_draft(self):
        self.assertEqual(choose_width([1,1,1],[1,1,1,1],max_width=1),1)

    def test_seven_modes_are_balanced(self):
        for column in zip(*(mode_order(r,2) for r in range(7))):
            self.assertEqual(set(column),set(MODES))
