import importlib.util
import math
import random
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch absent')
class DecisionHeadTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        self.torch.set_num_threads(self.threads)

    def rows(self):
        rng = random.Random(13)
        rows = []
        for _ in range(24):
            n = rng.randrange(1, 9)
            rows.append(dict(x=[rng.uniform(-1,1) for _ in range(137)],
                y=[int(n > d+1) for d in range(7)], emitted=[min(w,n) for w in range(1,9)], max_rows=8))
        return rows

    def test_zero_action_weight_reproduces_original_fitter(self):
        from spark_infer_q.mlp_head import fit_mlp
        from spark_infer_q.decision_head import fit_decision_mlp
        rows = self.rows()
        kwargs = dict(hidden=(3,2),seed=2,max_iter=4)
        with self.torch.inference_mode():
            a = fit_mlp(rows, rows, **kwargs)
            b = fit_decision_mlp(rows, rows, [[40.+4*w for w in range(8)]]*len(rows),
                                 .04, alpha=0., **kwargs)
            self.assertTrue(self.torch.is_inference_mode_enabled())
        self.assertEqual(a.metadata['layers'],b.metadata['layers'])
        self.assertEqual(a.metadata['calibration'],b.metadata['calibration'])

    def test_regret_gradient_penalizes_waste_and_masks_infeasible_widths(self):
        from spark_infer_q.decision_head import soft_action_regret
        torch = self.torch
        logits = torch.zeros((1,7),dtype=torch.float64,requires_grad=True)
        curves = torch.tensor([[40.+5*i for i in range(8)]],dtype=torch.float64)
        loss = soft_action_regret(torch,logits,curves,torch.ones((1,8),dtype=torch.bool),
                                  (curves-40.)/75.,.04,.1)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[0,0]),0.)
        logits.grad.zero_()
        mask = torch.zeros((1,8),dtype=torch.bool);mask[0,0]=True
        loss = soft_action_regret(torch,logits,curves,mask,(curves-40.)/75.,.04,.1)
        loss.backward()
        self.assertEqual(float(loss),0.)
        self.assertEqual(float(logits.grad.abs().sum()),0.)

    def test_action_objective_fits_under_resident_context_without_changing_features(self):
        from spark_infer_q.decision_head import fit_decision_mlp
        rows = self.rows()
        before = list(rows[0]['x'])
        with self.torch.inference_mode():
            model = fit_decision_mlp(rows,rows,[[40.+4*w for w in range(8)]]*len(rows),
                                     .04,hidden=(3,2),max_iter=4)
        self.assertEqual(rows[0]['x'],before)
        self.assertLess(model.metadata['objective_final'],model.metadata['objective_initial'])
        self.assertTrue(all(math.isfinite(v) for v in model.predict(before)))
        self.assertEqual(model.metadata['decision_objective']['alpha'],1.)


if __name__ == '__main__':
    unittest.main()
