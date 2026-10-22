import importlib.util
import math
import unittest
from spark_infer_q.regret_head import validated_weights,fit_regret_heads
from spark_infer_q.confidence_head import fit_heads


class RegretWeightsTests(unittest.TestCase):
    def test_rejects_nonfinite_out_of_bounds_training_weights(self):
        self.assertEqual(validated_weights([{},dict(weight=4.)]),[1.,4.])
        for weight in (0.,.5,4.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):validated_weights([dict(weight=weight)])


@unittest.skipUnless(importlib.util.find_spec('torch'),'optional PyTorch absent')
class WeightedHeadFitTests(unittest.TestCase):
    def data(self):
        return [dict(x=[float(i%11)-5.,float((i*7)%13)-6.],y=[int(i%11>4),int(i%11>8)]) for i in range(220)]

    def test_uniform_weights_reproduce_original_and_constant_rescaling(self):
        import torch
        torch.set_num_threads(2)
        data=self.data();cal=data[:66]
        ordinary=fit_heads(data,cal,max_iter=25)
        weighted=fit_regret_heads([dict(r,weight=1.) for r in data],cal,max_iter=25)
        scaled=fit_regret_heads([dict(r,weight=4.) for r in data],cal,max_iter=25)
        for r in cal[::10]:
            for a,b,c in zip(ordinary.predict(r['x']),weighted.predict(r['x']),scaled.predict(r['x'])):
                self.assertAlmostEqual(a,b,places=10);self.assertAlmostEqual(b,c,places=10)
        self.assertTrue(all(abs(h['normalized_weight_mean']-1.)<1e-12 for h in weighted.metadata['heads']))

    def test_weighted_fit_runs_inside_inference_mode_and_preserves_cumulative_rates(self):
        import torch
        torch.set_num_threads(2)
        data=self.data()
        with torch.inference_mode():model=fit_regret_heads([dict(r,weight=4. if r['y'][0]==0 else 1.) for r in data],data[:66],max_iter=25)
        for r in data[::20]:
            p=model.predict(r['x']);self.assertEqual(p,sorted(p,reverse=True))
            self.assertTrue(all(math.isfinite(v) and 0<=v<=1 for v in p))
        self.assertEqual(model.metadata['calibration_rows'],66)
