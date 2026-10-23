import copy
import importlib.util
import math
import random
import unittest

from spark_infer_q.local_policy import ProgressModel
from spark_infer_q.confidence_head import choose_rows
from spark_infer_q.residual_head import NumpyLinearProgressModel, ResidualProgressModel, fit_residual


def base():
    return dict(mean=[.2,-.4], scale=[.5,2.],
        heads=[dict(weights=[.7,-.1],bias=.3),dict(weights=[-.2,.8],bias=-.1)],
        calibration=[dict(slope=1.4,bias=-.2),dict(slope=.6,bias=.5)])


def artifact(alpha=1.):
    return dict(architecture='frozen_linear_residual_tanh_v1',base=base(),alpha=alpha,
        layers=[dict(weights=[[.4,-.2],[.1,.5]],bias=[.1,-.3]),
                dict(weights=[[.3,-.7],[-.4,.2]],bias=[.1,-.1])])


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional NumPy absent')
class ResidualRuntimeTests(unittest.TestCase):
    def test_offset_is_added_after_original_calibration_and_is_monotonic(self):
        m=artifact(.5);model=ResidualProgressModel(m);x=[.9,-1.2]
        z=[max(-6,min(6,(v-a)/b)) for v,a,b in zip(x,m['base']['mean'],m['base']['scale'])]
        h=[math.tanh(sum(w*v for w,v in zip(row,z))+b) for row,b in zip(m['layers'][0]['weights'],m['layers'][0]['bias'])]
        raw=ProgressModel(m['base']).raw_logits(x);expected=[];product=1.
        for d,c in enumerate(m['base']['calibration']):
            delta=sum(w*v for w,v in zip(m['layers'][1]['weights'][d],h))+m['layers'][1]['bias'][d]
            product*=1/(1+math.exp(-(c['slope']*raw[d]+c['bias']+.5*delta)))
            expected.append(product)
        self.assertLess(max(abs(a-b) for a,b in zip(model.predict(x),expected)),1e-14)
        self.assertGreaterEqual(expected[0],expected[1])

    def test_disable_is_exact_original_and_fast_linear_actions_match(self):
        rng=random.Random(26);original=ProgressModel(base());fast=NumpyLinearProgressModel(base());disabled=ResidualProgressModel(artifact(0.))
        for _ in range(100):
            x=[rng.uniform(-20,20) for _ in range(2)]
            self.assertEqual(original.predict(x),disabled.predict(x))
            self.assertLess(max(abs(a-b) for a,b in zip(original.predict(x),fast.predict(x))),1e-14)
            self.assertEqual(choose_rows(original.predict(x),[40,44,48]),choose_rows(fast.predict(x),[40,44,48]))

    def test_nonfinite_and_wrong_shapes_are_rejected(self):
        model=ResidualProgressModel(artifact())
        for values in ([float('nan'),1.],[],[[1.,2.]],[1.,2.,3.]):
            with self.assertRaises(ValueError):model.predict(values)
        for edit in ('alpha','weights','scale','slope'):
            m=artifact()
            if edit=='alpha':m['alpha']=-.1
            if edit=='weights':m['layers'][1]['weights'][0]=[1.]
            if edit=='scale':m['base']['scale'][0]=0.
            if edit=='slope':m['base']['calibration'][0]['slope']=0.
            with self.assertRaises(ValueError):ResidualProgressModel(m)

    def test_zero_residual_preserves_base_and_serialized_model_is_independent(self):
        m=artifact()
        for row in m['layers'][1]['weights']:
            for j in range(len(row)):row[j]=0.
        m['layers'][1]['bias']=[0.,0.]
        model=ResidualProgressModel(m)
        self.assertEqual(model.predict([.5,.8]),NumpyLinearProgressModel(m['base']).predict([.5,.8]))
        self.assertEqual(model.residual_parameter_count,12)
        p=model.predict([.5,.8]);m['base']['heads'][0]['weights'][0]=999.
        self.assertEqual(model.predict([.5,.8]),p)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'), 'optional training runtime absent')
class ResidualFitTests(unittest.TestCase):
    def test_frozen_base_survives_fit_inside_inference_mode(self):
        import torch
        rng=random.Random(26);rows=[]
        for _ in range(120):
            x=[rng.uniform(-2,2),rng.uniform(-2,2)]
            first=int(x[0]*x[1]>0);second=first*int(x[0]>.2)
            rows.append(dict(x=x,y=[first,second]))
        b=base();before=copy.deepcopy(b)
        with torch.inference_mode():model=fit_residual(rows,b,hidden=4,max_iter=35)
        self.assertEqual(b,before);self.assertEqual(model.metadata['base'],before)
        self.assertLess(model.metadata['objective_final'],model.metadata['objective_initial'])
        self.assertEqual(model.residual_parameter_count,22)


if __name__=='__main__':unittest.main()
