import importlib.util
import unittest
import sys
from pathlib import Path

from spark_infer_q.draft_backbone import PackedLinear,quantize_w8

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from e28_online import order,summarize


class DraftTimingTests(unittest.TestCase):
    def test_four_repetitions_balance_each_prompt(self):
        for ti in range(8):
            for mode in ('bf16','w8'):
                self.assertEqual(sum(order(ti,rep)[0]==mode for rep in range(4)),2)

    def test_actual_decode_accounting_and_incomplete_rejection(self):
        tasks=[]
        for i in range(8):
            ts=[]
            for rep in range(4):
                for mode,ms in [('bf16',1000.),('w8',900.)]:
                    steps=[dict(emitted=4,accepted=3,observed=7,draft_ms=1.,verify_ms=30.,commit_ms=1.) for _ in range(25)]
                    ts.append(dict(mode=mode,repetition=rep,decode_ms=ms,steps=steps,exact=True))
            tasks.append(dict(group=str(i),completion_tokens=101,timings=ts))
        s=summarize(tasks,samples=100)
        self.assertEqual(s['timed_tokens_per_arm'],3200)
        self.assertEqual(s['trace']['w8']['emitted'],3200)
        self.assertEqual(s['decode_tps']['bf16'],100.)
        self.assertAlmostEqual(s['relative_gain'],1/9)
        tasks[0]['timings'].pop()
        with self.assertRaises(AssertionError):summarize(tasks,samples=100)


@unittest.skipUnless(importlib.util.find_spec('torch'),'optional PyTorch absent')
class DraftBackboneTests(unittest.TestCase):
    def test_packed_linear_dispatch_preserves_activations_and_original_weight(self):
        import torch
        w=torch.tensor([[0.,0.,0.,0.],[.1,-.2,.3,-.4]],dtype=torch.bfloat16)
        before=w.clone();packed=quantize_w8(w);calls=[]
        def backend(x,p):
            calls.append(x.dtype)
            return (x.float()@(p.values.float()*p.scales[:,None]).T).to(x.dtype)
        proxy=PackedLinear(packed,backend);x=torch.ones((3,4),dtype=torch.bfloat16)
        got=torch.nn.functional.linear(x,proxy)
        expected=backend(x,packed)
        self.assertTrue(torch.equal(got,expected));self.assertTrue(torch.equal(w,before))
        self.assertEqual(calls,[torch.bfloat16,torch.bfloat16]);self.assertTrue(torch.isfinite(got).all())
        self.assertEqual(packed.nbytes(),16)
        with self.assertRaises(ValueError):torch.nn.functional.linear(x,proxy,torch.zeros(2))

    def test_invalid_weight_and_channel_scales(self):
        import torch
        for w in (torch.zeros(2,3),torch.zeros(2,dtype=torch.bfloat16),torch.full((2,3),float('nan'),dtype=torch.bfloat16)):
            with self.assertRaises(ValueError):quantize_w8(w)
        w=torch.tensor([[127.,-127.],[1.,-1.]],dtype=torch.bfloat16);p=quantize_w8(w)
        self.assertTrue(torch.equal(p.values,torch.tensor([[127,-127],[127,-127]],dtype=torch.int8)))
        self.assertEqual(p.scales[0],1.)


if __name__=='__main__':unittest.main()
