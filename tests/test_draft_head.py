import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from spark_infer_q.draft_head import Q4Backend,make_projection,vocabulary_ids
from e6_tasks import HEADS,POLICIES,TASKS,CALIBRATION,order


class ConfigurationTests(unittest.TestCase):
    def test_vocabulary_preserves_eos_and_rejects_wrong_ids(self):
        self.assertEqual(vocabulary_ids(range(16),[20,20],21),list(range(16))+[20])
        for values in ([-1]+list(range(16)),list(range(17))):
            with self.assertRaises(ValueError):vocabulary_ids(values,[],16)
        with self.assertRaises(TypeError):vocabulary_ids([1.5]+list(range(16)),[],20)

    def test_backend_is_optional_and_fallback_is_observable(self):
        with patch('spark_infer_q.draft_head.HeadProjection'),self.assertRaises(RuntimeError):
            make_projection(object(),precision='q4')
        backend=Q4Backend('unsupported',lambda w:False,None,None)
        with patch('spark_infer_q.draft_head.HeadProjection') as projection:
            with self.assertWarnsRegex(RuntimeWarning,'using BF16'):
                got=make_projection(object(),precision='q4',backend=backend,unavailable='bf16')
            self.assertEqual(got.effective_precision,'bf16');self.assertEqual(got.requested_precision,'q4')
            self.assertEqual(got.backend,'torch.linear')
        with self.assertRaises(ValueError):make_projection(object(),precision='int3')

    def test_protocol_is_frozen_and_head_positions_balance(self):
        self.assertEqual(len(TASKS),8);self.assertEqual(len(CALIBRATION),2)
        self.assertFalse({t['prompt'] for t in TASKS}&{t['prompt'] for t in CALIBRATION})
        for ti in range(8):
            for slot in range(4):
                self.assertEqual({order(rep,ti)[slot*2][0] for rep in range(4)},set(HEADS))
            self.assertEqual(set(order(0,ti)),{(h,p) for h in HEADS for p in POLICIES})


@unittest.skipUnless(importlib.util.find_spec('torch'),'optional PyTorch absent')
class ProjectionTests(unittest.TestCase):
    def test_global_id_remap_and_shared_target_unchanged(self):
        import torch
        from spark_infer_q.draft_head import HeadProjection
        weight=torch.arange(40,dtype=torch.float32)[:,None];before=weight.clone()
        ids=torch.tensor(list(range(0,40,2)));hidden=torch.ones(7,1)
        full=HeadProjection(weight);subset=HeadProjection(weight,ids=ids)
        self.assertEqual(full.weight.data_ptr(),weight.data_ptr())
        self.assertEqual(full.payload_bytes,0)
        self.assertEqual(subset.topk(hidden)[1][0].tolist(),list(range(38,6,-2)))
        self.assertTrue(torch.equal(weight,before))
        with self.assertRaises(ValueError):HeadProjection(weight,ids=torch.tensor([0]*16))

    def test_quantized_provider_and_default_are_separate(self):
        import torch
        weight=torch.randn(32,8);hidden=torch.randn(7,8);before=weight.clone()
        class Packed:
            def __init__(self,w):self.w=w.clone()
            def nbytes(self):return self.w.numel()*self.w.element_size()
        backend=Q4Backend('test-only',lambda w:True,Packed,lambda x,q:x@q.w.t())
        got=make_projection(weight,precision='q4',backend=backend)
        ref=make_projection(weight)
        self.assertTrue(torch.allclose(got.logits(hidden),ref.logits(hidden)))
        self.assertTrue(torch.equal(weight,before));self.assertEqual(got.effective_precision,'q4')
        subset_backend=Q4Backend('subset-only',lambda w:len(w)==16,Packed,lambda x,q:x@q.w.t())
        subset=make_projection(weight,ids=torch.arange(16),precision='q4',backend=subset_backend)
        self.assertEqual(subset.logits(hidden).shape,(7,16))


if __name__=='__main__':unittest.main()
