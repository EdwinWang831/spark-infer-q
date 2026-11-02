import importlib.util
import unittest
from types import SimpleNamespace as NS
from spark_infer_q.draft_batch_graph import BufferedBatchedDFlash
from spark_infer_q.draft_batch import BatchedDFlash, DraftInput, ContextCache


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch absent')
class DraftBuffers(unittest.TestCase):
    def test_exact_attention_and_reuse_across_companion_lengths(self):
        import torch
        torch.manual_seed(121)
        worker=NS(block_size=8,feature_group='f1',torch=torch,
            embed=torch.empty(100,1),norm=lambda x,p:x,rope=lambda x,p:x)
        ref=BatchedDFlash(worker);own=BufferedBatchedDFlash(worker)
        for lengths in ((3,9),(7,9),(9,3),(9,),(3,9)):
            contexts=[torch.randn(n,8,128).to(torch.bfloat16) for n in lengths]
            values=[torch.randn_like(x) for x in contexts]
            ck,cv=torch.cat(contexts),torch.cat(values)
            for b in (ref,own):
                b._context_linear=lambda x,name,updates:ck if 'k_proj' in name else cv
            requests=[DraftInput(i,torch.empty(n,12800),1) for i,n in enumerate(lengths)]
            updates=[];at=0
            for n in lengths:
                updates.append((None,0,n,at));at+=n
            q=torch.randn(len(lengths)*8,32,128).to(torch.bfloat16)
            k=torch.randn(len(lengths)*8,8,128).to(torch.bfloat16);v=torch.randn_like(k)
            expected=ref._attention(requests,updates,ck,None,0,q,k,v)
            actual=own._attention(requests,updates,ck,None,0,q,k,v)
            self.assertTrue(torch.equal(expected,actual))
        self.assertEqual(own.counters['attention_buffer_allocations'],4)
        self.assertEqual(own.counters['attention_buffer_reuses'],3)
        self.assertEqual(len(own.attention_buffers),2)

    def test_cached_prefix_append_and_unwritten_padding(self):
        import torch
        torch.manual_seed(122)
        worker=NS(block_size=8,feature_group='f1',torch=torch,
            embed=torch.empty(100,1),norm=lambda x,p:x,rope=lambda x,p:x)
        ref=BatchedDFlash(worker);own=BufferedBatchedDFlash(worker)
        ck=torch.randn(8,8,128).to(torch.bfloat16);cv=torch.randn_like(ck)
        prefix_k=torch.randn(5,20,8,128).to(torch.bfloat16);prefix_v=torch.randn_like(prefix_k)
        requests=[DraftInput(i,torch.empty(n,12800),1) for i,n in enumerate((8,12))]
        q=torch.randn(16,32,128).to(torch.bfloat16)
        k=torch.randn(16,8,128).to(torch.bfloat16);v=torch.randn_like(k)
        outputs=[]
        for b in (ref,own):
            caches=[ContextCache(prefix_k.clone(),prefix_v.clone(),old) for old in (4,8)]
            updates=[(caches[0],4,4,0),(caches[1],8,4,4)]
            b._context_linear=lambda x,name,updates:ck if 'k_proj' in name else cv
            outputs.append(b._attention(requests,updates,ck,None,2,q,k,v))
            self.assertTrue(torch.equal(caches[0].keys[2,:4],prefix_k[2,:4]))
            self.assertTrue(torch.equal(caches[1].values[2,8:12],cv[4:]))
        self.assertTrue(torch.equal(*outputs))

    def test_close_rejects_live_generations(self):
        worker=NS(block_size=8,feature_group='f1',torch=None)
        own=BufferedBatchedDFlash(worker);own.caches['live']=object()
        with self.assertRaisesRegex(RuntimeError,'release draft generations'):
            own.close()


if __name__=='__main__':unittest.main()
