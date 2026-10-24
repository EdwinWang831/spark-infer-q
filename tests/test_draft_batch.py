import importlib.util
import unittest
from types import SimpleNamespace as NS
from contextlib import nullcontext
from unittest.mock import patch

from spark_infer_q.kaifra_batch import BatchedRequestDraftProvider
from spark_infer_q.draft_batch_w8 import load_library
from spark_infer_q.kaifra_speculative import SpeculativeSchedulerBridge


class BatchPolicyTests(unittest.TestCase):
    def fixture(self):
        provider=BatchedRequestDraftProvider.__new__(BatchedRequestDraftProvider)
        calls=[]
        def propose(requests):
            calls.append(tuple(r.handle for r in requests))
            return [([r.anchor+1]*7,dict(signals=[0.]*133)) for r in requests]
        provider.backend=NS(max_requests=4,propose_batch=propose)
        provider.feature_group='f1'
        provider.model=NS(predict=lambda f:[.5**i for i in range(1,8)])
        provider.budget=NS(choose=lambda p,c,r:min(3,r))
        return provider,calls

    def test_one_token_cap_is_skipped_and_e17_remains_request_specific(self):
        provider,calls=self.fixture()
        a,b=NS(context=[None]*3,output_tokens=2),NS(context=[None]*9,output_tokens=7)
        rows=provider.propose_batch([('a',a,11,1),('b',b,22,2)])
        self.assertEqual(calls,[('b',)])
        self.assertEqual(rows[0],([0]*7,[0.]*7))
        self.assertEqual((a.next_width,a.cumulative),(1,()))
        self.assertEqual(rows[1],([23]*7,[.5]*7))
        self.assertEqual(b.next_width,2)
        self.assertEqual(len(b.cumulative),7)

    def test_validate_batch_before_any_compute_and_empty_cancelled_batch(self):
        provider,calls=self.fixture();state=NS(context=[1],output_tokens=1)
        self.assertEqual(provider.propose_batch([]),[])
        for rows in ([('a',state,1,0)], [('a',state,1,True)],
                     [('a',state,1,3),('a',state,2,3)],
                     [(i,state,1,3) for i in range(5)]):
            with self.assertRaises(ValueError):provider.propose_batch(rows)
        self.assertEqual(calls,[])

    def test_no_partial_policy_publication_on_invalid_later_result(self):
        provider,calls=self.fixture();count=0
        def choose(*args):
            nonlocal count
            count+=1
            return 2 if count==1 else 20
        provider.budget=NS(choose=choose)
        states=[NS(context=[1],output_tokens=1,cumulative=(),next_width=1) for _ in range(2)]
        with self.assertRaisesRegex(RuntimeError,'invalid batched draft'):
            provider.propose_batch([(str(i),s,1,4) for i,s in enumerate(states)])
        self.assertTrue(all(s.next_width==1 and s.cumulative==() for s in states))

    def test_library_hash_checked_before_loading_native_code(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fake.so';p.write_bytes(b'not an extension')
            with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):load_library(p,'0'*64)

    def test_old_joint_cost_table_rejects_batch_kernel_even_through_python_api(self):
        from pathlib import Path
        import json
        from spark_infer_q.joint_budget import JointBudget
        root=Path(__file__).resolve().parents[1]
        table=json.loads((root/'results/k4/2026-10-24-joint-costs.json').read_text())
        worker=NS(block_size=8,feature_group='f1',torch=None,weights={},fused={},used=set())
        base=NS(deferred_draft=True,model=object(),budget=object(),feature_group='f1',
            max_context=2048,worker=worker,profile_metadata=table['profile'])
        own=BatchedRequestDraftProvider(base,None,kernel_identity={'binary_sha256':'f'*64})
        with self.assertRaisesRegex(ValueError,'execution profile mismatch'):
            JointBudget(table).validate_runtime(own.profile_metadata,NS())
        self.assertEqual(base.profile_metadata,table['profile'])

    def test_native_prepare_uses_one_batch_and_excludes_cancelled_requests(self):
        bridge=SpeculativeSchedulerBridge.__new__(SpeculativeSchedulerBridge)
        states=[NS(output_tokens=2,next_width=2) for _ in range(3)]
        entries={i:dict(state=s,cancelled=i==1,seq=NS(sampling_params=NS(max_tokens=9)))
                 for i,s in enumerate(states)}
        states[1].next_width=1
        calls=[]
        def propose(rows):
            calls.append([r[0] for r in rows])
            return [([r[2]+1]*7,[.5]*7) for r in rows]
        bridge.backend=NS(provider=NS(propose_batch=propose))
        bridge.postdraft=True;bridge._entry=entries.__getitem__
        bridge.torch=NS(inference_mode=nullcontext,cuda=NS(synchronize=lambda:None))
        bridge._cpu_arrays=dict(widths=[0]*3)
        bridge._cpu_views=dict(widths={3:None},positions={3:None})
        rows=tuple(NS(handle=i,token_ids=(10,11,12),prompt=False) for i in range(3))
        plan=NS(rows=rows,metadata=(1,2,3),native_output=NS(scheduled_seqs=(1,2,3),
            update_metadata_for_sampling_zos=lambda *args:None))
        bridge._work_row=lambda meta,seq,entry:rows[meta-1]
        with patch('spark_infer_q.deepinfer_bridge.SchedulerBridge.prepare',return_value=plan):
            self.assertIs(bridge.prepare(),plan)
        self.assertEqual(calls,[[0,2]])
        self.assertFalse(hasattr(states[1],'current_drafts'))
        self.assertEqual(states[0].current_drafts,(11,)*7)
        self.assertEqual(plan.draft_batch_size,2)
        self.assertEqual(plan.draft_ms[1],0.)
        self.assertAlmostEqual(sum(plan.draft_ms),plan.draft_batch_ms)


@unittest.skipUnless(importlib.util.find_spec('torch'),'optional PyTorch absent')
class BatchTensorTests(unittest.TestCase):
    def test_convolution_cannot_cross_block_boundaries(self):
        import torch
        from spark_infer_q.draft_batch import block_convolution
        torch.manual_seed(19)
        x=torch.randn(24,2560).to(torch.bfloat16)
        delta=torch.randn(24,2,160).to(torch.bfloat16)
        base=torch.randn(2,160,16).to(torch.bfloat16)
        actual=block_convolution(torch,x,delta,base,8)
        references=[]
        for i in range(3):
            block=x[i*8:(i+1)*8].reshape(8,160,16)
            coefficients=base[None]+delta[i*8:(i+1)*8,...,None]
            expected=coefficients[:,0]*block
            expected[1:]+=coefficients[1:,1]*block[:-1]
            references.append(expected.flatten(-2))
        self.assertTrue(torch.equal(actual,torch.cat(references)))
        changed=x.clone();changed[7].fill_(10000)
        after=block_convolution(torch,changed,delta,base,8)
        self.assertTrue(torch.equal(actual[8:],after[8:]))

    def test_unequal_attention_isolated_and_padding_masked(self):
        import torch
        from spark_infer_q.draft_batch import BatchedDFlash,DraftInput
        backend=BatchedDFlash.__new__(BatchedDFlash);backend.torch=torch
        torch.manual_seed(23)
        contexts=[torch.randn(n,8,128).to(torch.bfloat16) for n in (3,9)]
        values=[torch.randn_like(c) for c in contexts]
        ck,cv=torch.cat(contexts),torch.cat(values)
        backend.worker=NS(rope=lambda x,p:x,norm=lambda x,p:x)
        backend._context_linear=lambda x,name,updates:ck if 'k_proj' in name else cv
        q=torch.randn(16,32,128).to(torch.bfloat16)
        k=torch.randn(16,8,128).to(torch.bfloat16);v=torch.randn_like(k)
        requests=[DraftInput(i,torch.empty(len(c),12800),1) for i,c in enumerate(contexts)]
        actual=backend._attention(requests,[(None,0,3,0),(None,0,9,3)],torch.empty(12),None,0,q,k,v)
        expected=[]
        for i in range(2):
            keys=torch.cat((contexts[i],k[i*8:(i+1)*8])).repeat_interleave(4,dim=1).transpose(0,1).float()
            vals=torch.cat((values[i],v[i*8:(i+1)*8])).repeat_interleave(4,dim=1).transpose(0,1).float()
            scores=q[i*8:(i+1)*8].transpose(0,1).float()@keys.transpose(-1,-2)*128**-.5
            expected.append((scores.softmax(-1)@vals).transpose(0,1).reshape(8,4096).to(q.dtype))
        self.assertTrue(torch.equal(actual,torch.cat(expected)))


if __name__=='__main__':unittest.main()
