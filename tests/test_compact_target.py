import importlib.util
from types import SimpleNamespace as NS
import unittest

from spark_infer_q.compact_target import CompactTarget


class CompactTransactionTests(unittest.TestCase):
    def fixture(self):
        target=CompactTarget.__new__(CompactTarget)
        states=[NS(epoch=3,pos=9),NS(epoch=7,pos=11)]
        target.states=states;target.leased=[True,True]
        windows=tuple(NS(state=s,ids=(1,2,3),at=i*3,physical_rows=3) for i,s in enumerate(states))
        ticket=NS(phase='ready',windows=windows,epochs=(3,7))
        target.inflight=ticket
        return target,ticket,states

    def test_prevalidate_all_prefixes_before_any_metadata_or_state_mutation(self):
        target,ticket,states=self.fixture()
        for invalid in (-1,4,True,2.,None):
            with self.assertRaises(ValueError):target.commit(ticket,{id(states[0]):2,id(states[1]):invalid})
            self.assertEqual([s.pos for s in states],[9,11]);self.assertEqual(ticket.phase,'ready')
        with self.assertRaises(ValueError):target.commit(ticket,{id(object()):1})

    def test_stale_generation_cannot_commit_or_release_another_lease(self):
        target,ticket,states=self.fixture();states[1].epoch+=1
        with self.assertRaisesRegex(RuntimeError,'generation'):
            target.commit(ticket,{id(states[0]):1,id(states[1]):1})
        self.assertEqual([s.pos for s in states],[9,11])
        with self.assertRaises(RuntimeError):target.release_state(states[0])
        ticket.phase='committing'
        with self.assertRaises(RuntimeError):target.commit(ticket,{})


@unittest.skipUnless(importlib.util.find_spec('torch'),'optional PyTorch absent')
class OwnedStateTests(unittest.TestCase):
    def fixture(self):
        import torch
        from spark_infer_q.qwen_state import QwenState
        import numpy as np
        c=NS(conv_kernel=4,conv_dim=24,nv=2,dv=4,dk=4,nk=2,index_ratio=4,index_dim=4,
            kv_heads=1,head_dim=8,ple_kernel=4,ngram_size=3,streams=2,hidden=8,ple_layers=[1],
            ngram=lambda i:NS(initial_history=lambda:np.array([0,0],dtype=np.int64)))
        w=NS(cfg=c,device='cpu',mtp=None,comm=None,layers=[NS(index=0,linear=True),NS(index=1,linear=False)])
        return torch,w,QwenState(torch,w,16)

    def test_clone_restore_and_reset_keep_storage_ownership_explicit(self):
        torch,w,state=self.fixture()
        state.rec[0].fill_(3);state.conv.fill_(2);state.kc[0].k[:3].fill_(4);state.set_pos(3)
        other=state.clone();self.assertNotEqual(state.rec.data_ptr(),other.rec.data_ptr())
        self.assertNotEqual(state.kc[0].k.data_ptr(),other.kc[0].k.data_ptr())
        snapshot=state.snapshot();state.rec[0].zero_();state.restore(snapshot)
        self.assertTrue(torch.equal(state.rec[0],other.rec[0]))
        old_pointer=state.rec.data_ptr();state.reset(w)
        self.assertEqual(state.rec.data_ptr(),old_pointer);self.assertEqual(state.pos,0)
        self.assertEqual(state.epoch,1);self.assertFalse(bool(state.rec.any()))
        state.copy_from(other);self.assertEqual(state.pos,3);self.assertTrue(torch.equal(state.conv,other.conv))
        self.assertEqual(state.rec.data_ptr(),old_pointer)
        self.assertEqual(state.ensure(16),0)
        with self.assertRaises(ValueError):state.ensure(17)


if __name__=='__main__':unittest.main()
