"""Scale storage is part of committed state; softmax shift diagnostics are separate."""
from types import SimpleNamespace
import unittest
import numpy as np
from scripts.e47_state import digest,tail,exact,selector


class Tensor:
    def __init__(self,value):self.data=np.asarray(value)
    def __len__(self):return len(self.data)
    def __getitem__(self,key):return Tensor(self.data[key])
    def clone(self):return Tensor(self.data.copy())
    def contiguous(self):return self
    def view(self,dtype):return Tensor(self.data.view(dtype))
    def cpu(self):return self
    def numpy(self):return self.data
    def item(self):return self.data.item()


def fixture():
    torch=SimpleNamespace(uint8=np.uint8,equal=lambda a,b:np.array_equal(a.data,b.data))
    c=SimpleNamespace(k=Tensor(np.arange(16,dtype=np.int8).reshape(8,2)),v=Tensor(np.ones((8,2),dtype=np.int8)),
                      ks=Tensor(np.ones((8,1),dtype=np.float16)),vs=Tensor(np.ones((8,1),dtype=np.float16)),quantized=True)
    st=SimpleNamespace(kc=[c],ikc=[Tensor(np.zeros((8,2),dtype=np.float16))],pooled=[Tensor(np.zeros((2,2),dtype=np.float16))],pos=4,ratio=4)
    snap=dict(pos=4,mtp_len=0,ple_history=np.array([1,2]),rec=Tensor([3]),conv=Tensor([4]),ple_tail=Tensor([5]))
    st.snapshot=lambda:snap
    return torch,st,snap


class QuantizedStateChecks(unittest.TestCase):
    def test_selector_uses_each_requests_actual_last_tile_before_reuse(self):
        for pos,positions in ((261888,[261889,261985,262080]),(261952,[261953,262017,262080])):
            class Scores:
                def __getitem__(self,key):return Tensor(np.arange(key[1].stop,dtype=np.float32))
            class IDs:
                def __getitem__(self,key):
                    end=pos+key[0]+1;complete=end//4
                    return Tensor(np.arange((complete-512)*4,end,dtype=np.int32))
            class Counts:
                def __getitem__(self,r):return Tensor(2048+(pos+r+1)%4)
            class Sparse:
                def __getitem__(self,r):return Tensor(1)
            s=SimpleNamespace(qsa=True,nb=65540,ratio=4,budget=2048,scores=Scores(),ids=IDs(),nk=Counts(),sparse=Sparse())
            workspace=SimpleNamespace(attn=s,pos_blk=Tensor(pos))
            rows=selector(None,workspace,262080)
            self.assertEqual([r['position'] for r in rows],positions)
            workspace.pos_blk=Tensor(0) # A later short request overwrote shared scratch.
            with self.assertRaises(AssertionError):selector(None,workspace,262080)

    def test_live_scale_mutation_changes_digest_and_fails_tail_equality(self):
        for name in ('ks','vs'):
            torch,st,snap=fixture();before=digest(torch,st,4);expected=tail(st,0)
            getattr(st.kc[0],name).data[1,0]+=1
            self.assertNotEqual(digest(torch,st,4),before)
            with self.assertRaises(AssertionError):exact(torch,st,snap,expected,0)

    def test_uncommitted_tail_is_excluded_until_committed(self):
        torch,st,_=fixture();before=digest(torch,st,4)
        st.kc[0].ks.data[6,0]+=1
        self.assertEqual(digest(torch,st,4),before)
        before_extended=digest(torch,st,8);st.kc[0].ks.data[6,0]+=1
        self.assertNotEqual(digest(torch,st,8),before_extended)


if __name__=='__main__':unittest.main()
