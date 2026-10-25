from types import SimpleNamespace
from contextlib import nullcontext
import unittest

from spark_infer_q.kaifra_graph import GraphTarget


class GraphStateOwnership(unittest.TestCase):
    def target(self):
        class State:
            resets=0
            def reset(self,w):self.resets+=1
        self.fences=0
        def sync():self.fences+=1
        return SimpleNamespace(buffers={i:None for i in range(1,9)},w=object(),
            state=State,torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=sync)))

    def test_static_lease_reuse_and_foreign_state_rejected(self):
        target=GraphTarget(self.target());first=target.state()
        with self.assertRaisesRegex(RuntimeError,'already leased'):target.state()
        with self.assertRaisesRegex(RuntimeError,'request is active'):target.warmup()
        with self.assertRaisesRegex(RuntimeError,'active request'):target.close()
        with self.assertRaisesRegex(RuntimeError,'foreign'):target.release_state(object())
        target.release_state(first)
        with self.assertRaisesRegex(RuntimeError,'released'):target.release_state(first)
        second=target.state();self.assertIs(first,second);self.assertEqual(first.resets,1)
        target.release_state(second);target.close()
        self.assertFalse(target.capabilities['cuda_graph'])
        with self.assertRaisesRegex(RuntimeError,'closed'):target.state()

    def test_bucket_configuration_and_unleased_execution_rejected(self):
        for widths in ((),(0,),(9,),(1,1),(True,),(1.5,)):
            with self.subTest(widths=widths),self.assertRaises(ValueError):GraphTarget(self.target(),widths=widths)
        target=GraphTarget(self.target())
        with self.assertRaisesRegex(RuntimeError,'active request'):target.run(object(),[1])

    def test_large_capture_reuses_committed_kv_prefix_without_full_cache_clone(self):
        import numpy as np
        from tests.test_rolling_history import Rows
        class State:
            version=0
            def __init__(self):
                self.cache=[0]*16;self.pos=0;self.cur=[0];self.rec=Rows(np.zeros((2,1)))
            def snapshot(self):return dict(pos=self.pos,rec=self.rec[self.cur[0]].clone())
            def restore(self,snap):
                self.pos=snap['pos'];self.rec[0].copy_(snap['rec']);self.cur=[0]
            def clone(self):raise AssertionError('large KV cache must not be cloned')
        state=State();calls=[]
        def step(s,ids,capture):
            s.cache[0]=123;s.pos=1;s.cur=[1];s.rec[1].copy_(Rows([77.]))
        def stage(w,b,segments):
            self.assertEqual((state.pos,state.cache[0]),(1,123))
            self.assertEqual(state.rec.data[state.cur[0],0],77.)
        cuda=SimpleNamespace(synchronize=lambda:None,graph_pool_handle=object,
            CUDAGraph=lambda:object(),graph=lambda *a,**k:nullcontext())
        eager=SimpleNamespace(capacity=262160,w=object(),buffers={r:object() for r in range(1,9)},
            state=lambda:state,step=step,forward=SimpleNamespace(stage=stage),
            torch=SimpleNamespace(cuda=cuda,inference_mode=nullcontext))
        target=GraphTarget(eager)
        def compute(s,rows):
            self.assertEqual(s.cache[0],123)
            s.cache[1:1+rows]=[rows]*rows;calls.append((rows,s.cur[0]));return (rows,)
        target._compute_features=compute;result=target.warmup()
        self.assertEqual(result['graph_captures'],16)
        self.assertEqual(set(calls),{(r,p) for r in range(1,9) for p in (0,1)})
        self.assertEqual((state.cache[0],state.pos,target.leased),(123,1,False))
        target.close()


if __name__=='__main__':unittest.main()
