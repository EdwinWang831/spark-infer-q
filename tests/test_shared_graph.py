"""Shared serial workspace ownership; fake slots are not GPU batch evidence."""
from types import SimpleNamespace
import unittest

from spark_infer_q.shared_graph import SharedGraphTarget


class Slot:
    def __init__(self,eager,screen):
        self.eager,self.screen=eager,screen;self.leased=False;self.workspace=None;self.fence_fails=False
        self.capabilities=dict(graph_captures=16,graph_replays=0,head_calls=0,head_fallbacks=0,graph_fallbacks={})
    def warmup(self):pass
    def state(self):
        if self.workspace is None:self.workspace=SimpleNamespace(pos=0,values=[])
        else:self.workspace.pos=0;self.workspace.values.clear()
        self.leased=True;return self.workspace
    def release_state(self,state):
        if self.fence_fails:raise RuntimeError('device fence failed')
        self.leased=False
    def run(self,state,ids,capture=True):
        self.capabilities['graph_replays']+=1
        return (state.values[:],list(ids))
    def commit(self,state,buffers,rows,keep):state.pos+=keep;state.values+=list(range(keep))
    def close(self):self.workspace=None


def fixture():
    eager=SimpleNamespace(capacity=262160,kv_dtype='int8',buffers={i:object() for i in range(1,9)})
    screen=object();slots=[Slot(eager,screen),Slot(eager,screen)];return SharedGraphTarget(slots)


class SharedScratchOwnership(unittest.TestCase):
    def test_two_private_states_share_buffers_and_isolated_release_reuse(self):
        t=fixture();t.warmup();a,b=t.state(),t.state()
        self.assertIsNot(a,b);self.assertEqual(t.capabilities['graph_captures'],32)
        self.assertFalse(t.capabilities['gpu_batch'])
        with self.assertRaisesRegex(RuntimeError,'slots full'):t.state()
        t.commit(a,None,2,2);self.assertEqual((a.pos,b.pos),(2,0))
        t.release_state(a);reused=t.state()
        self.assertIs(reused,a);self.assertEqual((reused.pos,b.pos),(0,0))
        self.assertIs(t.targets[0].eager.buffers,t.targets[1].eager.buffers)
        t.release_state(reused);t.release_state(b);t.close()

    def test_foreign_states_active_close_and_scratch_overlap_rejected(self):
        t=fixture();a=t.state()
        with self.assertRaisesRegex(RuntimeError,'foreign'):t.run(object(),[1])
        with self.assertRaisesRegex(RuntimeError,'release'):t.close()
        with self.assertRaisesRegex(RuntimeError,'idle'):t.warmup()
        t._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError,'scratch'):t.run(a,[1])
        finally:t._lock.release()
        t.release_state(a)
        with self.assertRaisesRegex(RuntimeError,'released'):t.run(a,[1])
        t.close()
        with self.assertRaisesRegex(RuntimeError,'closed'):t.state()

    def test_failed_release_fence_keeps_owner_until_success(self):
        t=fixture();a=t.state();t.targets[0].fence_fails=True
        with self.assertRaisesRegex(RuntimeError,'fence'):t.release_state(a)
        self.assertIn(id(a),t.owners);self.assertTrue(t.targets[0].leased)
        t.targets[0].fence_fails=False;t.release_state(a);t.close()

    def test_unshared_or_duplicate_graphs_refused(self):
        t=fixture();a,b=t.targets
        for targets in ([],[a,a],[a,b,a]):
            with self.assertRaises(ValueError):SharedGraphTarget(targets)
        b.screen=object()
        with self.assertRaisesRegex(ValueError,'share'):SharedGraphTarget([a,b])


if __name__=='__main__':unittest.main()
