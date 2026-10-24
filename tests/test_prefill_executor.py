"""Lifecycle and routing contracts; GPU tests supply arithmetic evidence."""
from types import SimpleNamespace
import unittest

from spark_infer_q.prefill import PrefillExecutor
from spark_infer_q.prefill_tensorfold import _DecodeMXFP8, _matrix_view


class Provider:
    capacity = 64
    vocab_size = 100
    max_chunk_rows = 32
    arithmetic = 'test_history'

    def __init__(self):
        self.stream = 1
        self.allocations = self.fences = 0
        self.calls = []
        self.fail = self.bad_position = self.bad_fence = False

    def stream_id(self): return self.stream
    def allocate(self, rows):
        self.allocations += 1
        return object()
    def concatenate(self, values): return sum(values, [])
    def synchronize(self):
        self.fences += 1
        if self.bad_fence: raise RuntimeError('fence failed')
    def run(self, workspace, state, ids, capture):
        self.calls.append((workspace, state.pos, ids, capture))
        values = []
        for token in ids:
            state.history.append(token)
            values.append(tuple(state.history))
        if not self.bad_position: state.pos += len(ids)
        if self.fail: raise RuntimeError('partial computation')
        return [sum(state.history)], values if capture else None


def state(): return SimpleNamespace(pos=0, capacity=64, history=[])


class PrefillExecutorTests(unittest.TestCase):
    def test_multichunk_tail_continuation_and_workspace_reuse(self):
        p = Provider()
        with PrefillExecutor(p, chunk_rows=8) as executor:
            s = state()
            a = executor.advance(s, range(19))
            b = executor.advance(s, [20, 21])
            self.assertEqual((a.chunks, a.start_position, a.end_position), ((8,8,3),0,19))
            self.assertEqual((b.start_position, b.end_position), (19,21))
            self.assertEqual(a.features[-1], tuple(range(19)))
            self.assertEqual(b.features[-1], tuple(range(19))+(20,21))
            self.assertEqual(a.logits, [sum(range(19))])
            other = executor.advance(state(), [9], capture=False)
            self.assertIsNone(other.features)
            self.assertEqual(other.logits, [9])
            self.assertEqual(p.allocations, 1)
            self.assertEqual(len({id(c[0]) for c in p.calls}), 1)
        self.assertIsNone(executor.workspace)
        with self.assertRaises(RuntimeError): executor.advance(state(), [1])
        executor.close()

    def test_invalid_input_never_runs_or_poisons(self):
        p = Provider()
        executor = PrefillExecutor(p, chunk_rows=8)
        for ids in ([], [True], [-1], [100], [1.5], [1]*65):
            s = state()
            with self.assertRaises(ValueError): executor.advance(s, ids)
            self.assertEqual((s.pos,s.history), (0,[]))
            self.assertFalse(executor.failed)
        with self.assertRaises(ValueError): executor.advance(state(), [1], capture=1)
        self.assertEqual(p.allocations, 0)
        for rows in (0, 33, True):
            with self.assertRaises(ValueError): PrefillExecutor(p, chunk_rows=rows)
        executor.close()

    def test_stream_and_concurrent_use_rejected_before_mutation(self):
        p = Provider(); executor = PrefillExecutor(p, chunk_rows=8)
        p.stream = 2
        with self.assertRaisesRegex(RuntimeError, 'stream'): executor.advance(state(), [1])
        p.stream = 1
        executor._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'already'): executor.advance(state(), [1])
            with self.assertRaisesRegex(RuntimeError, 'active'): executor.close()
        finally: executor._lock.release()
        self.assertEqual(p.allocations, 0)
        executor.close()

    def test_partial_failure_quarantines_and_fence_failure_retains_workspace(self):
        p = Provider(); executor = PrefillExecutor(p, chunk_rows=8)
        s = state(); p.fail = True
        with self.assertRaisesRegex(RuntimeError, 'partial'): executor.advance(s,[1])
        self.assertEqual(s.pos,1)
        self.assertTrue(executor.failed)
        with self.assertRaisesRegex(RuntimeError, 'quarantined'): executor.advance(state(),[2])
        p.bad_fence = True
        with self.assertRaisesRegex(RuntimeError, 'fence'): executor.close()
        self.assertIsNotNone(executor.workspace)
        self.assertFalse(executor.closed)
        p.bad_fence = False; executor.close()
        self.assertIsNone(executor.workspace)

    def test_provider_commit_contract_violation_is_fatal(self):
        p = Provider(); p.bad_position = True
        with PrefillExecutor(p, chunk_rows=8) as executor:
            with self.assertRaisesRegex(RuntimeError, 'commit'): executor.advance(state(), [1])
            self.assertTrue(executor.failed)

    def test_close_fence_failure_also_quarantines_a_previously_healthy_executor(self):
        p=Provider();executor=PrefillExecutor(p,chunk_rows=8)
        executor.advance(state(),[1])
        p.bad_fence=True
        with self.assertRaisesRegex(RuntimeError,'fence'):executor.close()
        self.assertFalse(executor.closed)
        self.assertTrue(executor.failed)
        self.assertIsNotNone(executor.workspace)
        with self.assertRaisesRegex(RuntimeError,'quarantined'):executor.advance(state(),[2])
        p.bad_fence=False;executor.close()

    def test_matrix_view_changes_dispatch_without_touching_shared_weights(self):
        class Matrix:
            kernel = 'mx8'
            def __init__(self): self.weight = object()
            def __call__(self,x,out=None): return ('decode',x,out)
            def prefill(self,x,out=None): return ('prompt',x,out)
        original = SimpleNamespace(proj=Matrix(), other=object())
        view = _matrix_view(original, ('proj','other'))
        self.assertIsInstance(view.proj, _DecodeMXFP8)
        self.assertEqual(view.proj.prefill(3,4), ('decode',3,4))
        self.assertEqual(original.proj.prefill(3,4), ('prompt',3,4))
        self.assertIs(view.proj.weight,original.proj.weight)
        self.assertIs(view.other,original.other)


if __name__ == '__main__': unittest.main()
