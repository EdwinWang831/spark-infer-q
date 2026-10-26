"""Real CPU arrays test buffer binding; these tests do not prove CUDA parity."""
from types import SimpleNamespace
import unittest
import numpy as np

from spark_infer_q.draft_projection_cache import DraftProjectionCache


class Buffer(np.ndarray):
    def numel(self):return self.size
    def element_size(self):return self.itemsize


def worker():
    return SimpleNamespace(torch=SimpleNamespace(empty_like=np.empty_like),request=None,
        _context_rows=np.zeros((4,3),dtype=np.float32).view(Buffer),
        _keys=np.zeros((2,4,2),dtype=np.float32).view(Buffer),
        _values=np.zeros((2,4,2),dtype=np.float32).view(Buffer),
        cached_rows=0,appended_rows=0,reused_rows=0,window_start=0)


class ProjectionCache(unittest.TestCase):
    def test_requests_keep_private_buffers_and_counters_across_switches(self):
        w=worker();base=w._keys;a=DraftProjectionCache(w,'a');b=DraftProjectionCache(w,'b')
        for cache,key,value in ((a,'a',3),(b,'b',7),(a,'a',11)):
            with cache.bind(w,key):
                self.assertIs(w._keys,cache.tensors['_keys'])
                self.assertEqual(w.cached_rows,cache.counters['cached_rows'])
                if key=='a' and value==11:self.assertTrue(np.all(w._keys==3))
                w._keys.fill(value);w.cached_rows=value;w.window_start=value+100
            self.assertIs(w._keys,base);self.assertIsNone(w.request)
            self.assertEqual(w.cached_rows,0);self.assertEqual(w.window_start,0)
        self.assertTrue(np.all(b.tensors['_keys']==7));self.assertEqual(a.counters['cached_rows'],11)
        self.assertEqual(a.nbytes,176);a.close();self.assertEqual(a.nbytes,0)
        with self.assertRaisesRegex(RuntimeError,'released'):
            with a.bind(w,'a'):pass

    def test_exception_restores_worker_and_nested_or_foreign_binding_is_refused(self):
        w=worker();original=dict(vars(w));a=DraftProjectionCache(w,'a');b=DraftProjectionCache(w,'b')
        with self.assertRaisesRegex(ValueError,'probe failure'):
            with a.bind(w,'a'):
                w.cached_rows=2
                for cache,owner in ((a,'a'),(b,'b')):
                    with self.assertRaisesRegex(RuntimeError,'in use'):
                        with cache.bind(w,owner):pass
                with self.assertRaisesRegex(RuntimeError,'active'):a.close()
                raise ValueError('probe failure')
        for key,value in original.items():self.assertIs(getattr(w,key),value)
        self.assertEqual(a.counters['cached_rows'],2)
        for other,owner in ((worker(),'a'),(w,'b')):
            with self.assertRaisesRegex(RuntimeError,'foreign'):
                with a.bind(other,owner):pass
        a.close();b.close()
