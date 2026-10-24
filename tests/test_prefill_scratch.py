from types import SimpleNamespace
import unittest
from spark_infer_q.prefill_scratch import PrefillScratchPool


class FakeStorage:
    def __getitem__(self,key):return (self,key.stop)


class PrefillScratchTests(unittest.TestCase):
    def test_one_storage_and_stable_views_for_large_small_large(self):
        calls=[]
        def factory(n):
            calls.append(n)
            return SimpleNamespace(**{k:FakeStorage() for k in ('k','v','g','b')})
        pool=PrefillScratchPool(factory,lambda:7,512,16,48)
        large=pool.prepare(512);small=pool.prepare(64)
        self.assertIs(pool.get(512,16,48),large)
        self.assertIs(pool.get(64,16,48),small)
        self.assertIs(pool.get(512,16,48),large)
        self.assertEqual(calls,[512]);self.assertEqual(pool.calls,3)
        self.assertIs(large.k[0],small.k[0]);self.assertEqual(small.k[1],64)

    def test_reject_unprepared_size_dimensions_and_foreign_stream_owner(self):
        ids=dict(stream=7,owner=11)
        pool=PrefillScratchPool(lambda n:SimpleNamespace(**{k:FakeStorage() for k in ('k','v','g','b')}),
            lambda:ids['stream'],512,16,48,owner_id=lambda:ids['owner'])
        pool.prepare(64)
        for n,nk,nv in [(513,16,48),(0,16,48),(True,16,48),(128,16,48),(64,8,24)]:
            with self.subTest(n=n,nk=nk),self.assertRaises(ValueError):pool.get(n,nk,nv)
        ids['stream']=8
        with self.assertRaises(RuntimeError):pool.get(64,16,48)
        ids['stream']=7;ids['owner']=12
        with self.assertRaises(RuntimeError):pool.get(64,16,48)
