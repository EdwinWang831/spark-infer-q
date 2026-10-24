import gc
from types import ModuleType
import unittest
import weakref

from scripts.e38_cleanup import detach


def fixture(closed=True):
    name='_e38_component_test_forward'
    ns=dict(__name__=name,_e36_closed=closed)
    exec('def moe_block(): return 1',ns);base=ns['moe_block']
    def wrapper(block):
        def observed_block():return block()
        observed_block.__qualname__='component.<locals>.observed_block'
        return observed_block
    ns['moe_block']=wrapper(base)
    observed=ModuleType('_e38_component_test_observed')
    class Payload:pass
    payload=Payload();ref=weakref.ref(payload)
    def observer(p):return lambda:p
    observed.moe=observer(payload);ns['nvfp4_moe']=observed
    return ns,name,base,ref


class CleanupTests(unittest.TestCase):
    def test_owned_closed_observer_releases_payload_without_touching_upstream(self):
        ns,name,base,payload=fixture();original=ModuleType('original');original.moe=lambda:2
        self.assertIsNotNone(payload())
        detach(ns,original,name);gc.collect()
        self.assertIsNone(payload())
        self.assertIs(ns['moe_block'],base);self.assertIs(ns['nvfp4_moe'],original)
        self.assertEqual(original.moe(),2)

    def test_active_or_foreign_forward_is_rejected(self):
        ns,name,base,ref=fixture(False)
        with self.assertRaises(ValueError):detach(ns,ModuleType('original'),name)
        self.assertIsNotNone(ref())
        ns['_e36_closed']=True
        with self.assertRaises(ValueError):detach(ns,ModuleType('original'),'foreign')


if __name__=='__main__':unittest.main()
