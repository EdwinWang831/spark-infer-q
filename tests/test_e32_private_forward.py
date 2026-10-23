import sys
import types
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from e32_probe_v2 import clone_module,private_method_proxy


class PrivateForwardTests(unittest.TestCase):
    def test_instrumented_nested_calls_leave_original_globals_unchanged(self):
        source=types.ModuleType('e32_fixture')
        exec('def leaf(x, *, bias=2):\n    return x+bias\ndef forward(x):\n    return leaf(x)*3\n',source.__dict__)
        private=clone_module(source)
        seen=[];leaf=private.leaf
        def observe(x,**kwargs):
            seen.append(x)
            return leaf(x,**kwargs)
        private.leaf=observe
        self.assertEqual(private.forward(5),21)
        self.assertEqual(seen,[5])
        self.assertEqual(source.forward(5),21)
        self.assertEqual(seen,[5])
        self.assertIs(source.forward.__globals__,source.__dict__)
        self.assertIs(private.forward.__globals__,private.__dict__)
        self.assertEqual(leaf(1),3)

    def test_shared_expert_class_method_is_observed_through_private_proxy(self):
        source=types.ModuleType('e32_shared_fixture')
        exec('def leaf(x):\n    return x+1\nclass Expert:\n    def shared_act(self,x,prompt=False):\n        return leaf(x)\ndef forward(ex,x):\n    return ex.shared_act(x)\n',source.__dict__)
        private=clone_module(source);seen=[];leaf=private.leaf
        def observe(x):
            seen.append(x)
            return leaf(x)
        private.leaf=observe
        expert=source.Expert()
        self.assertEqual(private.forward(expert,5),6)
        self.assertEqual(seen,[]) # a copied module alone misses original class-method globals
        proxy=private_method_proxy(expert,'shared_act',private.__dict__)
        self.assertEqual(private.forward(proxy,5),6)
        self.assertEqual(seen,[5])
        self.assertEqual(source.forward(expert,5),6)
        self.assertEqual(seen,[5])
        self.assertNotIn('shared_act',expert.__dict__)
        self.assertIs(type(proxy),type(expert))


if __name__=='__main__':unittest.main()
