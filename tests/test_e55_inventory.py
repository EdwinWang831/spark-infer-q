"""Alias and callable-owner traversal; fixture storage is not GPU evidence."""
from types import SimpleNamespace
import unittest
from scripts.e55_inventory import inventory


class Tensor:
    is_cuda=True
    device='cuda:0'
    dtype='fixture.uint8'
    shape=(16,)
    def __init__(self,pointer,size):self.pointer,self.size=pointer,size
    def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:self.pointer,nbytes=lambda:self.size)


class CallableMatrix:
    def __init__(self,tensor):self.weight=tensor
    def __call__(self,*args):raise AssertionError('inventory must not execute matrices')


class Inventory(unittest.TestCase):
    def test_callable_matrices_are_not_confused_with_functions_and_aliases_count_once(self):
        a,b=Tensor(10,16),Tensor(20,32)
        left=SimpleNamespace(matrix=CallableMatrix(a),aliases=[a,a])
        right=SimpleNamespace(shared=a,exclusive=b,function=lambda:a)
        value=inventory(SimpleNamespace(Tensor=Tensor),dict(left=left,right=right))
        self.assertEqual(value['unique_bytes'],48)
        self.assertEqual(value['storage_count'],2)
        self.assertEqual(value['per_root']['left'],dict(reachable_bytes=16,exclusive_bytes=0))
        self.assertEqual(value['per_root']['right'],dict(reachable_bytes=48,exclusive_bytes=32))
        owners=next(x['owners'] for x in value['storages'] if x['bytes']==16)
        self.assertIn('left.matrix.weight',owners['left'])

    def test_cycles_views_and_cpu_tensors_do_not_inflate_cuda_bytes(self):
        a=Tensor(10,64);view=Tensor(10,64);cpu=Tensor(20,128);cpu.is_cuda=False
        root=SimpleNamespace(a=a,view=view,cpu=cpu);root.self=root
        value=inventory(SimpleNamespace(Tensor=Tensor),dict(root=root))
        self.assertEqual((value['unique_bytes'],value['storage_count']),(64,1))
