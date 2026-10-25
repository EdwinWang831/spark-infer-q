"""Bounded ownership/absolute-position tests; CPU fixtures are not long-model evidence."""
from contextlib import nullcontext
from types import SimpleNamespace
import math
import unittest
from unittest.mock import Mock,patch
import numpy as np

from spark_infer_q.feature_window import FeatureWindow
from spark_infer_q.rolling_draft import RollingContextDraft
from spark_infer_q.kaifra_speculative import DFlashProvider,SpeculativeBackend


class Rows:
    def __init__(self,data,device='cpu'):
        self.data=np.asarray(data);self.device=device
    @property
    def ndim(self):return self.data.ndim
    @property
    def shape(self):return self.data.shape
    @property
    def dtype(self):return self.data.dtype
    def __len__(self):return len(self.data)
    def __getitem__(self,key):return Rows(self.data[key],self.device)
    def new_empty(self,shape):return Rows(np.empty(shape,dtype=self.dtype),self.device)
    def copy_(self,source):np.copyto(self.data,source.data);return self
    def clone(self):return Rows(self.data.copy(),self.device)
    def numel(self):return self.data.size
    def element_size(self):return self.data.itemsize


class RollingHistoryTests(unittest.TestCase):
    def test_wrap_tail_large_append_and_exact_storage_bound(self):
        for capacity in (1,3,8,31):
            window=FeatureWindow(capacity);history=[];end=0
            for count in (1,capacity-1 or 1,3,capacity,capacity+1,2*capacity+3,1,7):
                values=list(range(end,end+count));end+=count;history+=values
                got=window.append(Rows(np.asarray(values,dtype=np.int32)[:,None]))
                np.testing.assert_array_equal(got.data[:,0],history[-capacity:])
                self.assertEqual((window.start,window.end),(max(0,end-capacity),end))
                self.assertEqual(window.nbytes,2*capacity*4)

    def test_million_logical_rows_do_not_grow_storage(self):
        window=FeatureWindow(32)
        for start in range(0,1048576,1024):
            value=window.append(Rows(np.arange(start,start+1024,dtype=np.int32)[:,None]))
        self.assertEqual((window.start,window.end,window.nbytes),(1048576-32,1048576,256))
        np.testing.assert_array_equal(value.data[:,0],np.arange(1048576-32,1048576))

    def test_invalid_layout_does_not_advance_or_overwrite(self):
        for capacity in (0,-1,True,1.5):
            with self.assertRaises(ValueError):FeatureWindow(capacity)
        window=FeatureWindow(4);window.append(Rows(np.arange(6,dtype=np.int32).reshape(3,2)));gold=window.view().clone()
        for value in (Rows(np.empty((0,2),dtype=np.int32)),Rows(np.zeros((1,3),dtype=np.int32)),
                      Rows(np.zeros((1,2),dtype=np.float32)),Rows(np.zeros((1,2),dtype=np.int32),'other')):
            with self.assertRaises(ValueError):window.append(value)
            self.assertEqual((window.start,window.end),(0,3));np.testing.assert_array_equal(window.view().data,gold.data)

    def test_rolling_projection_reuses_suffix_and_rejects_rewind(self):
        worker=object.__new__(RollingContextDraft);worker.window_start=0;worker.request=None;worker.cached_rows=5
        worker._context_rows=Rows(np.arange(16).reshape(8,2));worker._keys=Rows(np.arange(32).reshape(2,8,2));worker._values=worker._keys.clone()
        with self.assertRaises(ValueError):worker.set_window(0)
        worker.request='a';original=worker._keys.clone();worker.set_window(2)
        self.assertEqual((worker.window_start,worker.cached_rows),(2,3))
        np.testing.assert_array_equal(worker._keys.data[:,:3],original.data[:,2:5])
        for pos in (1,True,2.5):
            with self.assertRaises(ValueError):worker.set_window(pos)
        worker.set_window(100);self.assertEqual(worker.cached_rows,0)
        worker.begin('b');self.assertEqual((worker.window_start,worker.cached_rows,worker.request),(0,0,'b'))

    def test_rotary_positions_use_absolute_start(self):
        worker=object.__new__(RollingContextDraft);worker.window_start=4096
        with patch('spark_infer_q.dflash2_reference.Draft.rope',side_effect=lambda x,p:p):
            np.testing.assert_array_equal(worker.rope(None,np.array([0,3,2040])),[4096,4099,6136])

    def test_provider_global_features_costs_and_window_contract(self):
        worker=SimpleNamespace(max_context=2048,block_size=8,request=None,set_window=Mock(),
            begin=Mock(),propose=Mock(return_value=([1]*7,dict(signals=[0.]*133))))
        model=SimpleNamespace(predict=Mock(return_value=[.8,.6,.4,.3,.2,.1,.05]))
        budget=SimpleNamespace(choose=Mock(return_value=4))
        provider=DFlashProvider(worker,model=model,budget=budget,history_window_rows=2040,max_request_context=4608)
        state=SimpleNamespace(context=[0]*5,context_start=4091,output_tokens=3)
        provider.propose('req',state,7,8)
        worker.set_window.assert_called_once_with(4091)
        self.assertAlmostEqual(model.predict.call_args.args[0][21],math.log1p(4096))
        self.assertEqual(budget.choose.call_args.args[1],4096)
        self.assertEqual((provider.max_context,provider.history_window_rows,state.next_width),(4608,2040,4))
        for kw in (dict(max_request_context=4608),dict(history_window_rows=2041),dict(history_window_rows=True)):
            with self.assertRaises(ValueError):DFlashProvider(worker,**kw)

    def test_backend_per_request_window_and_release_ownership(self):
        target=SimpleNamespace(buffers={i:None for i in range(1,9)},torch=SimpleNamespace(inference_mode=nullcontext),state=lambda:SimpleNamespace(pos=0))
        backend=SpeculativeBackend(target,SimpleNamespace(history_window_rows=4),eos_ids=(9,),depth=7,prior=[.5]*7)
        a,b=backend.create('a'),backend.create('b')
        backend._append_context(a,Rows(np.arange(7)[:,None]));backend._append_context(b,Rows(np.array([[99]])))
        self.assertEqual((a.context_start,len(a.context),b.context_start,len(b.context)),(3,4,0,1))
        np.testing.assert_array_equal(a.context.data[:,0],[3,4,5,6]);np.testing.assert_array_equal(b.context.data[:,0],[99])
        backend.release('a',a);self.assertIsNone(a.context);self.assertIsNone(a.feature_window)
        np.testing.assert_array_equal(b.context.data[:,0],[99]);backend.release('b',b)


if __name__=='__main__':unittest.main()
