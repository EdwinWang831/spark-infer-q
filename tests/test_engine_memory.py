"""Measured slot growth must gate additional allocations before they happen."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from spark_infer_q.engine_server import check_memory, warm_long_slots


class MemoryAdmissionTests(unittest.TestCase):
    def test_budget_charges_actual_mirrored_feature_storage(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('optional NumPy absent')
        from test_rolling_history import Rows
        from spark_infer_q.feature_window import FeatureWindow
        window=FeatureWindow(2040);window.append(Rows(np.zeros((1,12800),dtype=np.float16)))
        arrays={name:Rows(np.zeros(n,dtype=np.float16)) for name,n in
            (('_context_rows',7),('_keys',8),('_values',9))}
        provider=NS(history_window_rows=2040,worker=NS(**arrays))
        target,snapshot,_=self.fixture(70,1)
        with patch('spark_infer_q.engine_server.memory_snapshot',snapshot):
            result=warm_long_slots(target,provider,None)
        self.assertEqual(result['budget']['slot_bytes'],1024**3+window.nbytes+
                         sum(value.data.nbytes for value in arrays.values()))

    def test_64k_bootstrap_refuses_before_allocating_into_its_reserve(self):
        target,snapshot,calls=self.fixture(86,1);target.capacity=65552
        with patch('spark_infer_q.engine_server.memory_snapshot',snapshot):
            with self.assertRaisesRegex(RuntimeError,'first request slot'):
                warm_long_slots(target,None,None)
        self.assertEqual(calls,[])

    def test_128k_bootstrap_charges_larger_state_before_first_allocation(self):
        target,snapshot,calls=self.fixture(85,1);target.capacity=131088
        with patch('spark_infer_q.engine_server.memory_snapshot',snapshot):
            with self.assertRaisesRegex(RuntimeError,'first request slot'):
                warm_long_slots(target,None,None)
        self.assertEqual(calls,[])

    def fixture(self,base,private):
        unit=1024**3
        usage=[base*unit];calls=[]
        def snapshot(_):
            return dict(allocated_bytes=usage[0],reserved_bytes=usage[0],
                peak_allocated_bytes=usage[0],peak_reserved_bytes=usage[0],host_available_bytes=40*unit)
        def warmup():usage[0]+=private*unit;calls.append(usage[0])
        target=NS(targets=[NS(warmup=warmup) for _ in range(4)])
        return target,snapshot,calls

    def test_four_slots_are_charged_with_transient_reserve(self):
        target,snapshot,calls=self.fixture(70,1)
        with patch('spark_infer_q.engine_server.memory_snapshot',snapshot):
            result=warm_long_slots(target,None,None)
        self.assertEqual(len(calls),4)
        self.assertEqual(result['admission']['forecast_peak_bytes'],75*1024**3)
        self.assertTrue(result['admission']['admission_fits'])
        self.assertFalse(result['admission']['execution_validated'])

    def test_forecast_refuses_remaining_slots_before_second_allocation(self):
        target,snapshot,calls=self.fixture(82,2)
        with patch('spark_infer_q.engine_server.memory_snapshot',snapshot):
            with self.assertRaisesRegex(RuntimeError,'admission refused'):
                warm_long_slots(target,None,None)
        self.assertEqual(len(calls),1)

    def test_host_reserve_and_reserved_memory_are_independent_gates(self):
        base=dict(allocated_bytes=70*1024**3,reserved_bytes=70*1024**3,
                  peak_allocated_bytes=70*1024**3,peak_reserved_bytes=70*1024**3,
                  host_available_bytes=10*1024**3)
        for changes in (dict(reserved_bytes=89*1024**3),dict(host_available_bytes=7*1024**3)):
            with patch('spark_infer_q.engine_server.memory_snapshot',return_value=dict(base,**changes)):
                with self.assertRaisesRegex(RuntimeError,'reserve exceeded'):check_memory(None)


if __name__=='__main__':unittest.main()
