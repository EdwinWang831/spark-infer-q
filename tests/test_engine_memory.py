"""Measured slot growth must gate additional allocations before they happen."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from spark_infer_q.engine_server import check_memory, warm_long_slots


class MemoryAdmissionTests(unittest.TestCase):
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
