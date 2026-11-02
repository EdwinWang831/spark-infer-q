import unittest
from spark_infer_q.long_capacity import LongCapacityBudget


class Capacity(unittest.TestCase):
    def test_cap_boundary_and_transient_are_charged(self):
        budget=LongCapacityBudget(70,4,2,88,8)
        snapshot=dict(allocated_bytes=74,reserved_bytes=80,host_available_bytes=40)
        self.assertEqual(budget.require(4,**snapshot)['forecast_peak_bytes'],88)
        self.assertEqual(budget.evaluate(5,**snapshot)['process_deficit_bytes'],4)
        with self.assertRaisesRegex(RuntimeError,'refused'):budget.require(5,**snapshot)

    def test_reserved_slack_is_not_added_to_host_ram(self):
        budget=LongCapacityBudget(70,4,2,120,8)
        snap=dict(allocated_bytes=74,reserved_bytes=100,host_available_bytes=10)
        result=budget.evaluate(2,**snap)
        self.assertTrue(result['process_limit_fits'])
        self.assertEqual(result['conservative_host_remaining_bytes'],4)
        self.assertFalse(result['admission_fits'])
        self.assertFalse(result['execution_validated'])

    def test_bad_snapshots_and_boolean_sizes_rejected(self):
        with self.assertRaises(ValueError):LongCapacityBudget(True,4,2,88,8)
        budget=LongCapacityBudget(70,4,2,88,8)
        for n in (0,9,True):
            with self.assertRaises(ValueError):budget.evaluate(n,allocated_bytes=1,reserved_bytes=1,host_available_bytes=40)
        with self.assertRaises(ValueError):budget.evaluate(1,allocated_bytes=2,reserved_bytes=1,host_available_bytes=40)
