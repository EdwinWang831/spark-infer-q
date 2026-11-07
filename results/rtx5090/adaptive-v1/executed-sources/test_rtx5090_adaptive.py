from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rtx5090_adaptive_policy import Controller


class AdaptiveTests(unittest.TestCase):
    def test_skip_does_not_feed_history_and_forces_recovery(self):
        policy = Controller({'kind': 'candidate', 'short_threshold': 4, 'skip_patience': 1})
        self.assertEqual(policy.choose(), 16)
        policy.observe(16, 1)
        self.assertEqual(policy.choose(), 1)
        policy.observe(1, 1)
        self.assertEqual((policy.last, policy.ewma, policy.low_streak), (1, 1, 0))
        self.assertEqual(policy.choose(), 16)
        policy.observe(16, 6)
        self.assertEqual(policy.choose(), 16)

    def test_two_low_drafted_rounds_trigger_skip(self):
        policy = Controller({'kind': 'candidate', 'short_threshold': 2, 'skip_patience': 2})
        self.assertEqual(policy.choose(), 16)
        policy.observe(16, 1)
        self.assertEqual(policy.choose(), 8)
        policy.observe(8, 1)
        self.assertEqual(policy.choose(), 1)

    def test_lagged_and_ewma_use_only_completed_rounds(self):
        lagged, ewma = Controller({'kind': 'lagged'}), Controller({'kind': 'ewma'})
        for policy in (lagged, ewma):
            self.assertEqual(policy.choose(), 16)
            policy.observe(16, 8)
            self.assertEqual(policy.choose(), 16)
            policy.observe(16, 2)
        self.assertEqual(lagged.choose(), 8)
        self.assertEqual(ewma.choose(), 16)


if __name__ == '__main__':
    unittest.main()
