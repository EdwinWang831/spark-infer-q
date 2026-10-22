import unittest
from spark_infer_q.draft_signals import observed_labels


class DraftSignalTests(unittest.TestCase):
    def test_rejection_censors_later_positions(self):
        self.assertEqual(observed_labels([5,6,7],[5,9,7,8],2),[(0,1),(1,0)])

    def test_matched_eos_is_success_and_censors_suffix(self):
        self.assertEqual(observed_labels([5,9,7],[5,9,7,8],2,eos_ids={9}),[(0,1),(1,1)])

    def test_output_cap_and_missing_tail_are_not_failures(self):
        self.assertEqual(observed_labels([5,6,7],[5,6],2),[(0,1),(1,1)])
        self.assertEqual(observed_labels([5,6,7],[5,6,7,8],1),[(0,1)])
        self.assertEqual(observed_labels([], [4],1),[])


if __name__ == '__main__':
    unittest.main()
