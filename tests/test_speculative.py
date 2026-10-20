import unittest
from spark_infer_q.speculative import greedy_result


class VerificationTests(unittest.TestCase):
    def test_each_rejection_and_full_acceptance(self):
        for accepted in range(4):
            drafts=[11,12,13]
            predictions=[11,12,13,14]
            if accepted<3: drafts[accepted]=99
            outputs,keep,n,stopped=greedy_result(10,drafts,predictions)
            self.assertEqual(outputs,[11,12,13,14][:accepted+1])
            self.assertEqual((keep,n,stopped),(accepted+1,accepted,False))

    def test_no_draft(self):
        self.assertEqual(greedy_result(10,[],[11]),([11],1,0,False))

    def test_eos_and_length_do_not_commit_pending_or_suffix(self):
        self.assertEqual(greedy_result(10,[11,12,13],[11,12,13,14],eos_ids={12}),([11,12],2,1,True))
        self.assertEqual(greedy_result(10,[11,12,13],[11,12,13,14],max_new=1),([11],1,0,False))

    def test_eos_beyond_length_limit_is_not_reported_as_emitted(self):
        self.assertEqual(greedy_result(10,[11,12,13],[11,12,13,14],eos_ids={13},max_new=1),([11],1,0,False))

    def test_invalid_layout(self):
        with self.assertRaises(ValueError): greedy_result(10,[11],[11])
