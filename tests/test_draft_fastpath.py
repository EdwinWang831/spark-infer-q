import itertools
import unittest
from spark_infer_q.draft_fastpath import before_eos
from spark_infer_q.speculative import greedy_result


class EosPrefixTests(unittest.TestCase):
    def test_exhaustive_greedy_output_and_commit_contract(self):
        # Every accept/reject/EOS pattern for up to three proposals, including
        # EOS in a correction/bonus and output-length caps.
        for count in range(4):
            for drafts in itertools.product(range(3),repeat=count):
                cropped=before_eos(drafts,{2})
                for predictions in itertools.product(range(3),repeat=count+1):
                    for limit in (1,2,3,4,None):
                        full=greedy_result(0,drafts,predictions,eos_ids={2},max_new=limit)
                        short=greedy_result(0,cropped,predictions[:len(cropped)+1],eos_ids={2},max_new=limit)
                        self.assertEqual(full,short,(drafts,predictions,limit))
