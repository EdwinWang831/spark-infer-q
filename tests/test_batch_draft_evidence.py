from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from k4batch_draft_export import GROUPS,summarize


class BatchDraftEvidenceTests(unittest.TestCase):
    def rows(self):
        return [dict(group=group,mode=mode,repetition=rep,outputs_exact=True,new_jit_shapes=[],
            decode_ms=(100 if mode=='original' else 80)*(rep+1),decode_tokens=20,
            completion_tokens=22) for group in GROUPS for rep in range(5) for mode in ('original','batch_draft')]

    def test_paired_cohort_ratio_and_interval_use_all_repetitions(self):
        result=summarize(self.rows())
        for row in result.values():
            self.assertAlmostEqual(row['speed_gain'],.25)
            self.assertEqual(row['paired_repetition_speed_gain'],[.25]*5)
            self.assertEqual(row['paired_cohort_bootstrap_95'],[.25,.25])

    def test_missing_or_duplicated_replication_and_changed_work_are_rejected(self):
        rows=self.rows()
        with self.assertRaises(AssertionError):summarize(rows[:-1])
        changed=[dict(r) for r in rows];changed[-1]=dict(changed[-2])
        with self.assertRaises(AssertionError):summarize(changed)
        changed=[dict(r) for r in rows];changed[-1]['decode_tokens']+=1
        with self.assertRaises(AssertionError):summarize(changed)


if __name__=='__main__':unittest.main()
