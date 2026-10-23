"""Guard weighted throughput, paired work accounting and missing measurements."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('k4batch_export',Path(__file__).resolve().parents[1]/'scripts/k4batch_export.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class BatchStatisticsTests(unittest.TestCase):
    def rows(self):
        return [dict(repetition=r,joint=j,pair=p,full_ms=(1000 if p==0 else 5000)/(2 if j else 1)+1000,
            decode_ms=(1000 if p==0 else 5000)/(2 if j else 1),decode_tokens=10 if p==0 else 100,
            completion_tokens=11 if p==0 else 101,ttft_ms=[1,2],tpot_amortized_ms=[3,4],
            counters=dict(joint_forwards=int(j),actual_rows=20+p,physical_rows=24+p),policy_exact=True,new_head_jit_shapes=0)
            for r in range(3) for j in (False,True) for p in range(2)]
    def test_total_tokens_over_total_time_not_average_task_rates(self):
        result=module.summarize(self.rows())
        self.assertAlmostEqual(result['split']['decode_tps'],110/6)
        self.assertAlmostEqual(result['joint']['decode_tps'],110/3)
        self.assertAlmostEqual(result['split']['full_tps'],112/8)
    def test_duplicate_missing_and_unmatched_work_are_rejected(self):
        rows=self.rows()
        with self.assertRaises(AssertionError):module.summarize(rows[:-1])
        rows[-1]=rows[0]
        with self.assertRaises(AssertionError):module.summarize(rows)
        rows=self.rows();rows[2]['counters']['physical_rows']+=1
        with self.assertRaises(AssertionError):module.summarize(rows)
    def test_jit_or_changed_policy_cannot_enter_formal_denominator(self):
        for key,value in (('new_head_jit_shapes',1),('policy_exact',False)):
            rows=self.rows();rows[0][key]=value
            with self.assertRaises(AssertionError):module.summarize(rows)


if __name__=='__main__':unittest.main()
