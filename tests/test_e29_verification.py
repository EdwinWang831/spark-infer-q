import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from e29_online import MODES,order,summarize
from e29_analysis import curve_for,summarize as summarize_prefixes


class VerificationAccountingTests(unittest.TestCase):
    def test_each_pair_has_balanced_order_within_every_prompt(self):
        for ti in range(8):
            for a in MODES:
                for b in MODES:
                    if a!=b:
                        self.assertEqual(sum(order(ti,r).index(a)<order(ti,r).index(b) for r in range(4)),2)

    def test_actual_tokens_and_incomplete_run_rejection(self):
        tasks=[]
        for i in range(8):
            runs=[]
            for rep in range(4):
                for mode,ms in zip(MODES,(1000.,900.,800.)):
                    steps=[dict(emitted=2,width=4,accepted=1,observed=2,draft_budget_ms=2.,verify_ms=15.,commit_ms=1.) for _ in range(50)]
                    runs.append(dict(mode=mode,repetition=rep,exact=True,decode_ms=ms,steps=steps))
            tasks.append(dict(group=str(i),completion_tokens=101,timings=runs))
        summary=summarize(tasks,samples=100)
        self.assertEqual(summary['timed_tokens_per_arm'],3200)
        self.assertEqual(summary['trace']['eager_bf16']['emitted'],3200)
        self.assertEqual(summary['decode_tps']['eager_bf16'],100.)
        self.assertAlmostEqual(summary['versus_eager_bf16']['graph_w8']['relative_gain'],.25)
        tasks[0]['timings'].pop()
        with self.assertRaises(AssertionError):summarize(tasks,samples=100)

    def test_paid_draft_cost_includes_anchor_only_and_oracle_is_diagnostic(self):
        samples=[]
        for n in (64,768):
            arms={}
            for mode in ('bf16','w8'):
                actions=[dict(emitted=min(w,3),timings=[dict(verify_ms=10.+w,commit_ms=1.)]*3) for w in range(1,9)]
                arms[mode]=dict(actions=actions,rates=[.9,.8,.1,.01,.001,.0001,.00001],paid_ms=5.)
            samples.append(dict(context_rows=n,modes=arms))
        self.assertEqual(curve_for(samples,'bf16',0)[0],12.)
        result=summarize_prefixes(samples)
        for mode in ('bf16','w8'):
            replay=result[mode]['same_prefix_replay']
            self.assertAlmostEqual(replay['fixed'],3./24.)
            self.assertGreaterEqual(replay['oracle'],replay['policy'])
        with self.assertRaises(ValueError):curve_for(samples[:1],'bf16',1000)


if __name__=='__main__':unittest.main()
