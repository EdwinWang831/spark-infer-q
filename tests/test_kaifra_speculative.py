import math
from types import SimpleNamespace
import unittest

from spark_infer_q.kaifra_speculative import conditional_rates,DFlashProvider,PostDraftBudget,speculative_engine
from spark_infer_q.native_scheduler import NativeScheduler


class SpeculativeContracts(unittest.TestCase):
    def test_conditional_publication_preserves_cumulative_rates(self):
        cumulative=[.8,.4,.1,0.,0.,0.,0.]
        conditional=conditional_rates(cumulative)
        self.assertEqual(conditional,[.8,.5,.25,0.,0.,0.,0.])
        actual=[];p=1.
        for value in conditional:p*=value;actual.append(p)
        self.assertEqual(actual,cumulative)
        for rates in ([.2,.3],[float('nan')],[-.1],[1.1],[.5,float('inf')]):
            with self.assertRaises(ValueError):conditional_rates(rates)

    def test_current_budget_uses_current_rates_remaining_and_context_costs(self):
        class Probe:
            def curve(self,values):self.values=values
            def schedule(self,params,entries):
                self.entries=entries
                return [len(entries[0]['draft_tokens'])+1]
        costs=lambda base:dict(native_draft_ms=2.,feature_increment_ms=.2,
                              verify_commit_ms=[base+i for i in range(8)])
        budget=PostDraftBudget(dict(costs=dict(short=costs(1),long=costs(10)),cpu_predict_ms=.1),
                               SimpleNamespace(BudgetProbe=Probe))
        rates=[.8,.4,.2,.1,.05,.02,.01]
        for context,remaining,first in ((767,3,3.3),(768,8,12.3)):
            self.assertEqual(budget.choose(rates,context,remaining),remaining)
            self.assertAlmostEqual(budget.probe.values[0][1],first)
            entry=budget.probe.entries[0]
            self.assertEqual(entry['accept_rates'],rates[:remaining-1])
            self.assertEqual(entry['prev_accept_rates'],entry['accept_rates'])
            self.assertEqual(entry['process_cnt'],entry['schedule_cnt'])
        self.assertEqual(budget.choose(rates,768,1),1)

    def test_context_depth_and_capacity_rejected_before_native_dispatch(self):
        for value in (-1,8,True,1.5):
            with self.assertRaises(ValueError):NativeScheduler('/unused','/unused',max_drafts=value)
        with self.assertRaises(ValueError):
            speculative_engine(SimpleNamespace(max_drafts=0),None,None,eos_ids={9},prior=[.5]*7)

    def test_retired_fixed_modes_rejected_before_accessing_model(self):
        native=SimpleNamespace(max_drafts=7)
        for mode in ('fixed1','fixed3','fixed7'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError,'unsupported speculative mode'):
                speculative_engine(native,None,None,eos_ids={9},prior=[.5]*7,mode=mode)

    def test_projection_cache_changes_owner_on_interleaving(self):
        class Worker:
            max_context=256
            request=None
            def begin(self,h):self.request=h
            def propose(self,context,anchor):return [anchor+1]*7,{}
        worker=Worker();provider=DFlashProvider(worker)
        for handle in ('a','b','a'):
            tokens,rates=provider.propose(handle,SimpleNamespace(context='committed'),1,10)
            self.assertEqual(worker.request,handle)
            self.assertEqual(tokens,[2]*7);self.assertIsNone(rates)


if __name__=='__main__':unittest.main()
