import math
from contextlib import nullcontext
from types import SimpleNamespace
import unittest

from spark_infer_q.kaifra_speculative import conditional_rates,DFlashProvider,PostDraftBudget,speculative_engine,SpeculativeBackend,SpeculativeSchedulerBridge
from spark_infer_q.deepinfer_bridge import RequestHandle,WorkRow
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

    def test_serialized_postdraft_rejects_unowned_or_batched_slots_before_dispatch(self):
        backend=SimpleNamespace(target=SimpleNamespace(capabilities={}),provider=SimpleNamespace(deferred_draft=True))
        with self.assertRaisesRegex(ValueError,'private shared-scratch'):
            SpeculativeSchedulerBridge(None,None,None,None,backend,postdraft=True,
                serialized_postdraft=True,max_requests=2)
        backend.target.capabilities=dict(max_requests=2,gpu_batch=False,
            graph_scope='private_request_Graphs_shared_serial_scratch')
        for kwargs in (dict(max_requests=3),dict(max_requests=2,curve=[(1,1)]),dict(max_requests=1)):
            with self.assertRaisesRegex(ValueError,'private shared-scratch'):
                SpeculativeSchedulerBridge(None,None,None,None,backend,postdraft=True,
                    serialized_postdraft=True,**kwargs)
        backend.target.capabilities['gpu_batch']=True
        with self.assertRaisesRegex(ValueError,'private shared-scratch'):
            SpeculativeSchedulerBridge(None,None,None,None,backend,postdraft=True,
                serialized_postdraft=True,max_requests=2)

    def test_owned_prefill_keeps_all_draft_features_across_scheduler_chunks(self):
        calls=[]
        class Logits:
            def __getitem__(self,key): return self
            def argmax(self): return 7
        def advance(state,ids,capture):
            self.assertTrue(capture)
            features=list(range(state.pos,state.pos+len(ids)))
            calls.append(tuple(ids));state.pos+=len(ids)
            return SimpleNamespace(logits=Logits(),features=features)
        target=SimpleNamespace(buffers={n:None for n in range(1,9)},
            torch=SimpleNamespace(inference_mode=nullcontext,cat=lambda xs:sum(xs,[])),
            state=lambda:SimpleNamespace(pos=0))
        executor=SimpleNamespace(advance=advance)
        backend=SpeculativeBackend(target,SimpleNamespace(deferred_draft=True),
            eos_ids=(99,),depth=7,prior=[.7]*7,prefill_executor=executor)
        handle=RequestHandle('test',0,1,'a');state=backend.create(handle)
        a=backend.advance(WorkRow(handle,0,tuple(range(9)),True,False),state,3)
        b=backend.advance(WorkRow(handle,9,tuple(range(9,17)),True,True),state,3)
        self.assertEqual((a.outputs,b.outputs),((),(7,)))
        self.assertEqual(state.context,list(range(17)))
        self.assertEqual(state.target.pos,17)
        self.assertEqual(calls,[tuple(range(9)),tuple(range(9,17))])
        backend.release(handle,state)
        self.assertIsNone(state.context)
        with self.assertRaisesRegex(ValueError,'observers'):
            SpeculativeBackend(target,None,eos_ids=(99,),depth=7,prior=[.7]*7,
                               observer=object(),prefill_executor=executor)


if __name__=='__main__':unittest.main()
