import itertools
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS

from spark_infer_q.joint_budget import BudgetRequest,JointBudget
from spark_infer_q.kaifra_batch import BatchSpeculativeBridge


class FakeTarget:
    graphs_enabled=True


def artifact():
    entries=[]
    for n in (1,2):
        for slots in itertools.combinations(range(2),n):
            for contexts in itertools.product(('short','long'),repeat=n):
                for buckets in itertools.product((1,2,4,8),repeat=n):
                    entries.append(dict(slots=list(slots),contexts=list(contexts),buckets=list(buckets),
                        verify_commit_ms=10.+sum(buckets)))
    return dict(schema=1,max_requests=2,buckets=[1,2,4,8],context_threshold=768,entries=entries,
        profile={},target_source_sha256='test',paid_draft_ms=dict(short=5.,long=8.))


class JointBudgetTests(unittest.TestCase):
    def test_cost_artifact_and_runtime_fingerprints_cannot_silently_drift(self):
        table=artifact();table['profile']={'model_sha256':'frozen'}
        table['target_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        data=json.dumps(table).encode();sha=hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'costs.json';path.write_bytes(data)
            budget=JointBudget.load(path,sha256=sha)
            with self.assertRaisesRegex(ValueError,'SHA256'):JointBudget.load(path,sha256='bad')
        budget.validate_runtime({'model_sha256':'frozen'},FakeTarget())
        with self.assertRaisesRegex(ValueError,'profile'):budget.validate_runtime({'model_sha256':'other'},FakeTarget())
        target=FakeTarget();target.graphs_enabled=False
        with self.assertRaisesRegex(ValueError,'Graph'):budget.validate_runtime({'model_sha256':'frozen'},target)

    def test_same_physical_bucket_can_justify_longer_prefix(self):
        budget=JointBudget(artifact())
        request=BudgetRequest(0,40,4,(1.,1.,1.),3)
        decision=budget.choose([request])
        self.assertEqual(decision['widths'],(4,))
        self.assertGreater(decision['score'],decision['independent_score'])
        self.assertEqual(decision['paid_draft_ms'],5.)

    def test_shared_cost_changes_best_width_without_changing_rates(self):
        table=artifact();table['paid_draft_ms']=dict(short=.01,long=.01)
        for row in table['entries']:
            row['verify_commit_ms']=(20.+sum(row['buckets']) if len(row['slots'])==2 else sum(row['buckets']))
        budget=JointBudget(table);rates=(.5,.3,.2,.1,.05,.02,.01)
        first=BudgetRequest(0,30,8,rates,1);second=BudgetRequest(1,30,8,rates,1)
        self.assertEqual(budget.choose([first])['widths'],(1,))
        both=budget.choose([first,second])
        self.assertGreater(sum(both['widths']),2)
        self.assertGreater(both['score'],both['independent_score'])

    def test_caps_slot_order_and_paid_cost_even_when_width_one(self):
        budget=JointBudget(artifact())
        a=BudgetRequest(1,1024,1,(),1);b=BudgetRequest(0,20,2,(0.,)*7,1)
        decision=budget.choose([a,b]);reverse=budget.choose([b,a])
        self.assertEqual(decision['widths'],(1,1));self.assertEqual(decision['paid_draft_ms'],5.)
        self.assertEqual(decision['score'],reverse['score'])
        self.assertEqual(decision['total_ms'],17.)
        self.assertEqual(budget.choose([])['widths'],())

    def test_missing_costs_probabilities_and_foreign_slots_fail_closed(self):
        table=artifact();table['entries'].pop()
        with self.assertRaises(ValueError):JointBudget(table)
        budget=JointBudget(artifact())
        for request in (BudgetRequest(0,20,8,(.2,.8),1),BudgetRequest(2,20,1,(),1),BudgetRequest(0,20,1,(),2)):
            with self.assertRaises(ValueError):budget.choose([request])
        request=BudgetRequest(0,20,1,(),1)
        with self.assertRaises(ValueError):budget.choose([request,request])

    def test_bridge_maps_physical_slots_and_skips_cancelled_entries(self):
        bridge=object.__new__(BatchSpeculativeBridge);calls=[]
        bridge.joint_budget=NS(choose=lambda requests:(calls.extend(requests) or dict(widths=(4,))))
        a=NS(next_width=3,target=object(),context=[0]*20,cumulative=(1.,)*7,output_tokens=1)
        b=NS(next_width=1,target=object(),context=[0]*30,cumulative=(),output_tokens=0)
        entries={'a':dict(state=a,cancelled=False,seq=NS(sampling_params=NS(max_tokens=9))),
            'b':dict(state=b,cancelled=True,seq=NS(sampling_params=NS(max_tokens=2)))}
        bridge._entry=entries.__getitem__;bridge.backend=NS(target=NS(_slot=lambda target:1 if target is a.target else 0))
        plan=NS(rows=(NS(handle='a',token_ids=range(8)),NS(handle='b',token_ids=range(8))))
        self.assertEqual(bridge._postdraft_widths(plan),[4,1])
        self.assertEqual(calls[0].slot,1);self.assertEqual(a.next_width,4)


if __name__=='__main__':unittest.main()
