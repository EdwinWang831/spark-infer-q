"""A device wait cannot substitute for verified poll/publication or advance state."""
from types import SimpleNamespace as NS
import unittest
from spark_infer_q.kaifra_batch import AsyncBatchMixin,SplitExecution
from spark_infer_q.deepinfer_bridge import BridgeError


class Fence:
    def __init__(self,fail=False):self.calls=0;self.fail=fail
    def synchronize(self):
        self.calls+=1
        if self.fail:raise RuntimeError('device fence failed')


class Bridge(AsyncBatchMixin):
    poisoned=False
    def _plan(self,plan):
        if plan is not self.pending:raise BridgeError('foreign pending plan')


class DeviceWaitTests(unittest.TestCase):
    def test_wait_selects_current_split_head_and_commit_without_publication(self):
        bridge=Bridge();verify=Fence();commit=Fence()
        current=NS(phase='verify',target_ticket=NS(event=verify),event=None)
        plan=NS(executed=False,execution=SplitExecution((),(),current));bridge.pending=plan
        self.assertTrue(bridge.wait_for_device(plan));self.assertEqual(verify.calls,1)
        self.assertFalse(plan.executed)
        current.phase='commit';current.event=commit
        self.assertTrue(bridge.wait_for_device(plan));self.assertEqual(commit.calls,1)
        self.assertFalse(plan.executed)
        current.phase='done';self.assertFalse(bridge.wait_for_device(plan))
        plan.executed=True;self.assertFalse(bridge.wait_for_device(plan))
    def test_missing_event_foreign_plan_and_failed_fence_remain_explicit(self):
        bridge=Bridge();plan=NS(executed=False);bridge.pending=plan
        self.assertFalse(bridge.wait_for_device(plan))
        with self.assertRaises(BridgeError):bridge.wait_for_device(NS(executed=False))
        event=Fence(True);plan.execution=NS(phase='verify',target_ticket=NS(event=event))
        with self.assertRaisesRegex(RuntimeError,'device fence failed'):bridge.wait_for_device(plan)
        self.assertTrue(bridge.poisoned);self.assertFalse(plan.executed)


if __name__=='__main__':unittest.main()
