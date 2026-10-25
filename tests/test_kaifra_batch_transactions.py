"""Accepted-prefix transactions must validate both slots before any state write."""
from types import SimpleNamespace as NS
from contextlib import nullcontext
import unittest
from unittest.mock import patch
from spark_infer_q.kaifra_joint_target import JointTarget
from spark_infer_q.kaifra_batch import BatchBackend,Execution
from spark_infer_q.kaifra_speculative import VerifiedStep


class Event:
    def __init__(self):self.done=False
    def query(self):return self.done
    def synchronize(self):self.done=True


class BatchTransactionTests(unittest.TestCase):
    def fixture(self):
        target=JointTarget.__new__(JointTarget);a,b=NS(pos=3),NS(pos=7)
        target.states=[a,b];target.leased=[True,True];target.w=None;calls=[]
        def commit(w,st,buf,rows,keep,at):calls.append((st,rows,keep,at));st.pos+=keep
        target.forward=NS(commit=commit);target._event=Event
        ticket=NS(phase='ready',buffers=None,windows=(NS(state=a,ids=(1,2,3),physical_rows=4,at=0),
            NS(state=b,ids=(4,5),physical_rows=2,at=4)),commit_event=None,event=Event())
        target.inflight=ticket;return target,ticket,a,b,calls
    def test_invalid_later_prefix_never_partially_advances_batch(self):
        target,ticket,a,b,calls=self.fixture()
        for bad in (3,-1,True,1.,None):
            with self.assertRaises(ValueError):target.commit(ticket,{id(a):2,id(b):bad})
            self.assertEqual((a.pos,b.pos),(3,7));self.assertEqual(calls,[])
        with self.assertRaises(ValueError):target.commit(ticket,{id(object()):1})
        self.assertEqual(calls,[])
    def test_physical_offsets_zero_keep_duplicate_commit_and_retirement_fence(self):
        target,ticket,a,b,calls=self.fixture();event=target.commit(ticket,{id(a):0,id(b):1})
        self.assertEqual(calls,[(b,2,1,4)]);self.assertEqual((a.pos,b.pos),(3,8))
        with self.assertRaises(RuntimeError):target.commit(ticket,{id(a):1})
        with self.assertRaises(RuntimeError):target.release_state(a)
        with self.assertRaises(RuntimeError):target.retire(ticket)
        event.synchronize();target.retire(ticket);self.assertIsNone(target.inflight)
        with self.assertRaises(RuntimeError):target.retire(ticket)
    def test_discard_fences_verification_and_commit_before_slot_reuse(self):
        target,ticket,a,b,calls=self.fixture();event=target.commit(ticket,{id(a):1})
        target.discard(ticket)
        self.assertTrue(ticket.event.done);self.assertTrue(event.done);self.assertIsNone(target.inflight)
        target.release_state(a);self.assertFalse(target.leased[0])

    def test_async_commit_metrics_wait_for_fence_and_charge_batch_wall_once(self):
        backend=BatchBackend.__new__(BatchBackend);event=Event();retired=[]
        backend.target=NS(torch=NS(inference_mode=nullcontext),retire=retired.append,_slot=lambda state:0)
        backend.observer=None;ticket=object()
        row=NS(position=3);state=NS(target=NS(pos=5),context=[0]*5)
        result=VerifiedStep((1,2),2,(),(),0.,7.,0.)
        execution=Execution((row,row),({'state':state},{'state':state}),ticket,
            results=(result,result),event=event,phase='commit',committed=10.)
        self.assertFalse(backend.poll(execution));self.assertEqual(retired,[])
        self.assertEqual(execution.results[0].commit_ms,0.)
        event.done=True
        with patch('spark_infer_q.kaifra_batch.time.perf_counter',return_value=10.025):
            self.assertTrue(backend.poll(execution))
        self.assertEqual(retired,[ticket]);self.assertEqual(execution.phase,'done')
        self.assertAlmostEqual(sum(r.commit_ms for r in execution.results),25.)
        self.assertEqual(result.commit_ms,0.)


if __name__=='__main__':unittest.main()
