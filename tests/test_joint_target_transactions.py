from contextlib import nullcontext
from types import SimpleNamespace as NS
import unittest

from spark_infer_q.deepinfer_bridge import BridgeError,RequestHandle,WorkRow
from spark_infer_q.kaifra_joint_target import JointTarget
from spark_infer_q.kaifra_joint_runtime import JointTargetBackend,JointSchedulerBridge


class Event:
    def __init__(self):self.done=False
    def query(self):return self.done
    def synchronize(self):self.done=True


class JointTransactionTests(unittest.TestCase):
    def target(self):
        target=JointTarget.__new__(JointTarget)
        a,b=NS(pos=3),NS(pos=7)
        target.states=[a,b];target.leased=[True,True];target.w=None
        target.torch=NS(cuda=NS(Event=Event))
        calls=[]
        target.forward=NS(commit=lambda w,st,buf,rows,keep,at: (calls.append((st,rows,keep,at)),setattr(st,'pos',st.pos+keep)))
        target._event=Event
        ticket=NS(phase='ready',buffers=None,windows=(NS(state=a,ids=(1,2,3),physical_rows=4,at=0),
            NS(state=b,ids=(4,5),physical_rows=2,at=4)),commit_event=None)
        target.inflight=ticket
        return target,ticket,a,b,calls

    def test_invalid_second_prefix_does_not_advance_first_state(self):
        target,ticket,a,b,calls=self.target()
        with self.assertRaises(ValueError):target.commit(ticket,{id(a):2,id(b):3})
        self.assertEqual((a.pos,b.pos),(3,7));self.assertEqual(calls,[])
        with self.assertRaises(ValueError):target.commit(ticket,{id(object()):1})
        self.assertEqual(calls,[])

    def test_offsets_zero_keep_duplicate_commit_and_fence(self):
        target,ticket,a,b,calls=self.target()
        event=target.commit(ticket,{id(a):0,id(b):1})
        self.assertEqual((a.pos,b.pos),(3,8))
        self.assertEqual(calls,[(b,2,1,4)])
        with self.assertRaises(RuntimeError):target.commit(ticket,{id(a):1})
        with self.assertRaises(RuntimeError):target.retire(ticket)
        event.synchronize();target.retire(ticket)
        self.assertIsNone(target.inflight)
        with self.assertRaises(RuntimeError):target.retire(ticket)


class NativePlanJointTests(unittest.TestCase):
    def test_one_gpu_call_for_two_native_decode_rows(self):
        handles=[RequestHandle('owner',i,1,str(i)) for i in range(2)]
        states=[NS(pos=5),NS(pos=9)]
        rows=tuple(WorkRow(h,s.pos,(11+i,),False,False) for i,(h,s) in enumerate(zip(handles,states)))
        calls=[]
        class Backend:
            def advance_joint(self,work):
                calls.append(tuple(row.handle for row,_ in work))
                for _,s in work:s.pos+=1
                return {row.handle:20+i for i,(row,_) in enumerate(work)}
            def position(self,s):return s.pos
        bridge=JointSchedulerBridge.__new__(JointSchedulerBridge)
        bridge.closed=bridge.poisoned=False;bridge.owner='owner';bridge.vocab_size=100
        bridge.requests={h.request_id:dict(handle=h,state=s,cancelled=False) for h,s in zip(handles,states)}
        bridge.backend=Backend();plan=NS(owner='owner',rows=rows,executed=False)
        bridge.pending=plan;bridge.execute(plan)
        self.assertEqual(calls,[tuple(handles)]);self.assertEqual(plan.predictions,(20,21))
        with self.assertRaises(BridgeError):bridge.execute(plan)

    def test_window_order_is_mapped_by_owned_state(self):
        handles=[RequestHandle('owner',i,1,str(i)) for i in range(2)]
        states=[NS(pos=3),NS(pos=8)]
        rows=[WorkRow(h,s.pos,(9,),False,False) for h,s in zip(handles,states)]
        calls=[]
        ticket=NS(windows=(NS(state=states[1],at=0),NS(state=states[0],at=1)),
            cpu_predictions=[71,42],full_logits=None)
        def commit(t,keeps):
            self.assertEqual(keeps,{id(s):1 for s in states})
            for s in states:s.pos+=1
            event=Event();ticket.commit_event=event;return event
        target=NS(torch=NS(inference_mode=nullcontext),buffers={i:None for i in range(1,9)},
            launch=lambda requests,diagnostic:ticket,poll=lambda t:True,commit=commit,
            retire=lambda t:calls.append('retired'),inflight=ticket)
        backend=JointTargetBackend(target)
        backend.states=dict(zip(handles,states))
        result=backend.advance_joint(list(zip(rows,states)))
        self.assertEqual(result,{handles[0]:42,handles[1]:71})
        self.assertEqual(calls,['retired']);self.assertTrue(ticket.commit_event.done)
