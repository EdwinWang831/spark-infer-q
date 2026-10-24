"""Ready-decode fairness only changes the single-rank polling phase."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from spark_infer_q.deepinfer_bridge import SchedulerBridge, BridgeError


class ProgressTests(unittest.TestCase):
    def bridge(self,*,cached=10,cancelled=False,last_prompt=True,enabled=True):
        b=SchedulerBridge.__new__(SchedulerBridge)
        b.closed=b.poisoned=False;b.pending=None;b.serial=0;b.owner='test';b.pool=None
        b.interleave_prefill=enabled;b._last_prompt=last_prompt
        b.core=NS(SequenceStatus=NS(RUNNING=0,WAITING=1,SWAPPED=2))
        seq=NS(status=0,num_cached_tokens=cached,data=NS(get_len=lambda:11))
        b.requests={'short':dict(seq=seq,prompt_length=10,cancelled=cancelled)}
        empty=NS(size=0);output=NS(prompt_run=True,scheduled_seqs=[],clone_states=empty,clone_blocks=empty,zero_states=empty)
        observed=[]
        def schedule(pool):observed.append(b.native.decode_step);return [],output
        b.native=NS(decode_step=0,schedule_cnt=17,process_cnt=17,schedule=schedule)
        return b,observed

    def test_ready_decode_gets_next_step_without_changing_mtp_counters(self):
        b,observed=self.bridge();b.prepare()
        self.assertEqual(observed,[1]);self.assertEqual((b.native.schedule_cnt,b.native.process_cnt),(17,17))

    def test_prompt_only_cancelled_or_unfinished_prompt_does_not_defer_prefill(self):
        for kwargs in ({'cached':3},{'cancelled':True},{'last_prompt':False},{'enabled':False}):
            b,observed=self.bridge(**kwargs);b.prepare();self.assertEqual(observed,[0])
        b,observed=self.bridge();b.requests.clear();b.prepare();self.assertEqual(observed,[0])

    def test_inflight_plan_cannot_change_phase_or_reschedule(self):
        b,observed=self.bridge();b.pending=object()
        with self.assertRaises(BridgeError):b.prepare()
        self.assertEqual(b.native.decode_step,0);self.assertEqual(observed,[])

    def test_reject_cross_rank_or_wrong_slot_count_before_native_construction(self):
        for slots,ranks in ((1,1),(2,2),(4,1)):
            with patch.dict('os.environ',{'DEEPINFER_USE_NEXTN_DECODE':'0'}), self.assertRaisesRegex(ValueError,'two-slot single-rank'):
                SchedulerBridge(None,None,NS(size=lambda:ranks),None,None,vocab_size=100,eos_ids=(99,),
                    max_requests=slots,interleave_prefill=True)


if __name__=='__main__':unittest.main()
