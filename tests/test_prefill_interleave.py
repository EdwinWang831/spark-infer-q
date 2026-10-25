"""Native phase-hint contract; real scheduler/GPU evidence lives in E48."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spark_infer_q.deepinfer_bridge import SchedulerBridge


def fixture(enabled=True):
    b=object.__new__(SchedulerBridge)
    b.prefill_interleave=enabled;b._last_prompt=True
    b.core=SimpleNamespace(SequenceStatus=SimpleNamespace(WAITING=0,RUNNING=1))
    b.native=SimpleNamespace(decode_step=0)
    b.interleave_hints=dict(decode_after_prompt=0,prompt_after_decode=0,prompt_without_ready_decode=0)
    b.requests=dict(a=dict(seq=SimpleNamespace(status=1,num_cached_tokens=8),prompt_length=8),
                    b=dict(seq=SimpleNamespace(status=0,num_cached_tokens=0),prompt_length=8192))
    return b


class NativePhaseHints(unittest.TestCase):
    def test_alternates_native_polling_phase_only_when_decode_ready(self):
        b=fixture();b._prefill_phase()
        self.assertEqual(b.native.decode_step,1)
        b._last_prompt=False;b._prefill_phase()
        self.assertEqual(b.native.decode_step,0)
        self.assertEqual(b.interleave_hints['decode_after_prompt'],1)
        self.assertEqual(b.interleave_hints['prompt_after_decode'],1)

    def test_cancelled_or_incomplete_decode_never_forces_empty_decode_step(self):
        b=fixture();del b.requests['a'];b.native.decode_step=1;b._prefill_phase()
        self.assertEqual(b.native.decode_step,0)
        b=fixture();b.requests['a']['seq'].num_cached_tokens=0;b._prefill_phase()
        self.assertEqual(b.native.decode_step,0)

    def test_default_and_decode_only_preserve_native_counter(self):
        for b in (fixture(False),fixture(True)):
            if b.prefill_interleave:b.requests['b']['seq'].status=1
            b.native.decode_step=19;b._prefill_phase()
            self.assertEqual(b.native.decode_step,19)
            self.assertFalse(any(b.interleave_hints.values()))

    def test_policy_rejects_multi_rank_mtp_wrong_slot_count_or_non_boolean(self):
        for slots,rank_count,drafts,value in ((1,1,0,True),(3,1,0,True),(2,2,0,True),(2,1,7,True),(2,1,0,1)):
            group=SimpleNamespace(size=lambda:rank_count)
            with self.subTest(slots=slots,ranks=rank_count,drafts=drafts,value=value),patch.dict(os.environ,{'DEEPINFER_USE_NEXTN_DECODE':str(drafts)}):
                with self.assertRaises(ValueError):
                    SchedulerBridge(None,None,group,None,None,vocab_size=100,eos_ids=(99,),
                        max_requests=slots,max_drafts=drafts,prefill_interleave=value)


if __name__=='__main__':unittest.main()
