"""Cached publication must still follow native fields and terminal history."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from spark_infer_q.deepinfer_bridge import SchedulerBridge,BridgeError,RequestHandle
from spark_infer_q.kaifra_optimized import optimized_provider


class Data:
    def __init__(self):self.ids=[1,2];self.full_reads=0
    def get_len(self):return len(self.ids)
    def get_last_token_id(self):return self.ids[-1]
    def get_token_ids(self):self.full_reads+=1;return self.ids.copy()


def fixture(enabled=True):
    b=object.__new__(SchedulerBridge);b.output_cache=enabled;b.owner='o';b.closed=b.poisoned=False;b.pending=None
    b.core=NS(SequenceStatus=NS(WAITING=0,RUNNING=1,FINISHED_STOPPED=2,SWAPPED=3))
    h=RequestHandle('o',0,1,'r');data=Data();seq=NS(data=data,status=1,num_cached_tokens=2,last_accepted_token=[],last_output_token=0)
    e=dict(handle=h,seq=seq,state=object(),cancelled=False,prompt_length=2,published=[])
    b.requests={'r':e};b.completed={};b.slots=[h];b.native=NS(free_seq=lambda s:None);b.backend=NS(release=lambda h,s:None)
    return b,h,e,data


class OutputCacheTests(unittest.TestCase):
    def test_partial_acceptance_reads_bounded_native_fields(self):
        b,h,e,d=fixture();d.ids.extend([7,8,9]);e['seq'].last_accepted_token=[7,8];e['seq'].last_output_token=9
        b._cache_published(e,(7,8,9));self.assertEqual(b.output_tokens(h),(7,8,9));self.assertEqual(d.full_reads,0)
        e['seq'].status=2;b._retire();self.assertEqual(b.completed[h]['token_ids'],[7,8,9]);self.assertEqual(d.full_reads,1)
        with self.assertRaises(BridgeError):b.output_tokens(h)

    def test_native_middle_length_and_tail_corruption_rejected(self):
        for bad in ('middle','length','tail'):
            b,h,e,d=fixture();d.ids.extend([7,8]);e['seq'].last_accepted_token=[7];e['seq'].last_output_token=8
            if bad=='middle':e['seq'].last_accepted_token=[6]
            elif bad=='length':d.ids.append(1)
            else:d.ids[-1]=6
            with self.subTest(bad=bad),self.assertRaises(BridgeError):b._cache_published(e,(7,8))
            self.assertEqual(e['published'],[])

    def test_terminal_history_crosscheck_and_waiting_cancel(self):
        b,h,e,d=fixture();d.ids.extend([7]);e['seq'].last_output_token=7;b._cache_published(e,(7,))
        d.ids[-1]=8;e['seq'].status=2
        with self.assertRaisesRegex(BridgeError,'retirement'):b._retire()
        self.assertTrue(b.poisoned)
        b,h,e,d=fixture();e.update(cancelled=True);e['seq'].status=2;b._cache_published(e,());b._retire()
        self.assertEqual(b.completed[h]['token_ids'],[])

    def test_default_keeps_native_history_reader(self):
        b,h,e,d=fixture(False);d.ids.extend([7]);b._cache_published(e,(7,))
        self.assertEqual(e['published'],[]);self.assertEqual(b.output_tokens(h),(7,));self.assertEqual(d.full_reads,1)

    def test_invalid_factory_options_fail_before_loading(self):
        for kwargs in (dict(query_graphs=1,history_window_rows=2040),dict(query_graphs=True)):
            with self.assertRaises(ValueError):optimized_provider(None,**kwargs)
        with patch.dict('os.environ',{'DEEPINFER_USE_NEXTN_DECODE':'0'}),self.assertRaisesRegex(ValueError,'output_cache'):
            SchedulerBridge(None,None,None,None,None,vocab_size=100,eos_ids=(99,),output_cache=1)


if __name__=='__main__':unittest.main()
