"""Native event timing and work accounting; fixtures are not GPU evidence."""
import unittest
from scripts.e48_export import check_record


def fixture():
    requests=[]
    for n,first,last in (('a',3.,9.),('b',4.,10.)):
        requests.append(dict(name=n,prompt_tokens=3,input_sha256=n*64,output_tokens=2,output_sha256='c'*64,
            final_position=4,state_sha256='d'*64,generated_text='KEY',retrieval_keys=['KEY'],retrieval_hits=[True],
            ttft_ms=first,completion_ms=last,maximum_output_gap_ms=last-first,finish_reason='stop',submitted_ms=0.,
            events=[dict(emitted=1,finish=None,position=3,ttft_seconds=first/1000,elapsed_seconds=first/1000),
                    dict(emitted=1,finish='stop',position=4,ttft_seconds=first/1000,elapsed_seconds=last/1000)]))
    return dict(interleave=True,warmup=False,repetition=0,output_cap=2,requests=requests,outputs_and_state_exact=True,
        graph_captures=32,head_fallbacks=0,graph_fallbacks=[{},{}],timed_kernel_sources_unchanged=True,
        request_ms=12.,graph_replays=2,
        plans=[dict(ready_and_waiting=False,phase='prompt',rows=[dict(name=n,position=0,rows=3,prompt=True) for n in ('a','b')]),
               dict(ready_and_waiting=False,phase='decode',rows=[dict(name=n,position=3,rows=1,prompt=False) for n in ('a','b')])])


class FairnessEvidenceChecks(unittest.TestCase):
    def test_two_request_event_work_and_graph_counts(self):
        self.assertEqual(check_record(fixture()),0)

    def test_wrong_gap_work_state_or_replays_rejected(self):
        for case in ('gap','work','state','replays'):
            r=fixture()
            if case=='gap':r['requests'][0]['maximum_output_gap_ms']=0.
            elif case=='work':r['plans'][0]['rows'][0]['position']=1
            elif case=='state':r['outputs_and_state_exact']=False
            else:r['graph_replays']=0
            with self.subTest(case=case),self.assertRaises(ValueError):check_record(r)

    def test_first_warmup_can_prepare_kernels_but_formal_timer_cannot(self):
        r=fixture();r['timed_kernel_sources_unchanged']=False
        with self.assertRaisesRegex(ValueError,'preparation'):check_record(r)
        r.update(warmup=True,repetition=-1);check_record(r)

    def test_prefill_burst_allowed_in_control_but_rejected_in_candidate(self):
        r=fixture();r['plans']=[
            dict(ready_and_waiting=False,phase='prompt',rows=[dict(name='a',position=0,rows=3,prompt=True)]),
            dict(ready_and_waiting=True,phase='prompt',rows=[dict(name='b',position=0,rows=1,prompt=True)]),
            dict(ready_and_waiting=True,phase='prompt',rows=[dict(name='b',position=1,rows=2,prompt=True)]),
            r['plans'][-1]]
        with self.assertRaisesRegex(ValueError,'starved'):check_record(r)
        r['interleave']=False;self.assertEqual(check_record(r),2)


if __name__=='__main__':unittest.main()
