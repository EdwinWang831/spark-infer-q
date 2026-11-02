"""Reject timing/work/source-policy corruption; fixtures are not GPU evidence."""
import copy
import unittest
from scripts.e49_export import check_run,token_sha


def fixture(mode='target_only'):
    outputs=[11,12,13,14]
    plans=[dict(position=0,rows=3,keep=3,prompt=True)]
    plans.extend(dict(position=n,rows=1,keep=1,prompt=False) for n in (3,4,5))
    events=[dict(emitted=1,position=3+n,finish='length' if n==3 else None,
        elapsed_seconds=.003*(n+1),ttft_seconds=.003) for n in range(4)]
    metrics=[dict(prompt=False,accepted=0,observed=0,verified_rows=1,rates=[.5]*7,head_fallback=False,
        draft_ms=.1,verify_ms=.2,commit_ms=.1) for _ in range(3)] if mode=='adaptive' else []
    return dict(mode=mode,warmup=True,prompt_tokens=3,output_cap=4,output_tokens=4,output_ids=outputs,output_sha256=token_sha(outputs),
        graph_captures=16,head_fallbacks=0,graph_fallbacks=[{}],timed_kernel_sources_unchanged=True,
        events=events,finish_reason='length',ttft_ms=3.,completion_ms=12.,request_ms=13.,residual_decode_ms=9.,
        residual_decode_tokens=3,decode_tok_s=3*1000/9,plans=plans,final_position=6,graph_replays=3,verify_rounds=3,
        state_sha256='d'*64,metrics=metrics,accepted=0,observed=0,width_histogram={'1':3} if metrics else {},
        draft_ms=sum(m['draft_ms'] for m in metrics),verify_ms=sum(m['verify_ms'] for m in metrics),
        commit_ms=sum(m['commit_ms'] for m in metrics),retrieval_hits=[True],peak_allocated_bytes=1024)


class LongEvidenceChecks(unittest.TestCase):
    def test_control_and_adaptive_work(self):
        for mode in ('target_only','adaptive'):
            self.assertEqual(check_run(fixture(mode))['verify_rounds'],3)

    def test_corrupt_event_work_or_timing_rejected(self):
        for name in ('tokens','clock','rate','position','replays','fallback','arm'):
            r=fixture()
            if name=='tokens':r['output_ids'][0]=0
            elif name=='clock':r['events'][2]['elapsed_seconds']=.001
            elif name=='rate':r['decode_tok_s']*=2
            elif name=='position':r['plans'][2]['position']=0
            elif name=='replays':r['graph_replays']=0
            elif name=='fallback':r['head_fallbacks']=1
            else:r['mode']='fixed'
            with self.subTest(name=name),self.assertRaises(ValueError):check_run(r)

    def test_adaptive_acceptance_and_costs_rejected(self):
        base=fixture('adaptive')
        for name in ('accepted','width','draft_ms'):
            r=copy.deepcopy(base)
            if name=='accepted':r['accepted']=1
            elif name=='width':r['metrics'][0]['verified_rows']=8
            else:r['draft_ms']=99
            with self.subTest(name=name),self.assertRaises(ValueError):check_run(r)

    def test_formal_jit_and_fake_long_scope_rejected(self):
        r=fixture();r['warmup']=False;r['timed_kernel_sources_unchanged']=False
        with self.assertRaisesRegex(ValueError,'JIT'):check_run(r)
        r['timed_kernel_sources_unchanged']=True;r['minimum_host_available_bytes']=10*1024**3
        with self.assertRaisesRegex(ValueError,'scope'):check_run(r)


if __name__=='__main__':unittest.main()
