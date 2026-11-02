"""Reject misleading long-context accounting; fixtures are not model evidence."""
import copy
import unittest

from scripts.e45_export import CASES,FEATURE_BYTES,request_summary


def fixture():
    rows=[];oracles=[]
    for name,n in CASES.items():
        trace=[dict(prompt=True,verified_rows=min(1024,n-i),emitted=int(i+1024>=n)) for i in range(0,n,1024)]
        trace += [dict(prompt=False,verified_rows=w,emitted=w,observed=w-1,accepted=w-1) for w in [8]*7+[7]]
        positions=[n]+[n+8*i for i in range(1,8)]+[n+63]
        oracles.append(dict(name=name,prompt_tokens=n,output_tokens=64,position=n+63,output_sha256='b'*64))
        for rep in range(-1,4):
            for mode in (('full','bounded') if rep<0 or rep%2==0 else ('bounded','full')):
                end=n+63;bounded=mode=='bounded'
                rows.append(dict(name=name,prompt_tokens=n,token_sha256='a'*64,output_sha256='b'*64,
                    completion_tokens=64,mode=mode,repetition=rep,warmup=rep<0,
                    trace=copy.deepcopy(trace),positions=positions[:],context_end=end,
                    context_start=max(0,end-2040) if bounded else 0,
                    retained_rows=min(2040,end) if bounded else end,
                    feature_storage_bytes=FEATURE_BYTES if bounded else end*25600,
                    outputs_exact=True,features_suffix_exact=True,state_exact=True,trace_exact=True,oracle_exact=True,
                    request_ms=100. if bounded else 110.,ttft_ms=20.,decode_ms=80. if bounded else 90.,
                    captures_before=16,captures_after=16,kernels_before={'a':'b'},kernels_after={'a':'b'},
                    graph_replays=8,graph_fallbacks={},head_fallbacks=0))
    return rows,oracles


class LongContextAccountingTests(unittest.TestCase):
    def test_complete_wall_and_decode_token_accounting_excludes_warmups(self):
        rows,gold=fixture();v=request_summary(rows,gold)
        self.assertEqual((v['timed_requests'],v['warmup_requests']),(32,8))
        self.assertEqual(v['total']['bounded']['output_tokens'],1024)
        self.assertEqual(v['total']['bounded']['decode_tokens'],1008)
        self.assertEqual(v['total']['bounded']['request_ms'],1600.)
        self.assertAlmostEqual(v['total']['bounded']['decode_residual_tps'],787.5)
        self.assertEqual(v['per_case']['code4096']['observed_acceptance'],1.)

    def test_wrong_position_storage_and_oracle_cannot_publish(self):
        for field,value in (('context_start',0),('retained_rows',4096),('feature_storage_bytes',1),
                            ('output_sha256','c'*64),('completion_tokens',63),('oracle_exact',False)):
            rows,gold=fixture();r=next(x for x in rows if x['name']=='code4096' and x['mode']=='bounded')
            r[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):request_summary(rows,gold)

    def test_missing_or_reordered_pairs_and_changed_decisions_rejected(self):
        for mutation in ('missing','order','trace','positions'):
            rows,gold=fixture()
            if mutation=='missing':rows.pop()
            elif mutation=='order':rows[0],rows[1]=rows[1],rows[0]
            elif mutation=='trace':rows[-1]['trace'][-1]['accepted']-=1
            else:rows[-1]['positions'][-1]-=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):request_summary(rows,gold)

    def test_warmup_pollution_recapture_fallback_and_invalid_timing_rejected(self):
        for field,value in (('warmup',True),('captures_after',17),('graph_fallbacks',{'eager':1}),
                            ('head_fallbacks',1),('decode_ms',1.),('request_ms',float('nan'))):
            rows,gold=fixture();rows[-1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):request_summary(rows,gold)


if __name__=='__main__':unittest.main()
