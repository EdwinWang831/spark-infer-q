"""E41 coverage and matched-work rejection; these fixtures are not GPU evidence."""
import copy
from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch

from scripts.e41_export import numeric_summary,request_summary


class E41AccountingTests(unittest.TestCase):
    def test_numeric_requires_both_parities_and_partial_commit_coverage(self):
        cases=[]
        for i,(name,n) in enumerate((('english64',64),('code512',512),('code513',513),
                                     ('mixed1024',1024),('chinese1536',1536))):
            cases.append(dict(name=name,length=n,chunks=[512]*(n//512)+([n%512] if n%512 else []),
                logits_exact=True,features_exact=True,state_differences=[],
                windows=[dict(width=r,keep=r if r%2 else r//2,parity=(r+i)%2,exact=True) for r in range(1,9)]))
        graph=dict(graph_captures=16,graph_replays=40,graph_fallbacks={})
        result=numeric_summary(cases,graph)
        self.assertEqual((result['feature_rows'],result['full_logits_rows'],result['verify_windows']),(3649,185,40))
        for bad in ('features','keep','parity'):
            value=copy.deepcopy(cases)
            if bad=='features':value[0]['features_exact']=False
            if bad=='keep':value[0]['windows'][1]['keep']=2
            if bad=='parity':
                for c in value:
                    for w in c['windows']:w['parity']=0
            with self.subTest(bad=bad),self.assertRaises(ValueError):numeric_summary(value,graph)

    def requests(self):
        return [dict(name=name,prompt_tokens=n,mode=mode,repetition=r,warmup=r<0,
                     token_sha256='a'*64,output_sha256='b'*64,completion_tokens=64,
                     positions=[n,n+63],trace=[dict(verified_rows=4,accepted=2)],
                     outputs_exact=True,features_exact=True,state_exact=True,trace_exact=True,
                     captures_before=16,captures_after=16,graph_fallbacks={},graph_replays=20,
                     kernels_before={'subset:4:16':'c'*64},kernels_after={'subset:4:16':'c'*64},
                     head_fallbacks=0,ttft_ms=base if r>=0 else 100000,
                     request_ms=base+500 if r>=0 else 100500,decode_ms=500)
                for name,n in (('en',23),('zh',23),('code513',513),('mixed1536',1536))
                for mode,base in (('rows8',1000),('owned',500)) for r in range(-1,4)]

    def test_request_warmup_exclusion_and_compilation_work_or_state_rejected(self):
        rows=self.requests();result=request_summary(rows)
        self.assertEqual((result['timed_requests'],result['warmup_requests']),(32,8))
        self.assertEqual(result['per_case']['en']['arms']['owned']['request_ms'],4000)
        self.assertEqual(result['per_case']['en']['decode_residual_gain'],0.)
        for key,bad in (('state_exact',False),('captures_after',17),('graph_fallbacks',{'bucket':1}),
                        ('kernels_after',{}),('output_sha256','d'*64),('warmup',True),('head_fallbacks',1)):
            changed=copy.deepcopy(rows);changed[1][key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):request_summary(changed)

    def test_stage_requires_explicit_phase_and_audit_reference_before_submission(self):
        from scripts.e41_stage import main
        with patch('scripts.e41_stage.subprocess.run') as submit,redirect_stderr(io.StringIO()):
            with patch('sys.argv',['e41_stage.py','test-v1','test.json']):
                with self.assertRaises(SystemExit):main()
            with patch('sys.argv',['e41_stage.py','test-v1','test.json','--phase','audit']):
                with self.assertRaises(AssertionError):main()
            submit.assert_not_called()


if __name__=='__main__':unittest.main()
