"""Accounting rejects unequal work and accidentally timed cold requests."""
import copy
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr
import io
from scripts.e40_export import prompt_summary,request_summary


class E40AccountingTests(unittest.TestCase):
    def test_phase_must_be_explicit_before_staging_any_remote_job(self):
        from scripts.e40_stage import main
        with patch('sys.argv',['e40_stage.py','audit-example','example.json']), \
             patch('scripts.e40_stage.subprocess.run') as submit, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error: main()
            self.assertEqual(error.exception.code,2)
            submit.assert_not_called()

    def test_prompt_uses_total_tokens_over_total_time_and_requires_each_pair(self):
        rows=[dict(name=name,tokens=n,mode=mode,repetition=r,prompt_ms=n*cost)
              for name,n in (('english',64),('code',512),('mixed',512))
              for mode,cost in (('rows8',2),('owned',1)) for r in range(4)]
        value=prompt_summary(rows)
        self.assertEqual(value['tokens_per_arm'],4352)
        self.assertEqual(value['prompt_tps'],dict(rows8=500.,owned=1000.))
        damaged=copy.deepcopy(rows);damaged[1]['repetition']=0
        with self.assertRaises(ValueError):prompt_summary(damaged)

    def requests(self):
        return [dict(kind=kind,name=name,prompt_tokens=n,mode=mode,repetition=r,
                     warmup=r<0,tokens_sha256='a'*64,completion_tokens=24,final_position=n+23,
                     exact=True,state_exact=True,trace=[dict(accepted=2,verified_rows=4)],
                     request_ms=1000 if r<0 else cost*100,ttft_ms=500 if r<0 else cost*10)
                for kind in ('target','dflash') for name,n in (('en',23),('zh',23),('code512',512))
                for mode,cost in (('rows8',2),('owned',1)) for r in range(-1,4)]

    def test_warmup_excluded_and_changed_tokens_state_or_acceptance_rejected(self):
        rows=self.requests();value=request_summary(rows)
        self.assertEqual((value['timed_requests'],value['warmup_requests']),(48,12))
        case=value['per_case']['dflash']['en']
        self.assertEqual(case['arms']['owned']['request_ms'],400)
        self.assertEqual(case['request_cost_reduction'],.5)
        for key,bad in (('tokens_sha256','b'*64),('state_exact',False),('warmup',False),
                        ('trace',[dict(accepted=1,verified_rows=4)])):
            changed=copy.deepcopy(rows);changed[0][key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):request_summary(changed)


if __name__=='__main__':unittest.main()
