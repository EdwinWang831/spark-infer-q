"""Reject incomplete or mismatched full-generation performance evidence."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e8_integrated_export import MODES,summarize


class IntegratedEvidenceTests(unittest.TestCase):
    def fixture(self):
        baseline=json.loads((Path(__file__).resolve().parents[1]/'results/e7/2026-10-21-tensorfold-match.json').read_text())
        report={'status':'PASS_INTEGRATED','contract':{'modes':list(MODES),'repetitions':4},'harness_sha256':{},'tasks':[]}
        server=copy.deepcopy(baseline['candidate_servers'][0]);server.update(status='READY',default_prefill_rows=8,harness_sha256={})
        for ti,t in enumerate(baseline['tasks']):
            c=copy.deepcopy(t['candidate_correctness'][0]);c['serial_logit_rows']=len(c['token_ids'])-1
            c['checks']=[dict(mode=m,prefill_rows=w,kept_logits_exact=True,live_state_exact=True)
                for m,w in (('serial',8),('fixed7',8),('dynamic7',1),('dynamic7',8))]
            rec={k:t[k] for k in ('id','prompt','thinking','limit','check','arm')}
            if 'expected' in t:rec['expected']=t['expected']
            rec.update(correctness=c,timings=[])
            for rep in range(4):
                order=MODES if (rep+ti)%2==0 else MODES[::-1]
                for slot,mode in enumerate(order):
                    row=copy.deepcopy(next(r for r in t['modes']['ours_dynamic7']['timings'] if r['repetition']==rep))
                    row.update(mode=mode,order=slot,post_timing_check=dict(tokens_exact=True,live_state_exact=True))
                    row['runtime']['prefill_rows']=int(mode[-1]);rec['timings'].append(row)
            report['tasks'].append(rec)
        return report,server,baseline

    def test_paired_weighted_accounting(self):
        result=summarize(*self.fixture());s=result['summary']
        self.assertEqual(s['timings'],80);self.assertEqual(s['committed_decode_oracle_rows'],3348)
        self.assertEqual(s['aggregates']['all']['rows8_vs_rows1']['wall_ms'],1)
        self.assertEqual(s['timed_post_request_state_checks'],80)

    def test_missing_state_head_order_output_or_oracle_is_rejected(self):
        for fault in ('state','head','order','output','oracle','calibration'):
            report,server,baseline=self.fixture();t=report['tasks'][0]
            if fault=='state':t['timings'][0]['post_timing_check']['live_state_exact']=False
            elif fault=='head':server['head']['precision']='bf16'
            elif fault=='order':t['timings'][0]['order']=99
            elif fault=='output':t['timings'][0]['extra']['token_ids'][0]=-1
            elif fault=='oracle':t['correctness']['checks'][2]['kept_logits_exact']=False
            else:server['calibration']['conditional_prior'][0]=.12345
            with self.assertRaises(AssertionError,msg=fault):summarize(report,server,baseline)


if __name__=='__main__':unittest.main()
