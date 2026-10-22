import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e5_export import CASES, validate


class BridgeEvidenceChecks(unittest.TestCase):
    def fixture(self):
        cases=[]
        for name in CASES:
            done=[dict(slot=0,generation=1,cancelled=False,token_ids=[2,3,4])]
            if name.startswith('cancel_'):
                done=[dict(slot=0,generation=1,cancelled=True,token_ids=[]),
                    dict(slot=0,generation=2,cancelled=False,token_ids=[4])]
            cases.append(dict(case=name,status='PASS',free_native_blocks=10,total_native_blocks=10,
                completed=done,events=[dict(slot=0,generation=1,start=0,rows=2,published=0,cancelled=False,prompt=True)],
                backend_checks=dict(full_logit_rows=2,live_state_checks=1,released_states=[{} for _ in done])))
            if name=='two_requests_chunked_prompt_and_interleaved_decode':
                cases[-1]['events']=[dict(slot=slot,generation=1,start=position,rows=1,published=1,
                    cancelled=False,prompt=False,request_id=str(slot))
                    for slot,position in [(0,0),(1,0),(0,1)]]
            if name=='invalid_admission_and_full_slots':
                cases[-1]['events']=[]
                cases[-1]['backend_checks'].update(full_logit_rows=0,live_state_checks=0)
        return dict(status='PASS_BOUNDED_NATIVE_SCHEDULER_BRIDGE',kind='full_target_native_scheduler',
            cuda_initialized=True,core_sha256='f6aa14b790eb7bc729251eea152fd9c83cf215bc5ab912e25ee19c57d8fc1768',cases=cases)

    def test_complete_evidence_sums_real_checks(self):
        result=validate(self.fixture())
        self.assertEqual(result['full_logit_rows'],12)
        self.assertEqual(result['live_state_checks'],6)

    def test_rejects_block_leak_cancel_publish_and_position_skips(self):
        base=self.fixture()
        for kind in ('block','publish','position','reuse','missing'):
            report=copy.deepcopy(base)
            if kind=='block':report['cases'][0]['free_native_blocks']=9
            elif kind=='publish':report['cases'][0]['events'][0].update(cancelled=True,published=1)
            elif kind=='position':report['cases'][0]['events'][0]['start']=1
            elif kind=='reuse':report['cases'][3]['completed'][1]['generation']=1
            else:report['cases'].pop()
            with self.assertRaises(AssertionError,msg=kind):validate(report)


if __name__=='__main__':unittest.main()
