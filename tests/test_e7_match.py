import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import e7_export
from e7_tasks import TASKS,MODES,ordered_modes
import test_e4_checks as e4test


class E7Tests(unittest.TestCase):
    def fixture(self):
        tasks,study,sessions=e4test.MatchedChecks().export_fixture()
        study.update(harness_sha256={},preflight={'e6_report_sha256':'reference'})
        for s in sessions:
            engine=s['engine'];modes=MODES[engine]
            for ti,p in enumerate(s['tasks']):
                for row in p['timings']:row.update(content='answer',reasoning='')
                selected=[]
                for rep in range(s['session']*2,s['session']*2+2):
                    for slot,mode in enumerate(ordered_modes(engine,rep,ti)):
                        row=next(r for r in p['timings'] if r['repetition']==rep and r['mode']==mode)
                        row['order']=slot;selected.append(row)
                p['timings']=selected
                if engine=='native':p['warmup']=[copy.deepcopy(next(r for r in selected if r['mode']==mode)) for mode in modes]
                else:
                    p['warmup']={'completed':True};p['correctness']['modes']=['fixed7','dynamic7']
        identity={'binary_sha256':'cb2eb4e21e2031cad267ad646c80170df894f76b426ebddcba0895628682185d'}
        server=dict(head={'precision':'q4','rows':248320,'incremental_payload_bytes':397312000},harness_sha256={},e6_source_sha256='reference')
        return tasks,study,sessions,identity,[copy.deepcopy(server),copy.deepcopy(server)]

    def test_frozen_tasks_and_counts(self):
        self.assertEqual(len(TASKS),10)
        self.assertEqual(sum(t['arm']=='natural' for t in TASKS),8)
        self.assertEqual(sum(t['arm']=='capped' for t in TASKS),2)
        tasks,*args=self.fixture()
        with patch.object(e7_export,'TASKS',tasks):r=e7_export.summarize(*args)
        self.assertEqual(r['summary']['timing_runs'],40)
        self.assertEqual(r['summary']['complete_warmup_runs'],20)
        self.assertEqual(r['summary']['candidate_nonserial_oracle_runs'],8)
        self.assertFalse(r['tasks'][0]['native_vs_candidate_token_count_equal'])
        self.assertEqual(r['summary']['cross_engine_hash_agreement_cases'],1)
        self.assertAlmostEqual(r['summary']['aggregates']['all']['ours_dynamic7']['decode_tokens_per_s'],4000/33)

    def test_no_claim_without_warmup_matching_input_or_correct_head(self):
        for error in ('warmup','head','template','ordering','oracle'):
            tasks,study,sessions,identity,servers=self.fixture()
            if error=='warmup':sessions[0]['tasks'][0]['warmup'].pop()
            elif error=='head':servers[0]['head']['precision']='bf16'
            elif error=='template':sessions[0]['tasks'][0]['prompt_ids']=[999]
            elif error=='ordering':sessions[2]['tasks'][0]['timings'][0]['mode']='native_mtp'
            else:sessions[2]['tasks'][0]['correctness']['live_state_exact']=False
            with patch.object(e7_export,'TASKS',tasks),self.assertRaises(AssertionError,msg=error):
                e7_export.summarize(study,sessions,identity,servers)


if __name__=='__main__':unittest.main()
