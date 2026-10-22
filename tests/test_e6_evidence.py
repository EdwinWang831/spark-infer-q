import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e6_checks import answer_check
from e6_export import summarize
from e6_tasks import HEADS,POLICIES,TASKS,CALIBRATION,order


class AnswerTests(unittest.TestCase):
    def test_code_fixtures_reject_input_mutation_and_wrong_count(self):
        task=dict(thinking=False,check='even')
        self.assertTrue(answer_check(task,'def count_even(nums):\n return sum(x%2==0 for x in nums)',True)['pass_check'])
        self.assertFalse(answer_check(task,'def count_even(nums):\n return len(nums)',True)['pass_check'])
        task['check']='merge'
        self.assertTrue(answer_check(task,'def merge_counts(left,right):\n out=left.copy()\n for k,v in right.items(): out[k]=out.get(k,0)+v\n return out',True)['pass_check'])
        self.assertFalse(answer_check(task,'def merge_counts(left,right):\n left.update(right)\n return left',True)['pass_check'])
        self.assertFalse(answer_check(task,'import os',True)['pass_check'])

    def test_literal_type_and_truncation_boundaries(self):
        task=dict(thinking=False,check='json')
        self.assertTrue(answer_check(task,'{"城市":"上海","数量":3}',True)['pass_check'])
        self.assertFalse(answer_check(task,'{"城市":"上海","数量":3.0}',True)['pass_check'])
        self.assertNotIn('pass_check',answer_check(dict(thinking=False,check='even'),'def x(',False))
        self.assertFalse(answer_check(dict(thinking=True,check='last_integer',expected=305),'305',False)['pass_check'])


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        step=dict(width=8,verified_rows=1,observed_drafts=0,accepted_drafts=0,emitted_tokens=1,draft_ms=.1,verify_ms=.5,commit_ms=.1)
        report=dict(status='PASS_BOUNDED_DRAFT_HEAD',stage='complete',kernel=dict(status='PASS_Q4_STORED_WEIGHT_ARITHMETIC',cases=[{}]*12),
            reference=dict(target_head_sha256_before='same',target_head_sha256_after='same'),
            preflight=dict(subset_rows=16,vocab_size=32),measurement_contract=dict(repetitions=4),
            calibration=dict(prompts=[dict(id=t['id']) for t in CALIBRATION]),heads={},profile=[],tasks=[],cold_controls=[])
        for h in HEADS:
            report['heads'][h]=dict(rows=16 if h.endswith('subset') else 32,effective_precision='q4' if h.startswith('q4') else 'bf16')
        for _ in range(4):
            report['profile'].append(dict(baseline_adapter_exact=True,heads={h:dict(full_top16_overlap=1.,logits_relative_l2=0.,logits_max_abs=0.,full_top1_matches=7,full_proposal_matches=7,
                samples=[dict(backbone_ms=1.,head_topk_ms=1.,selector_host_ms=1.,proposer_ms=3.)]) for h in HEADS}))
        for ti,t in enumerate(TASKS):
            r={**t,'prompt_ids':[1],'token_ids':[2,3],'text':'answer','stopped_by_eos':True,'correctness':[],'timings':[]}
            for h in HEADS:
                for p in POLICIES:r['correctness'].append(dict(head=h,policy=p,tokens_equal=True,committed_logits_exact=True,live_state_exact=True,steps=[step.copy()]))
            for rep in range(4):
                for slot,(h,p) in enumerate(order(rep,ti)):
                    r['timings'].append(dict(head=h,policy=p,repetition=rep,order=slot,decode_ms=ti+1+rep,
                        tokens_equal=True,live_state_exact=True,steps=[step.copy()]))
            report['tasks'].append(r)
        for t in TASKS[:2]:report['cold_controls'].append(dict(id=t['id'],timings=[dict(head=h,total_ms=20.,decode_ms=10.,tokens_equal=True) for h in HEADS for _ in range(2)]))
        return report

    def test_weighted_tokens_and_full_protocol_counts(self):
        result=summarize(self.fixture())
        self.assertEqual(result['summary']['timing_runs'],256);self.assertEqual(result['summary']['oracle_runs'],64)
        self.assertEqual(result['summary']['cold_controls'],16)
        self.assertAlmostEqual(result['aggregates']['bf16_full_fixed7']['decode_tokens_per_s'],8000/48)

    def test_reject_incomplete_or_mismatched_evidence(self):
        for change in ('state','order','head','hash','missing'):
            report=self.fixture()
            if change=='state':report['tasks'][0]['timings'][0]['live_state_exact']=False
            elif change=='order':report['tasks'][0]['timings'][0]['order']=7
            elif change=='head':report['heads']['q4_full']['effective_precision']='bf16'
            elif change=='hash':report['reference']['target_head_sha256_after']='modified'
            else:report['tasks'][0]['correctness'].pop()
            with self.assertRaises(AssertionError,msg=change):summarize(report)


if __name__=='__main__':unittest.main()
