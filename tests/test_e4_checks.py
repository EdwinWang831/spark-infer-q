import io
import hashlib
import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e4_client import answer_check, request
from e4_tasks import TASKS, MODES, ordered_modes
import e4_export
import e4_warmup_export


class MatchedChecks(unittest.TestCase):
    def test_literal_checks_require_answer_not_reasoning(self):
        task={'arm':'natural','check':'integer','expected':136}
        self.assertTrue(answer_check(task,'136')['pass'])
        self.assertFalse(answer_check(task,'136</think>137')['pass'])
        self.assertFalse(answer_check(task,'The answer is 136')['pass'])
        js={'arm':'natural','check':'json'}
        self.assertTrue(answer_check(js,'{"name":"otter","count":5,"enabled":false}')['pass'])
        self.assertFalse(answer_check(js,'{"name":"otter","count":5,"enabled":0}')['pass'])

    def test_code_fixture_and_rejection(self):
        task={'arm':'natural','check':'positive'}
        self.assertTrue(answer_check(task,'def sum_positive(nums):\n return sum(x for x in nums if x>0)')['pass'])
        self.assertFalse(answer_check(task,'def sum_positive(nums):\n return sum(nums)')['pass'])
        self.assertFalse(answer_check(task,'import os\ndef sum_positive(nums):\n return 0')['pass'])
        self.assertFalse(answer_check(task,'def sum_positive(nums):\n return ().__class__')['pass'])

    def test_capped_code_is_never_labelled_correct_program(self):
        result=answer_check({'arm':'capped','check':'prefix'},'def incomplete(')
        self.assertTrue(result['nonempty']);self.assertNotIn('pass',result)

    def test_sse_usage_can_arrive_after_runtime(self):
        chunks=[{'choices':[{'delta':{'reasoning_content':'hmm'}}]},
            {'choices':[{'delta':{'content':'136'},'finish_reason':'stop'}],
             'tensorfold':{'enable_thinking':True,'tokens_per_second':40}},
            {'choices':[],'usage':{'completion_tokens':4,'prompt_tokens':10}}]
        data=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n'
        with patch('e4_client.post',return_value=io.BytesIO(data.encode())):
            r=request('http://unused',dict(prompt='p',thinking=True,limit=512,id='x'),'serial')
        self.assertEqual((r['content'],r['reasoning'],r['finish_reason']),('136','hmm','stop'))
        self.assertEqual(r['usage']['completion_tokens'],4)
        self.assertEqual(r['runtime']['tokens_per_second'],40)

    def test_case_split_and_balanced_order(self):
        self.assertEqual(len(TASKS),14)
        self.assertEqual(sum(t['arm']=='natural' for t in TASKS),12)
        self.assertEqual(len({t['id'] for t in TASKS}),14)
        for slot in zip(*(ordered_modes(r,5) for r in range(4))):self.assertEqual(set(slot),set(MODES))

    def export_fixture(self):
        tasks=[dict(id='a',prompt='p',thinking=False,limit=10,arm='natural'),
            dict(id='b',prompt='q',thinking=False,limit=3,arm='capped')]
        sha=hashlib.sha256(b'11,12,13').hexdigest()[:12]
        sessions=[]
        for engine in ('native','ours'):
            modes=('serial','native_mtp') if engine=='native' else MODES
            for sid in range(2):
                session=dict(engine=engine,session=sid,status='PASS_MEASURED',tasks=[])
                sessions.append(session)
                for ti,task in enumerate(tasks):
                    p={**task,'timings':[],'prompt_ids':[7,8]};session['tasks'].append(p)
                    p['correctness']=dict(token_ids=[11,12,13],prompt_ids=[7,8],kept_logits_exact=True,
                        live_state_exact=True,modes=['fixed6','fixed7','dynamic7'])
                    for rep in range(sid*2,sid*2+2):
                        ordered=(modes if (rep+ti)%2==0 else modes[::-1]) if engine=='native' else ordered_modes(rep,ti)
                        for slot,mode in enumerate(ordered):
                            n=4 if engine=='native' and ti==0 else 3
                            ms=(10 if ti==0 else 20)+rep
                            p['timings'].append(dict(mode=mode,repetition=rep,order=slot,
                                usage={'completion_tokens':n,'prompt_tokens':2},
                                runtime={'enable_thinking':False,'tokens_per_second':(n-1)*1000/ms,
                                    'token_sha':'native-other' if n==4 else sha},
                                extra={'token_ids':[11,12,13],'prompt_ids':[7,8],'steps':[] if mode=='serial' else
                                    [{'width':7,'draft_ms':1,'verify_ms':2,'commit_ms':1}]},
                                wall_ms=ms+5,wire_ttft_ms=5,finish_reason='stop' if ti==0 else 'length',answer_check={}))
        study=dict(status='PASS_MATCHED_MEASUREMENTS',sessions=[dict(engine=e,session=s,status='PASS_MEASURED')
            for e,s in [('native',0),('ours',0),('ours',1),('native',1)]])
        return tasks,study,sessions

    def test_export_keeps_cross_engine_differences_and_weights_tokens(self):
        tasks,study,sessions=self.export_fixture()
        with patch.object(e4_export,'TASKS',tasks):r=e4_export.summarize(study,sessions,{},[{},{}])
        self.assertEqual(r['summary']['timing_runs'],48)
        self.assertEqual(r['summary']['candidate_nonserial_oracle_runs'],12)
        self.assertEqual(r['summary']['cross_engine_hash_agreement_cases'],1)
        self.assertEqual(r['summary']['native_spec_vs_serial_hash_agreement_cases'],2)
        a=r['summary']['aggregates']['all']['ours_dynamic7']
        self.assertAlmostEqual(a['decode_tokens_per_s'],4000/33)
        self.assertFalse(r['tasks'][0]['native_vs_candidate_token_count_equal'])

    def test_export_refuses_prompt_template_mismatch(self):
        tasks,study,sessions=self.export_fixture()
        for s in sessions:
            if s['engine']=='native':s['tasks'][0]['prompt_ids']=[999]
        with patch.object(e4_export,'TASKS',tasks),self.assertRaisesRegex(AssertionError,'template mismatch'):
            e4_export.summarize(study,sessions,{},[{},{}])

    def warmup_fixture(self):
        tasks, study, sessions = self.export_fixture()
        for session in sessions:
            for task in session['tasks']:
                for row in task['timings']:
                    row.update(content='answer', reasoning='')
        with patch.object(e4_export, 'TASKS', tasks):
            main = e4_export.summarize(study, sessions, {}, [{},{}])
        control = dict(status='PASS_WARMUP_CONTROL', post_hoc=True, tasks=[])
        for task in main['tasks']:
            rows = [dict(r) for mode in ('serial','native_mtp')
                for r in task['modes']['native_'+mode]['timings']]
            warm = [dict(next(r for r in rows if r['mode']==mode)) for mode in ('serial','native_mtp')]
            control['tasks'].append({**{k:task[k] for k in tasks[0]},
                'prompt_ids':task['native_prompt_ids'], 'timings':rows, 'warmup':warm})
        return tasks, control, main

    def test_warmup_export_checks_outputs_and_preserves_separate_scope(self):
        tasks, control, main = self.warmup_fixture()
        with patch.object(e4_warmup_export, 'TASKS', tasks):
            result = e4_warmup_export.summarize(control, main)
        self.assertTrue(result['control']['post_hoc'])
        self.assertEqual(result['summary']['aggregates']['all']['native_mtp']['decode_tokens'], 5)
        self.assertAlmostEqual(result['summary']['aggregates']['all']['native_mtp']['decode_tokens_per_s'], 5000/33)
        control['tasks'][0]['warmup'][0]['content']='different'
        with patch.object(e4_warmup_export, 'TASKS', tasks), self.assertRaises(AssertionError):
            e4_warmup_export.summarize(control, main)

    def test_warmup_export_rejects_cache_and_changed_hash(self):
        for corruption in ('cache','hash'):
            tasks, control, main = self.warmup_fixture()
            row=control['tasks'][0]['timings'][0]
            if corruption=='cache':row['usage']={**row['usage'],'prompt_tokens_details':{'cached_tokens':1}}
            else:row['runtime']={**row['runtime'],'token_sha':'bad'}
            with patch.object(e4_warmup_export, 'TASKS', tasks), self.assertRaises(AssertionError):
                e4_warmup_export.summarize(control, main)


if __name__=='__main__':unittest.main()
