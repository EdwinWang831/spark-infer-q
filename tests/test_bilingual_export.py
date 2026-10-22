from pathlib import Path
import sys
import tempfile
import unittest
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e16_export import export, online_summary


class BilingualExportTests(unittest.TestCase):
    def test_running_online_is_never_reported_as_measured_speed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';raw.mkdir()
            (raw/'e16-ablation.json').write_text(json.dumps(dict(status='RUNNING',tasks=[{}])))
            result=export(raw,root/'public')
            self.assertEqual(result['status'],'IN_PROGRESS')
            self.assertEqual(result['ablation'],dict(status='RUNNING',tasks_started=1))
            self.assertNotIn('overall',result['ablation'])

    def test_summary_counts_real_checks_and_repetitions_but_bootstraps_groups(self):
        modes=['fixed','native_cached','f0','f1'];raw=dict(status='PASS_E16_ABLATION',contract=dict(modes=modes,repetitions=3),tasks=[],source_sha256={},model_sha256='test')
        for ti,language in enumerate(('en','zh')):
            task=dict(id=str(ti),group='paired_translation',language=language,family='explain',eos=True,output_cap=128,prompt_tokens=20,completion_tokens=4,
                      checks=[dict(mode=m,committed_logits_exact=True,live_state_exact=True,logit_rows=3) for m in modes],timings=[])
            for rep in range(3):
                order=modes[:]
                if rep%2:order.reverse()
                shift=(ti+rep)%len(order);order=order[shift:]+order[:shift]
                for slot,mode in enumerate(order):
                    task['timings'].append(dict(mode=mode,repetition=rep,order=slot,decode_ms={'fixed':6.,'native_cached':5.,'f0':4.,'f1':3.}[mode],exact=True,
                         steps=[dict(width=3,emitted=3,draft_feature_budget_ms=1.,verify_ms=1.,commit_ms=1.)]))
            raw['tasks'].append(task)
        result=online_summary(raw)
        self.assertEqual(result['formal_runs'],24);self.assertEqual(result['oracle_runs'],8)
        self.assertEqual(result['committed_logit_rows'],24)
        self.assertEqual(result['overall']['weighted_decode_tps']['f1'],1000.)
        self.assertEqual(result['overall']['comparisons']['fixed']['group_bootstrap_time_reduction_95'],[.5,.5])
        self.assertEqual(result['overall']['groups'],1)
        self.assertEqual(result['strata']['natural_eos']['tasks'],2)
        self.assertEqual(result['strata']['cap_reached']['tasks'],0)
        raw['tasks'][1]['group']='another_task_same_family'
        clustered=online_summary(raw)
        self.assertEqual(clustered['overall']['groups'],2)
        self.assertEqual(clustered['family_clustered_sensitivity']['groups'],1)
        self.assertEqual(clustered['overall']['weighted_decode_tps'],clustered['family_clustered_sensitivity']['weighted_decode_tps'])
        raw['tasks'][0]['timings'][0]['steps'][0]['emitted']=2
        with self.assertRaises(AssertionError):online_summary(raw)
