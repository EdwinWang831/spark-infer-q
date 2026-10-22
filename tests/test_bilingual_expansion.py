import collections
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from e16_tasks import TASKS, CONTRACT, selected_positions
from e16_ablation_tasks import TASKS as ABLATION
from e15_tasks import TASKS as PILOT
from e16_online_tasks import TASKS as ONLINE


class BilingualExpansionTests(unittest.TestCase):
    def test_languages_are_balanced_in_every_split_and_groups_never_leak(self):
        self.assertEqual(len(TASKS), 600)
        self.assertEqual(len({t['id'] for t in TASKS}), 600)
        groups=collections.defaultdict(list)
        for t in TASKS:groups[t['group']].append(t)
        self.assertEqual(len(groups), 120)
        for tasks in groups.values():
            self.assertEqual(len({t['split'] for t in tasks}), 1)
            self.assertEqual(collections.Counter(t['language'] for t in tasks), {'en':2,'zh':2,'mixed':1})
        for split, count in CONTRACT['splits'].items():
            tasks=[t for t in TASKS if t['split']==split]
            counts=collections.Counter(t['language'] for t in tasks)
            self.assertEqual(counts, {'en': count*2//5,'zh':count*2//5,'mixed':count//5})
        self.assertEqual(CONTRACT['splits'],dict(train=300,calibration=100,test=100,ood=100))

    def test_ood_families_and_independent_ablation_are_held_out(self):
        for t in TASKS:self.assertEqual(t['split']=='ood', t['family'].startswith('ood_'))
        training={t['prompt'] for t in TASKS}
        self.assertFalse(training & {t['prompt'] for t in ABLATION})
        self.assertFalse(training & {t['prompt'] for t in PILOT})
        self.assertFalse(training & {t['prompt'] for t in ONLINE})
        self.assertFalse({t['prompt'] for t in ONLINE} & {t['prompt'] for t in ABLATION})
        self.assertEqual(collections.Counter(t['language'] for t in ONLINE), {'en':8,'zh':8,'mixed':4})
        self.assertEqual(len(training),600)

    def test_subsampling_spans_the_whole_trajectory_and_handles_short_eos(self):
        self.assertEqual(selected_positions(0), [])
        self.assertEqual(selected_positions(3), [0,1,2])
        for n in (65,128,383,1023):
            p=selected_positions(n)
            self.assertEqual(len(p),64)
            self.assertEqual(p[0],0);self.assertEqual(p[-1],n-1)
            self.assertEqual(p, sorted(set(p)))
            gaps=[b-a for a,b in zip(p,p[1:])]
            self.assertLessEqual(max(gaps)-min(gaps),1)

    def test_long_context_plus_output_respects_target_and_draft_capacity(self):
        for t in TASKS:
            if 'target_prompt_tokens' in t:
                self.assertLessEqual(t['target_prompt_tokens']+t['output_cap']+8, CONTRACT['capacity'])
        for lang in ('en','zh'):
            self.assertTrue(any(t['language']==lang and t['output_cap']==1024 for t in TASKS))
