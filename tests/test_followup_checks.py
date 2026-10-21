import unittest
from scripts.e3_followup_work import selected_check


class SelectedChecks(unittest.TestCase):
    def test_thought_does_not_substitute_for_final_answer(self):
        task={'id':'reasoning_math'}
        self.assertFalse(selected_check(task,'<think>Maybe 55, but I will answer 56.</think>56<|im_end|>'))
        self.assertTrue(selected_check(task,'<think>Check the arithmetic.</think>55<|im_end|>'))

    def test_eos_markers_do_not_break_literal_or_json_checks(self):
        self.assertTrue(selected_check({'id':'addition'},'42<|im_end|>'))
        self.assertTrue(selected_check({'id':'json'},'{"name":"spark","count":3}<|im_end|>'))
        self.assertFalse(selected_check({'id':'json'},'```json\n{"name":"spark","count":3}\n```'))
