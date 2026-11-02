"""Preserve failed raw-logit gate even when all sampled answers agree."""
import unittest
from scripts.e47_export import pilot_summary


def fixture():
    v=dict(status='STOP_E47_QUALITY',selected_kv_dtype=None,gates=dict(mean_kl=.02,mean_relative_l2=.03,teacher_top1=.95),
        numeric_errors={},numeric=[],graphs={},quality=[],summaries={})
    for fmt in ('int8','fp8'):
        v['graphs'][fmt]=dict(graph_captures=16,graph_replays=32,graph_fallbacks={})
        for name,n in (('english64',64),('boundary2049',2049)):
            v['numeric'].append(dict(kv_dtype=fmt,name=name,prompt_tokens=n,logits_exact=True,features_exact=True,state_differences=[],
                windows=[dict(width=w,parity=p,keep=w if w%2 else w//2,exact=True,state_differences=[]) for w in range(1,9) for p in (0,1)]))
    for fmt in ('bf16','int8','fp8'):
        for name in ('en','zh','math','json','code','explain','long_en','long_zh'):
            exact=not(fmt=='fp8' and name=='explain')
            v['quality'].append(dict(name=name,kv_dtype=fmt,teacher_rows=8,teacher_top1_matches=8,output_tokens=24,
                prompt_tokens=4096 if name.startswith('long') else 23,input_sha256='a'*64,
                output_sha256=('b' if exact else 'c')*64,greedy_tokens_exact=exact,
                mean_kl=0. if fmt=='bf16' else .0004,mean_relative_l2=0. if fmt=='bf16' else .065,
                objective_passed=None if name=='explain' else True,bf16_objective_passed=None if name=='explain' else True))
        v['summaries'][fmt]=dict(teacher_rows=64,mean_kl=0. if fmt=='bf16' else .0004,
            mean_relative_l2=0. if fmt=='bf16' else .065,teacher_top1=1.,lost_objectives=[],
            greedy_exact_cases=7 if fmt=='fp8' else 8,quality_gate_passed=fmt=='bf16')
    return v


class FailedPrecisionGate(unittest.TestCase):
    def test_matching_answers_do_not_override_raw_logit_failure(self):
        v=pilot_summary(fixture())
        self.assertEqual(v['int8']['greedy_exact_cases'],8)
        self.assertEqual(v['int8']['teacher_top1'],1.)
        self.assertFalse(v['int8']['quality_gate_passed'])

    def test_gate_threshold_selected_format_and_summary_cannot_be_changed(self):
        for case in ('threshold','selection','summary'):
            v=fixture()
            if case=='threshold':v['gates']['mean_relative_l2']=.1
            elif case=='selection':v['selected_kv_dtype']='int8'
            else:v['summaries']['int8']['quality_gate_passed']=True
            with self.subTest(case=case),self.assertRaises(ValueError):pilot_summary(v)

    def test_missing_scale_state_or_long_boundary_cannot_publish(self):
        for case in ('scale','length','bucket'):
            v=fixture()
            if case=='scale':v['numeric'][0]['windows'][0]['state_differences']=['kv_0_ks']
            elif case=='length':v['numeric'][1]['prompt_tokens']=64
            else:v['numeric'][0]['windows'].pop()
            with self.subTest(case=case),self.assertRaises(ValueError):pilot_summary(v)

    def test_output_hash_must_support_greedy_equivalence_flag(self):
        v=fixture();v['quality'][8]['output_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'hash'):pilot_summary(v)


if __name__=='__main__':unittest.main()
