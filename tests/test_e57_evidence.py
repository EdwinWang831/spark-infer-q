"""Reject false promotions, leaked cuts and biased native-decode summaries."""
import copy
import json
from pathlib import Path
import unittest
from scripts.e57_export import check_screen,check_deferred,export

RAW=Path(__file__).resolve().parents[1]/'results/raw/e57/2026-10-28-final/worker'


@unittest.skipUnless(RAW.exists(),'optional local E57 GPU archive')
class Evidence(unittest.TestCase):
    def test_screen_scope_causality_and_quality(self):
        original=json.loads((RAW/'stages/screen-v1/result.json').read_text());check_screen(original)
        for field in ('promotion','cut','teacher','output','layout','codec','threshold'):
            r=copy.deepcopy(original)
            if field=='promotion':r['compressed_resident_state']=True
            elif field=='cut':r['runs'][-1]['intervention']['max_quantized_prefix']+=32
            elif field=='teacher':r['runs'][-1]['teacher_forced_ids'][0]+=1
            elif field=='output':r['runs'][0]['greedy_tokens_exact']=False
            elif field=='layout':r['summaries']['row32_r0']['projected_kv_layout']['metadata_bytes']=0
            elif field=='codec':r['codec_cases'][0]['graph_changed_input_exact']=False
            else:r['gates']['mean_relative_l2']=.2
            with self.subTest(field=field),self.assertRaises(ValueError):check_screen(r)

    def test_native_conversion_and_exclusion_of_first_logit(self):
        original=json.loads((RAW/'stages/deferred-v1/result.json').read_text());check_deferred(original)
        for field in ('preservation','format','storage','decode','graph','decision'):
            r=copy.deepcopy(original);x=next(x for x in r['runs'] if x['mode']=='deferred')
            if field=='preservation':x['nonkv_preserved']=False
            elif field=='format':x['actual_cache_formats']=['bf16']*12
            elif field=='storage':x['post_conversion_storage_bytes']+=1
            elif field=='decode':x['decode_teacher_metrics']=copy.deepcopy(x['metrics'])
            elif field=='graph':x['conversion_graphs_reused']=1
            else:r['summaries']['deferred']['decode_only']['screen_passed']=True
            with self.subTest(field=field),self.assertRaises(ValueError):check_deferred(r)

    def test_complete_source_and_audit_chain(self):
        r=export(RAW)
        self.assertEqual(r['status'],'PASS_E57_EVIDENCE')
        self.assertFalse(r['quality_promoted'])
        self.assertTrue(r['deferred_decode']['actual_native_int4_decode'])
        self.assertFalse(r['eight_execution_validated'])
