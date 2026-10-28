"""Tamper real archived native-conversion and shared-prefix study records."""
import copy
import json
from pathlib import Path
import unittest
from scripts.e58_export import check_gate,check_study,export

RAW=Path(__file__).resolve().parents[1]/'results/raw/e58/2026-10-28-final/worker'


@unittest.skipUnless(RAW.exists(),'optional local E58 GPU archive')
class Evidence(unittest.TestCase):
    def test_native_prep_gate_and_codec_graph(self):
        original=json.loads((RAW/'stages/gate-v2/result.json').read_text());check_gate(original)
        for field in ('equality','offset','graph'):
            r=copy.deepcopy(original)
            if field=='equality':r['cases'][-1]['native_prep_exact']=False
            elif field=='offset':r['cases'][-1]['start']=0
            else:r['graphs'][-1]['changed_input_exact']=False
            with self.subTest(field=field),self.assertRaises(ValueError):check_gate(r)

    def test_prefix_reuse_conversion_and_reproduction(self):
        original=json.loads((RAW/'stages/study-v1/result.json').read_text())
        prior=json.loads((RAW/'references/648-e57-deferred-v1.json').read_text())['result'];check_study(original,prior)
        for field in ('prefill','mask','ownership','format','storage','teacher','first_logit','native_match','reproduction','threshold'):
            r=copy.deepcopy(original);x=next(x for x in r['runs'] if x['mode']=='mixed2')
            if field=='prefill':r['prefill_count']=36
            elif field=='mask':r['masks']['mixed2']=[7,11]
            elif field=='ownership':x['source_prefix_preserved']=False
            elif field=='format':x['actual_cache_formats']=['int4']*12
            elif field=='storage':x['post_conversion_storage_bytes']+=1
            elif field=='teacher':x['teacher_forced_ids'][0]+=1
            elif field=='first_logit':x['decode_teacher_metrics']=copy.deepcopy(x['metrics'])
            elif field=='native_match':r['native_torch_matches'][0]['full_logits_exact']=False
            elif field=='reproduction':r['reproduction'][0]['output']=False
            else:r['gates']['mean_relative_l2']=.2
            with self.subTest(field=field),self.assertRaises(ValueError):check_study(r,prior)

    def test_complete_archive_and_teacher_artifacts(self):
        r=export(RAW)
        self.assertEqual(r['status'],'PASS_E58_EVIDENCE')
        self.assertEqual(len(r['teacher_artifacts']),6)
        self.assertEqual(r['study']['prefill_count'],6)
        self.assertFalse(r['quality_promoted'])
        self.assertFalse(r['eight_execution_validated'])
