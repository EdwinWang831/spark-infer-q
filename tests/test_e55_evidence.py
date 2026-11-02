"""Reject changed decisions and traces in real E55 archives, when available."""
import copy
import json
from pathlib import Path
import unittest
from scripts.e55_export import check_quality,check_long,export

RAW=Path(__file__).resolve().parents[1]/'results/raw/e55/2026-10-27-final/worker'


@unittest.skipUnless(RAW.exists(),'optional local E55 GPU archive')
class Evidence(unittest.TestCase):
    def test_quality_rejects_false_promotion_threshold_and_output(self):
        original=json.loads((RAW/'stages/quality-v1/result.json').read_text())
        check_quality(original)
        for field in ('promotion','threshold','output','numeric'):
            r=copy.deepcopy(original)
            if field=='promotion':r['summaries']['int4']['quality_gate_passed']=True
            elif field=='threshold':r['gates']['mean_relative_l2']=.2
            elif field=='output':r['quality'][-1]['output_ids'][0]+=1
            else:r['numeric'][0]['windows'][0]['exact']=False
            with self.subTest(field=field),self.assertRaises(ValueError):check_quality(r)

    def test_long_rejects_state_teacher_graph_and_concurrency(self):
        stage=RAW/'stages/long-v3';original=json.loads((stage/'result.json').read_text())
        check_long(original,stage)
        for field in ('state','teacher','graph','concurrency','tokens','memory'):
            r=copy.deepcopy(original);row=r['runs'][1]
            if field=='state':row['greedy_state']={'changed':True}
            elif field=='teacher':row['teacher_metrics']['mean_kl']+=.01
            elif field=='graph':row['capabilities']['head_fallbacks']=1
            elif field=='concurrency':r['eight_execution_validated']=True
            elif field=='tokens':row['greedy_output_ids'][0]+=1
            else:row['minimum_host_available_bytes']=1
            with self.subTest(field=field),self.assertRaises(ValueError):check_long(r,stage)

    def test_complete_provenance_and_negative_decision(self):
        r=export(RAW)
        self.assertEqual(r['status'],'PASS_E55_EVIDENCE_CANDIDATE_NOT_PROMOTED')
        self.assertFalse(r['quality_promoted'])
        self.assertFalse(r['eight_execution_validated'])
        self.assertGreater(r['capacity']['deficit_bytes'],0)
