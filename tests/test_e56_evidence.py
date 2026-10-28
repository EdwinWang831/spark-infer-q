"""Reject post-hoc masks, relaxed gates and false decisions in actual records."""
import copy
import json
from pathlib import Path
import unittest
from scripts.e56_export import check_study,check_probe,export

RAW=Path(__file__).resolve().parents[1]/'results/raw/e56/2026-10-28-final/worker'


@unittest.skipUnless(RAW.exists(),'optional local E56 GPU archive')
class Evidence(unittest.TestCase):
    def test_frozen_selection_and_quality_decisions(self):
        original=json.loads((RAW/'stages/study-v1/result.json').read_text());check_study(original)
        for field in ('ranking','mask','format','teacher','threshold','decision','output','memory','coverage'):
            r=copy.deepcopy(original)
            if field=='ranking':r['ranking'].reverse()
            elif field=='mask':r['masks']['1']=[r['ranking'][-1]['layer']]
            elif field=='format':r['heldout'][-1]['actual_cache_formats']=['int4']*12
            elif field=='teacher':r['heldout'][-1]['teacher_forced_ids'][0]+=1
            elif field=='threshold':r['gates']['mean_relative_l2']=.2
            elif field=='decision':r['summaries']['mixed4']['quality_gate_passed']=not r['summaries']['mixed4']['quality_gate_passed']
            elif field=='output':r['heldout'][0]['greedy_tokens_exact']=False
            elif field=='memory':r['heldout'][0]['memory']['host_available_bytes']=1
            else:r['calibration'].pop()
            with self.subTest(field=field),self.assertRaises(ValueError):check_study(r)

    def test_probe_rejects_false_storage_claim_and_changed_intervention(self):
        study=json.loads((RAW/'stages/study-v1/result.json').read_text())
        probe=json.loads((RAW/'stages/probe-v1/result.json').read_text());check_probe(probe,study)
        for field in ('capacity','scale','rows','metric'):
            r=copy.deepcopy(probe)
            if field=='capacity':r['capacity_reduction_validated']=True
            elif field=='scale':r['scale_factors']=[1.]
            elif field=='rows':r['runs'][-1]['intervention']['rows']-=1
            else:r['summaries']['search']['mean_kl']=0.
            with self.subTest(field=field),self.assertRaises(ValueError):check_probe(r,study)

    def test_complete_sealed_export(self):
        result=export(RAW)
        self.assertEqual(result['status'],'PASS_E56_EVIDENCE')
        self.assertFalse(result['quality_promoted'])
        self.assertFalse(result['eight_execution_validated'])
        self.assertGreater(result['capacity_forecasts']['mixed1']['bytes'],result['capacity_forecasts']['int4']['bytes'])
