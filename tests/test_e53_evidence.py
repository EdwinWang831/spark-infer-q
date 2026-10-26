"""Tamper checks on optional real archived GPU records; never fake GPU proof."""
import copy
import json
from pathlib import Path
import unittest

from scripts.e53_export import check_capacity,check_multi

RAW=Path(__file__).resolve().parents[1]/'results/raw/e53/2026-10-26-final/worker/stages'


@unittest.skipUnless(RAW.exists(),'optional local E53 GPU archive')
class Evidence(unittest.TestCase):
    def test_real_capacity_rejects_invented_headroom_and_missing_graph_cost(self):
        original=json.loads((RAW/'capacity-v3/result.json').read_text())
        check_capacity(original)
        for field,value in (('capacity_reserve_bytes',0),('conservative_slot_bytes',1),
                            ('target_graph_captures',[16,0]),('actual_full256k_requests',8)):
            changed=copy.deepcopy(original);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):check_capacity(changed)

    def test_real_multi_rejects_state_budget_and_fake_concurrency(self):
        original=json.loads((RAW/'multi-v3/result.json').read_text())
        check_multi(original)
        for field in ('state','budget','concurrency'):
            changed=copy.deepcopy(original)
            combined=next(r for r in changed['runs'] if r['name']=='two-256k')
            if field=='state':combined['states']['long-b']['context_start']+=1
            elif field=='budget':
                metric=next(m for m in combined['metrics'] if not m['prompt'] and m['request_id']=='long-b')
                metric['verified_rows']=8 if metric['verified_rows']!=8 else 7
            else:combined['max_populated_256k_live']=1
            with self.subTest(field=field),self.assertRaises(ValueError):check_multi(changed)
