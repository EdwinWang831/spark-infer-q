"""Tamper real archived records when available; optional archive is not GPU proof."""
import copy
import json
from pathlib import Path
import unittest
from scripts.e54_export import check_study,check_probes

RAW=Path(__file__).resolve().parents[1]/'results/raw/e54/2026-10-26-final/worker'


@unittest.skipUnless(RAW.exists(),'optional local E54 GPU archive')
class Evidence(unittest.TestCase):
    def load_records(self):
        return (json.loads((RAW/'stages/study-v1/result.json').read_text()),
                json.loads((RAW/'references/e53-multi-v3/result.json').read_text()))

    def test_real_study_rejects_state_decision_cache_and_fake_capacity(self):
        original,prior=self.load_records();check_study(original,prior)
        for field in ('state','rates','cache','concurrency'):
            r=copy.deepcopy(original);run=next(x for x in r['runs'] if x['name']=='two-256k')
            if field=='state':run['states']['long-a']['features']='0'*64
            elif field=='rates':
                m=next(m for m in run['metrics'] if m['request_id']=='long-a' and not m['prompt'])
                m['rates'][0]+=.001
            elif field=='cache':run['max_live_projection_bytes']=50*1024**2
            else:run['max_populated_256k_live']=1
            with self.subTest(field=field),self.assertRaises(ValueError):check_study(r,prior)

    def test_real_probe_rejects_false_equality_and_modified_real_cache(self):
        original,prior=self.load_records();check_probes(original['probes'],prior)
        for field in ('equality','preserved','work'):
            probes=copy.deepcopy(original['probes'])
            p=next(p for p in probes if not p['rates_exact'])
            if field=='equality':p['rates_exact']=True
            elif field=='preserved':p['request_cache_preserved']=False
            else:p['rebuilt_rows']=1
            with self.subTest(field=field),self.assertRaises(ValueError):check_probes(probes,prior)
