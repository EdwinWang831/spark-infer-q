"""Check the perfect-foresight headroom accounting on a small fixture."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from oracle_headroom import cost_curve, headroom, runs_by_task, zero_persistence


def step(rows, accepted, observed=None, draft=10.):
    return {'verified_rows': rows, 'accepted': accepted, 'observed': rows-1 if observed is None else observed,
            'draft_ms': draft, 'verify_ms': 40.+4*(rows-1)}


class OracleHeadroomTests(unittest.TestCase):
    def test_post_and_pre_draft_savings(self):
        run = {'mode': 'm', 'decode_ms': 400., 'extra': {'steps': [step(6, 5), step(6, 2), step(6, 0), step(1, 0, 0)]}}
        data = {'evidence': {'tasks': [{'id': 'a', 'timings': [run, dict(run, mode='other')]}]}}
        runs = runs_by_task(data, 'm')['a']
        self.assertEqual(len(runs), 1)
        cost, curve = cost_curve([s for t in runs for s in t['extra']['steps']])
        self.assertEqual(curve, {6: 60., 1: 40.})
        self.assertEqual(cost(3), 48.)
        self.assertEqual(cost(9), 60.)
        result = headroom(runs, cost)
        # Shorter verify: 0 + (60-48) + (60-40) ms; skip draft only on the drafted zero step.
        self.assertAlmostEqual(result['post_draft_fraction'], 32/400)
        self.assertAlmostEqual(result['pre_draft_extra_fraction'], 10/400)
        self.assertAlmostEqual(result['zero_accept_fraction'], 1/3)
        self.assertAlmostEqual(result['all_verified_accepted_fraction'], 1/3)

    def test_zero_acceptance_persistence(self):
        run = {'extra': {'steps': [step(4, 0), step(4, 0), step(4, 2), step(4, 0), step(1, 0, 0), step(4, 1)]}}
        result = zero_persistence([run])
        self.assertEqual(result['pairs'], 4)
        self.assertAlmostEqual(result['zero_after_zero'], 1/3)
        self.assertAlmostEqual(result['zero_after_nonzero'], 1.)


if __name__ == '__main__': unittest.main()
