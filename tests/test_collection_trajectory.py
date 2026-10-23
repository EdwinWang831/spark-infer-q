import copy
from types import SimpleNamespace
import unittest

from spark_infer_q.collection_trajectory import collect_trajectory
from spark_infer_q.confidence_head import cumulative_labels
from spark_infer_q.draft_fidelity import emitted_by_width


class Rows:
    def __init__(self, rows=None, *, storage=None, start=0, stop=None):
        self.storage = list(rows) if storage is None else storage
        self.start = start
        self.stop = len(self.storage) if stop is None else stop

    def __len__(self):
        return self.stop-self.start

    def __iter__(self):
        return iter(self.storage[self.start:self.stop])

    @property
    def shape(self):
        return len(self), 2

    def __getitem__(self, key):
        if isinstance(key, slice):
            a, b, step = key.indices(len(self))
            assert step == 1
            return Rows(storage=self.storage, start=self.start+a, stop=self.start+b)
        if not 0 <= key < len(self):
            raise IndexError(key)
        return self.storage[self.start+key]

    def copy_(self, other):
        self.storage[self.start:self.stop] = list(other)

    def new_empty(self, shape):
        return Rows([None]*shape[0])


class Logits:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, key):
        value = self.values[key]
        return Logits(value if isinstance(key, slice) else [value])

    def argmax(self, *args):
        return self

    def tolist(self):
        return self.values

    def __int__(self):
        assert len(self.values) == 1
        return self.values[0]

    def cpu(self):
        return self


class State:
    def __init__(self, history=(0,)):
        self.history = list(history)

    @property
    def pos(self):
        return len(self.history)

    def clone(self):
        return copy.deepcopy(self)


class Target:
    capacity = 64
    torch = SimpleNamespace(cat=lambda items: Logits(v for item in items for v in item.values))

    def run(self, state, ids):
        history = state.history[:]
        features = []
        for token in ids:
            history.append(token)
            features.append((tuple(history), sum(history)))
        return Logits(t+1 for t in ids), Rows(features), history

    def commit(self, state, buffer, rows, keep):
        state.history[:] = buffer[:state.pos+keep]

    def step(self, state, ids):
        lg, ft, buf = self.run(state, ids)
        self.commit(state, buf, len(ids), len(ids))
        return lg, ft


class Draft:
    def __init__(self, reject_at=None):
        self.reject_at = reject_at
        self.requests = []
        self.prefixes = []

    def begin(self, request):
        self.requests.append(request)
        self.prefixes = []

    def propose(self, context, anchor):
        self.prefixes.append(list(context))
        tokens = [anchor+i for i in range(1, 8)]
        if self.reject_at is not None:
            tokens[self.reject_at] = 500
        return tokens, {}


class CollectionTrajectoryTests(unittest.TestCase):
    def collect(self, **kwargs):
        return collect_trajectory(Target(), State(), Rows([((0,), 0)]), 1,
                                  request='new', **kwargs)

    def test_each_rejection_eos_and_cap_preserve_all_features_and_state(self):
        for reject in (None, *range(7)):
            for cap in (1, 2, 7, 8, 9, 18):
                for eos in ((), (1,), (4,), (9,)):
                    golden = self.collect(output_cap=cap, eos_ids=eos, trace=True)
                    fast = self.collect(output_cap=cap, eos_ids=eos, trace=True, draft=Draft(reject))
                    self.assertEqual(fast['tokens'], golden['tokens'])
                    self.assertEqual(list(fast['features']), list(golden['features']))
                    self.assertEqual(fast['state'].history, golden['state'].history)
                    self.assertEqual(fast['logits'].values if fast['logits'] else None,
                                     golden['logits'].values if golden['logits'] else None)
                    for position in range(len(fast['tokens'])-1):
                        draft = [fast['tokens'][position]+i for i in range(1, 8)]
                        a = emitted_by_width(draft, fast['tokens'][position+1:], eos, max_new=cap-position-1)
                        b = emitted_by_width(draft, golden['tokens'][position+1:], eos, max_new=cap-position-1)
                        self.assertEqual(cumulative_labels(a), cumulative_labels(b))

    def test_calibration_keeps_individual_early_snapshots_then_groups_rows(self):
        golden = self.collect(output_cap=24, early_serial=8, trace=True)
        fast = self.collect(output_cap=24, early_serial=8, draft=Draft(), trace=True)
        self.assertEqual([s.history for s in fast['early_states']],
                         [s.history for s in golden['early_states']])
        self.assertEqual([v.values for v in fast['early_logits']],
                         [v.values for v in golden['early_logits']])
        self.assertEqual(list(fast['features']), list(golden['features']))
        self.assertLess(fast['target_calls'], golden['target_calls'])

    def test_rejected_future_features_never_reach_draft_context(self):
        draft = Draft(1)
        fast = self.collect(output_cap=15, draft=draft)
        self.assertGreater(len(draft.prefixes), 1)
        self.assertTrue(all(500 not in feature[0] for prefix in draft.prefixes for feature in prefix))
        self.assertNotIn(500, fast['state'].history)

    def test_invalid_requests_and_capacity_fail_before_execution(self):
        for kwargs in (dict(output_cap=0), dict(output_cap=65), dict(output_cap=2, width=0),
                       dict(output_cap=2, early_serial=-1)):
            with self.assertRaises(ValueError):
                self.collect(**kwargs)
        with self.assertRaises(ValueError):
            collect_trajectory(Target(), State(), Rows([((0,), 0)]), 1,
                               output_cap=2, draft=Draft())


if __name__ == '__main__':
    unittest.main()
