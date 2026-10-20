"""Request lifecycle tests with delayed completion and injected backend faults."""
import copy
import unittest

from spark_infer_q.executor import (BackendContractError, Busy, Executor, Pending,
                                   Row, StaleHandle, WorkClosed)


class Fence:
    def __init__(self):
        self.done, self.broken = False, False

    def ready(self):
        if self.broken:
            raise RuntimeError('device unavailable')
        return self.done

    def wait(self):
        self.done = True


class Backend:
    def __init__(self):
        self.model_calls = 0
        self.fail_launch = False
        self.bad_advance = False
        self.fence = None

    def create(self):
        return {'history': [], 'version': 0, 'capacity': 2}

    def ensure(self, state, rows):
        if rows > 32:
            raise ValueError('capacity limit')
        if rows > state['capacity']:
            state['capacity'], state['version'] = rows, state['version'] + 1

    def layout(self, state):
        return state['version']

    def position(self, state):
        return len(state['history'])

    def checkpoint(self, state):
        return copy.deepcopy(state)

    def restore(self, state, saved):
        capacity, version = state['capacity'], state['version']
        state.clear()
        state.update(copy.deepcopy(saved), capacity=capacity, version=version)

    def validate_inputs(self, ticket, values):
        if len(values) != ticket.row_count:
            raise ValueError('wrong input rows')

    def launch(self, ticket, states, values):
        out = [0] * ticket.row_count
        for seg, state in zip(ticket.segments, states):
            self.model_calls += 1
            for row in seg.indices:
                state['history'].append(values[row])
                out[row] = sum(state['history'])
            if self.bad_advance:
                state['history'].append(999)
        if self.fail_launch:
            raise RuntimeError('injected failure after mutation')
        self.fence = Fence()
        return out, self.fence

    def mask_rows(self, values, valid):
        return [v if i in valid else 0 for i, v in enumerate(values)]

    def synchronize(self):
        pass

    def release(self, state):
        pass

    def export_state(self, state):
        return copy.deepcopy(state)


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.ex = Executor(self.backend)

    def run_rows(self, rows, values, mode='decode'):
        ticket = self.ex.prepare(rows, mode)
        self.ex.dispatch(ticket, values)
        return self.ex.finish(ticket)

    def test_interleaved_requests_and_padding_preserve_positions(self):
        a, b = self.ex.admit('A'), self.ex.admit('B')
        got = self.run_rows([Row(a, 0), Row(), Row(b, 0), Row(a, 1), Row(b, 1)],
                            [2, float('nan'), 10, 3, 4], 'prefill')
        self.assertEqual(got.output, [2, 0, 10, 5, 14])
        self.assertEqual(got.row_status, ('COMMITTED', 'PADDING', 'COMMITTED', 'COMMITTED', 'COMMITTED'))
        self.assertEqual(self.ex.debug_state(a)['history'], [2, 3])
        self.assertEqual(self.ex.debug_state(b)['history'], [10, 4])

    def test_logical_position_is_not_published_before_fence(self):
        h = self.ex.admit('A')
        ticket = self.ex.prepare([Row(h, 0)])
        self.ex.dispatch(ticket, [7])
        self.assertEqual(self.ex.view(h)['position'], 0)
        with self.assertRaises(Pending):
            self.ex.finish(ticket, wait=False)
        with self.assertRaises(Busy):
            self.ex.debug_state(h)
        self.backend.fence.done = True
        self.assertEqual(self.ex.finish(ticket).output, [7])
        self.assertEqual(self.ex.view(h)['position'], 1)

    def test_cancelled_inflight_slot_is_quarantined_then_generation_changes(self):
        self.ex = Executor(self.backend, max_requests=1)
        old = self.ex.admit('same-id')
        ticket = self.ex.prepare([Row(old, 0)])
        self.ex.dispatch(ticket, [99])
        self.ex.cancel(old)
        with self.assertRaises(Busy):
            self.ex.admit('same-id')
        with self.assertRaises(StaleHandle):
            self.ex.view(old)
        got = self.ex.finish(ticket)
        self.assertEqual(got.output, [0])
        self.assertEqual(got.row_status, ('CANCELED',))
        new = self.ex.admit('same-id')
        self.assertEqual(new.slot, old.slot)
        self.assertGreater(new.generation, old.generation)
        with self.assertRaises(WorkClosed):
            self.ex.finish(ticket)
        with self.assertRaises(StaleHandle):
            self.ex.prepare([Row(old, 0)])
        self.assertEqual(self.run_rows([Row(new, 0)], [4]).output, [4])

    def test_one_cancelled_request_does_not_discard_other_completion(self):
        self.ex = Executor(self.backend, max_requests=3)
        a, b = self.ex.admit('A'), self.ex.admit('B')
        ticket = self.ex.prepare([Row(a, 0), Row(b, 0)])
        self.ex.dispatch(ticket, [2, 8])
        self.ex.cancel(a)
        replacement = self.ex.admit('A')
        got = self.ex.finish(ticket)
        self.assertEqual(got.output, [0, 8])
        self.assertEqual(got.committed, (b,))
        self.assertEqual(self.ex.view(replacement)['position'], 0)
        with self.assertRaises(Busy):
            self.ex.admit('A')

    def test_cancellation_before_dispatch_invalidates_plan_without_compute(self):
        a, b = self.ex.admit('A'), self.ex.admit('B')
        ticket = self.ex.prepare([Row(a, 0), Row(b, 0)])
        self.ex.cancel(a)
        with self.assertRaises(WorkClosed):
            self.ex.dispatch(ticket, [1, 2])
        self.assertFalse(self.ex.view(b)['busy'])
        self.assertEqual(self.backend.model_calls, 0)
        self.assertEqual(self.run_rows([Row(b, 0)], [3]).output, [3])

    def test_bad_metadata_does_not_reserve_valid_earlier_rows(self):
        a, b = self.ex.admit('A'), self.ex.admit('B')
        with self.assertRaises(ValueError):
            self.ex.prepare([Row(a, 0), Row(b, 4)])
        self.assertFalse(self.ex.view(a)['busy'])
        self.assertEqual(self.ex.debug_state(a)['history'], [])
        foreign = Executor(Backend()).admit('A')
        with self.assertRaises(StaleHandle):
            self.ex.prepare([Row(foreign, 0)])

    def test_duplicate_or_reordered_positions_are_rejected(self):
        h = self.ex.admit('A')
        for positions in ([0, 0], [1, 0], [0, 2]):
            with self.assertRaises(ValueError):
                self.ex.prepare([Row(h, p) for p in positions])

    def test_input_contract_error_can_be_discarded_without_state_change(self):
        h = self.ex.admit('A')
        ticket = self.ex.prepare([Row(h, 0)])
        with self.assertRaises(ValueError):
            self.ex.dispatch(ticket, [])
        self.ex.discard(ticket)
        self.assertEqual(self.ex.view(h)['position'], 0)
        self.assertEqual(self.backend.model_calls, 0)

    def test_launch_failure_restores_prefix_and_releases_reservation(self):
        h = self.ex.admit('A')
        self.run_rows([Row(h, 0)], [5])
        self.backend.fail_launch = True
        ticket = self.ex.prepare([Row(h, 1)])
        with self.assertRaises(RuntimeError):
            self.ex.dispatch(ticket, [100])
        self.assertEqual(self.ex.debug_state(h)['history'], [5])
        self.backend.fail_launch = False
        self.assertEqual(self.run_rows([Row(h, 1)], [2]).output, [7])

    def test_bad_backend_advance_rolls_back_whole_batch(self):
        a, b = self.ex.admit('A'), self.ex.admit('B')
        self.backend.bad_advance = True
        ticket = self.ex.prepare([Row(a, 0), Row(b, 0)])
        self.ex.dispatch(ticket, [2, 3])
        with self.assertRaises(BackendContractError):
            self.ex.finish(ticket)
        self.assertEqual(self.ex.debug_state(a)['history'], [])
        self.assertEqual(self.ex.debug_state(b)['history'], [])

    def test_checkpoint_is_independent_and_cannot_cross_requests(self):
        a, b = self.ex.admit('A'), self.ex.admit('B')
        self.run_rows([Row(a, 0)], [2])
        snap = self.ex.checkpoint(a)
        self.run_rows([Row(a, 1)], [10])
        self.ex.restore(a, snap)
        self.assertEqual(self.run_rows([Row(a, 1)], [3]).output, [5])
        with self.assertRaises(StaleHandle):
            self.ex.restore(b, snap)
        self.ex.cancel(a)
        new = self.ex.admit('A')
        with self.assertRaises(StaleHandle):
            self.ex.restore(new, snap)

    def test_checkpoint_budget_and_busy_snapshot(self):
        self.ex = Executor(self.backend, max_checkpoints=1)
        h = self.ex.admit('A')
        snap = self.ex.checkpoint(h)
        with self.assertRaises(Busy):
            self.ex.checkpoint(h)
        self.ex.drop_checkpoint(snap)
        ticket = self.ex.prepare([Row(h, 0)])
        with self.assertRaises(Busy):
            self.ex.checkpoint(h)
        self.ex.discard(ticket)

    def test_layout_change_rejects_prepared_work(self):
        h = self.ex.admit('A')
        ticket = self.ex.prepare([Row(h, 0)])
        self.ex._slots[h.slot].state['version'] += 1  # deliberately violate backend ownership
        with self.assertRaises(StaleHandle):
            self.ex.dispatch(ticket, [1])
        self.ex.discard(ticket)
        self.assertEqual(self.backend.model_calls, 0)

    def test_disjoint_prepared_batches_are_serialized_at_dispatch(self):
        a, b = self.ex.admit('A'), self.ex.admit('B')
        first = self.ex.prepare([Row(a, 0)])
        second = self.ex.prepare([Row(b, 0)])
        self.ex.dispatch(first, [2])
        with self.assertRaises(Busy):
            self.ex.dispatch(second, [3])
        self.ex.finish(first)
        self.ex.dispatch(second, [3])
        self.assertEqual(self.ex.finish(second).output, [3])

    def test_failed_fence_prevents_reuse_and_new_work(self):
        h = self.ex.admit('A')
        ticket = self.ex.prepare([Row(h, 0)])
        self.ex.dispatch(ticket, [1])
        self.backend.fence.broken = True
        with self.assertRaises(BackendContractError):
            self.ex.finish(ticket)
        with self.assertRaises(BackendContractError):
            self.ex.admit('B')

    def test_all_padding_has_no_model_calls(self):
        out = self.run_rows([Row(), Row()], [float('nan'), float('nan')])
        self.assertEqual(out.output, [0, 0])
        self.assertEqual(out.row_status, ('PADDING', 'PADDING'))
        self.assertEqual(self.backend.model_calls, 0)


if __name__ == '__main__':
    unittest.main()
