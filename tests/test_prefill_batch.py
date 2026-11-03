"""Batch ownership, validation atomicity, shared accounting and phase lifetime."""
from contextlib import nullcontext
from types import SimpleNamespace as NS
import unittest

from spark_infer_q.deepinfer_bridge import BridgeError, RequestHandle, SchedulerBridge, WorkRow
from spark_infer_q.kaifra_speculative import SpeculativeBackend, SpeculativeSchedulerBridge, VerifiedStep
from spark_infer_q.prefill_batch import BatchPrefillExecutor
from spark_infer_q.prefill_workspace import workspace_layout
from spark_infer_q.prompt_batch import PromptBatchControl
from test_prefill_executor import Provider, state


class BatchProvider(Provider):
    def allocate(self, rows):
        self.allocated_rows = rows
        return super().allocate(rows)

    def run_many(self, workspace, windows, capture):
        return tuple(self.run(workspace, s, ids, capture) for s, ids in windows)


class BatchPrefillTests(unittest.TestCase):
    def test_four_private_histories_one_workspace_and_one_fence(self):
        p = BatchProvider()
        with BatchPrefillExecutor(p, chunk_rows=8) as executor:
            states = [state() for _ in range(4)]
            result = executor.advance_many([(s, range(i+1)) for i, s in enumerate(states)])
            self.assertEqual(p.allocated_rows, 32)
            self.assertEqual((p.allocations, p.fences), (1, 1))
            self.assertEqual([v.chunks for v in result], [(1,), (2,), (3,), (4,)])
            self.assertEqual([v.end_position for v in result], [1, 2, 3, 4])
            for i, value in enumerate(result):
                self.assertEqual(value.features[-1], tuple(range(i+1)))
            executor.advance_many([(states[0], [9])], capture=False)
            self.assertEqual(result[0].features, [(0,)])
            self.assertEqual(states[0].history, [0, 9])
            self.assertTrue(executor.capabilities['prefill_gpu_batch'])
            self.assertEqual(p.allocations, 1)

    def test_all_inputs_validate_before_any_request_changes(self):
        p = BatchProvider()
        with BatchPrefillExecutor(p, chunk_rows=8) as executor:
            a, b = state(), state()
            for windows in ([], [(a, [1])]*2, [(state(), [1]) for _ in range(5)],
                    [(a, [1]), (b, [-1])], [(a, [1]), (b, [2]*9)]):
                with self.subTest(windows=windows), self.assertRaises(ValueError):
                    executor.advance_many(windows)
                self.assertEqual((a.pos, b.pos, p.allocations), (0, 0, 0))
                self.assertFalse(executor.failed)
            b.limit = 0
            with self.assertRaises(ValueError): executor.advance_many([(a, [1]), (b, [2])])
            self.assertEqual(p.calls, [])
            with self.assertRaises(ValueError): executor.advance_many([(a, [1])], capture=1)

    def test_failure_quarantines_whole_workspace_and_retains_until_fence(self):
        p = BatchProvider()
        executor = BatchPrefillExecutor(p, chunk_rows=8)
        a, b = state(), state()
        p.fail = True
        with self.assertRaisesRegex(RuntimeError, 'partial'):
            executor.advance_many([(a, [1]), (b, [2])])
        self.assertEqual((a.pos, b.pos), (1, 0))
        self.assertTrue(executor.failed)
        with self.assertRaisesRegex(RuntimeError, 'quarantined'):
            executor.advance_many([(b, [3])])
        p.bad_fence = True
        with self.assertRaisesRegex(RuntimeError, 'fence'): executor.close()
        self.assertIsNotNone(executor.workspace)
        p.bad_fence = False
        executor.close()

    def test_stream_lock_and_provider_contract(self):
        p = BatchProvider()
        with BatchPrefillExecutor(p, chunk_rows=8) as executor:
            p.stream = 2
            with self.assertRaisesRegex(RuntimeError, 'stream'): executor.advance_many([(state(), [1])])
            p.stream = 1
            executor._lock.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, 'already'): executor.advance_many([(state(), [1])])
            finally: executor._lock.release()
            self.assertEqual(p.allocations, 0)
            p.bad_position = True
            with self.assertRaisesRegex(RuntimeError, 'commit'): executor.advance_many([(state(), [1])])
            self.assertTrue(executor.failed)

    def test_single_request_entry_point_retains_multichunk_behavior(self):
        p = BatchProvider()
        with BatchPrefillExecutor(p, chunk_rows=8) as executor:
            value = executor.advance(state(), range(19))
            self.assertEqual(value.chunks, (8, 8, 3))
            self.assertEqual(value.features[-1], tuple(range(19)))
            self.assertEqual(p.allocated_rows, 32)

    def test_layout_matches_pinned_gdn_and_aligned_shared_storage(self):
        cfg = NS(hidden=2560, streams=4, low=320, heads=24, kv_heads=2, head_dim=256,
            index_heads=4, index_dim=128, index_budget=2048, index_ratio=4,
            conv_dim=10240, nv=48, ple_dim=2560, heads_per_ngram=8,
            top_k=10, experts=512, moe_width=640)
        layout = workspace_layout(cfg, 4096, 131088)
        self.assertEqual(layout['views']['proj']['shape'], (1, 4096, 16480))
        self.assertEqual(layout['arena_bytes'], 4096*11*2560*4)
        for value in layout['views'].values():
            self.assertEqual(value['offset'] % 256, 0)
            self.assertLessEqual(value['offset']+value['bytes'], layout['arena_bytes'])
        for name in ('normed', 'proj', 'pa', 'ple_emb', 'moe.y'):
            self.assertEqual(layout['views'][name]['offset'], 0)
        for rows in (0, True, 4097):
            with self.assertRaises(ValueError): workspace_layout(cfg, rows, 131088)


class PromptPhaseTests(unittest.TestCase):
    def control(self):
        return PromptBatchControl(NS(mode=lambda s: s.mode,
            set_prompt_mode=lambda s, enabled: setattr(s, 'mode', int(enabled))))

    def test_normal_mode_restored_on_success_and_native_failure(self):
        native = NS(mode=0)
        control = self.control()
        native.schedule = lambda pool: native.mode
        self.assertEqual(control.schedule(native, None), 1)
        self.assertEqual(native.mode, 0)
        def fail(pool): raise RuntimeError('native failed')
        native.schedule = fail
        with self.assertRaisesRegex(RuntimeError, 'native failed'): control.schedule(native, None)
        self.assertEqual(native.mode, 0)
        native.mode = 2
        with self.assertRaisesRegex(RuntimeError, 'NORMAL'): control.schedule(native, None)

    def test_ready_decode_keeps_native_opportunity(self):
        b = object.__new__(SchedulerBridge)
        b.owner, b.serial, b.pool = 'test', 0, None
        b.closed = b.poisoned = False
        b.pending = None
        b.prefill_interleave = False
        b.interleave_prefill = b._last_prompt = True
        b.prompt_control = self.control()
        b._retire = lambda: None
        b.core = NS(SequenceStatus=NS(WAITING=0, RUNNING=1, SWAPPED=2))
        b.requests = dict(long=dict(cancelled=False, seq=NS(status=0, num_cached_tokens=0,
            data=NS(get_len=lambda: 20)), prompt_length=20))
        seen = []
        empty = NS(size=0)
        output = NS(prompt_run=True, scheduled_seqs=[], clone_states=empty, clone_blocks=empty, zero_states=empty)
        def schedule(pool):
            seen.append((b.native.mode, b.native.decode_step))
            return [], output
        b.native = NS(mode=0, decode_step=0, schedule=schedule)
        b.prepare()
        self.assertEqual(seen, [(1, 0)])
        self.assertEqual(b.native.mode, 0)
        b.pending = None
        b.requests['short'] = dict(cancelled=False, prompt_length=10,
            seq=NS(status=1, num_cached_tokens=10, data=NS(get_len=lambda: 11)))
        b.prepare()
        self.assertEqual(seen[-1], (0, 1))


class BatchDraftFeatureTests(unittest.TestCase):
    def test_prepared_cancellation_is_excluded_from_shared_forward(self):
        bridge = object.__new__(SpeculativeSchedulerBridge)
        bridge.prefill_batch_rows, bridge.depth, bridge.vocab_size = 1024, 7, 100
        bridge.poisoned = False
        bridge._plan = lambda plan: None
        rows = tuple(NS(handle=i, token_ids=(1,), position=0, prompt=True) for i in range(4))
        entries = {i: dict(cancelled=i == 1, state=NS(pos=0),
            seq=NS(sampling_params=NS(max_tokens=3))) for i in range(4)}
        bridge._entry = entries.__getitem__
        calls = []
        def advance(requests):
            calls.append([row.handle for row, _, _ in requests])
            for _, s, _ in requests:
                s.pos += 1
            return tuple(VerifiedStep((), 1, (0,)*7, (0.,)*7, 0., 1., 0.) for _ in requests)
        bridge.backend = NS(advance_prefill_many=advance, position=lambda s: s.pos)
        plan = NS(rows=rows, executed=False)
        bridge.execute(plan)
        self.assertEqual(calls, [[0, 2, 3]])
        self.assertEqual((plan.prefill_batch_size, plan.prefill_batch_rows), (3, 3))
        self.assertIsNone(plan.results[1])
        self.assertTrue(plan.executed)
        self.assertEqual(entries[1]['state'].pos, 0)

    def test_deferred_features_outputs_and_additive_wall_accounting(self):
        class Logits:
            def __getitem__(self, key): return self
            def argmax(self): return 7
        calls = []
        def advance_many(windows, capture):
            calls.append(windows)
            result = []
            for s, ids in windows:
                features = list(range(s.pos, s.pos+len(ids)))
                s.pos += len(ids)
                result.append(NS(logits=Logits(), features=features))
            return result
        target = NS(buffers={i: None for i in range(1, 9)}, state=state,
            torch=NS(inference_mode=nullcontext, cat=lambda xs: sum(xs, []),
                cuda=NS(synchronize=lambda: None)))
        backend = SpeculativeBackend(target, NS(deferred_draft=True), eos_ids={99}, depth=7,
            prior=[.7]*7, prefill_executor=NS(advance_many=advance_many))
        handles = [RequestHandle('test', i, 1, str(i)) for i in range(2)]
        states = [backend.create(h) for h in handles]
        rows = [WorkRow(handles[i], 0, tuple(range(i+1)), True, i == 1) for i in range(2)]
        result = backend.advance_prefill_many([(row, s, 3) for row, s in zip(rows, states)])
        self.assertEqual([r.outputs for r in result], [(), (7,)])
        self.assertEqual([s.output_tokens for s in states], [0, 1])
        self.assertEqual([s.context for s in states], [[0], [0, 1]])
        self.assertEqual(result[1].verify_ms, 2*result[0].verify_ms)
        self.assertEqual(result[1].next_drafts, (0,)*7)
        self.assertEqual(len(calls), 1)
        invalid = WorkRow(handles[1], 0, (3,), True, False)
        with self.assertRaisesRegex(BridgeError, 'ownership/position'):
            backend.advance_prefill_many([(WorkRow(handles[0], 1, (2,), True, False), states[0], 3),
                (invalid, states[1], 3)])
        self.assertEqual(len(calls), 1)
        self.assertEqual([s.target.pos for s in states], [1, 2])


if __name__ == '__main__': unittest.main()
