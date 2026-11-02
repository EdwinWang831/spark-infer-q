"""Runtime contract checks; fake scheduling here is not native/GPU evidence."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest

from spark_infer_q.deepinfer_bridge import BridgeError, RequestHandle, WorkRow
from spark_infer_q.kaifra_runtime import KaifraEngine, QwenTargetBackend


class FakeData:
    def __init__(self, ids): self.ids = list(ids)
    def get_token_ids(self): return list(self.ids)


class ScriptedBridge:
    """Only the bridge boundary is faked, to inject completion/failure timing."""
    def __init__(self):
        self.requests, self.completed, self.events = {}, {}, []
        self.capacity, self.eos_ids = 32, {9, 10}
        self.closed = self.poisoned = False
        self.generation = 0
        self.backend = SimpleNamespace(synchronize=lambda: None, release=self.release)
        self.released = []
        self.fail = False
        self.drained = False
    def _ready(self):
        if self.closed or self.poisoned: raise BridgeError('closed or quarantined')
    def add(self, name, ids, limit):
        self._ready()
        if not isinstance(name, str) or not name: raise ValueError('request ID')
        if not ids or len(ids)+limit > self.capacity or limit < 1: raise ValueError('capacity')
        if name in self.requests: raise BridgeError('duplicate')
        self.generation += 1
        h = RequestHandle('test', 0, self.generation, name)
        self.requests[name] = dict(handle=h, seq=SimpleNamespace(data=FakeData(ids),
            num_cached_tokens=0), state=object(), cancelled=False, prompt_length=len(ids), limit=limit)
        return h
    def cancel(self, handle):
        entry = self.requests.get(handle.request_id)
        if entry is None or entry['handle'] != handle: raise BridgeError('stale')
        entry['cancelled'] = True
    def step(self):
        self._ready()
        if self.fail: raise RuntimeError('injected backend failure')
        for name, e in list(self.requests.items()):
            seq, n = e['seq'], e['prompt_length']
            if not e['cancelled']:
                seq.num_cached_tokens = len(seq.data.ids)
                seq.data.ids.append(9 if name == 'eos' else 10 if name == 'other_eos' else 3)
            tokens = seq.data.ids[n:]
            self.events.append({'private': 'trace'})
            if e['cancelled'] or tokens[-1] in self.eos_ids or len(tokens) == e['limit']:
                self.completed[e['handle']] = dict(token_ids=list(tokens),
                    processed_position=seq.num_cached_tokens, cancelled=e['cancelled'])
                self.release(e['handle'], e['state'])
                del self.requests[name]
    def release(self, h, state): self.released.append(h)
    def drain(self): self.drained = True
    def close(self):
        for e in self.requests.values(): self.release(e['handle'], e['state'])
        self.requests.clear()
        self.closed = True


class RuntimeTests(unittest.TestCase):
    def engine(self, **kwargs): return KaifraEngine(ScriptedBridge(), **kwargs)

    def test_stream_prefix_pending_token_position_metrics_and_bounded_history(self):
        times = iter([1.0, 1.5, 2.0, 2.5])
        with self.engine(clock=lambda: next(times)) as e:
            h = e.submit_tokens('length', [1, 2], max_tokens=3)
            out = list(e.drain())
            self.assertEqual([x.token_ids for x in out], [(3,), (3,), (3,)])
            self.assertEqual([x.processed_position for x in out], [2, 3, 4])
            self.assertEqual([x.finish_reason for x in out], [None, None, 'length'])
            self.assertEqual(out[-1].completion_tokens, 3)
            self.assertEqual(out[-1].ttft_seconds, .5)
            self.assertEqual(out[-1].decode_seconds, 1.0)
            self.assertEqual(e.bridge.released, [h])
            self.assertEqual((e.active_requests, e.bridge.completed, e.bridge.events), (0, {}, []))
            self.assertTrue(e.bridge.drained)

    def test_each_eos_finishes_once(self):
        for name in ('eos', 'other_eos'):
            with self.engine() as e:
                e.submit_tokens(name, [1], max_tokens=5)
                out = list(e.drain())
                self.assertEqual(len(out), 1)
                self.assertEqual(out[0].finish_reason, 'stop')
                self.assertEqual(e.step(), [])

    def test_cancel_waiting_and_after_output_and_reuse(self):
        for emitted in (False, True):
            with self.engine() as e:
                old = e.submit_tokens('reuse', [1], max_tokens=5)
                if emitted: self.assertEqual(e.step()[0].token_ids, (3,))
                e.cancel(old)
                end = e.step()
                self.assertEqual(len(end), 1)
                self.assertEqual(end[0].finish_reason, 'cancelled')
                self.assertEqual(end[0].token_ids, ())
                new = e.submit_tokens('reuse', [1], max_tokens=1)
                self.assertGreater(new.generation, old.generation)
                with self.assertRaises(BridgeError): e.cancel(old)
                self.assertEqual(list(e.drain())[0].handle, new)

    def test_backend_error_quarantines_and_fenced_close_releases(self):
        e = self.engine()
        h = e.submit_tokens('failure', [1], max_tokens=2)
        e.bridge.fail = True
        with self.assertRaisesRegex(RuntimeError, 'injected'): e.step()
        with self.assertRaises(BridgeError): e.step()
        with self.assertRaises(BridgeError): e.submit_tokens('next', [1], max_tokens=2)
        e.close(); e.close()
        self.assertEqual(e.bridge.released, [h])
        self.assertEqual(e.active_requests, 0)

    def test_unsafe_fence_prevents_cleanup_and_reuse(self):
        e = self.engine()
        e.submit_tokens('failure', [1], max_tokens=2)
        e.bridge.poisoned = True
        def bad_fence(): raise RuntimeError('device fault')
        e.bridge.backend.synchronize = bad_fence
        with self.assertRaisesRegex(RuntimeError, 'device fault'): e.close()
        self.assertFalse(e.closed)
        self.assertEqual(e.bridge.released, [])

    def test_message_template_and_unicode_are_snapshots(self):
        seen = []
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                seen.append((messages, kwargs)); return [1, 2]
            def decode(self, tokens, **kwargs): return '\ufffd' if len(tokens)==1 else '中'
        with self.engine(tokenizer=Tokenizer()) as e:
            messages = [dict(role='system', content='简短'), dict(role='user', content='你好')]
            e.submit('zh', messages, max_tokens=2)
            self.assertEqual([x.text for x in e.drain()], ['\ufffd', '中'])
            self.assertEqual(seen[0][0], messages)
            self.assertFalse(seen[0][1]['enable_thinking'])
            self.assertFalse(seen[0][1]['return_dict'])
            with self.assertRaises(ValueError): e.submit('bad', messages, temperature=.5)
            with self.assertRaises(ValueError): e.submit('bad', [dict(role='user', content=[])])

    def test_step_limit_and_close_of_active_requests(self):
        e = self.engine()
        h = e.submit_tokens('a', [1], max_tokens=5)
        with self.assertRaisesRegex(BridgeError, 'step limit'): list(e.drain(max_steps=1))
        e.close()
        self.assertEqual(e.bridge.released, [h])
        with self.assertRaises(BridgeError): e.step()


class BackendTests(unittest.TestCase):
    def target(self):
        class Logit:
            def __getitem__(self, key): return self
            def argmax(self): return 7
        self.calls = []
        def step(state, ids, capture):
            self.calls.append((state.pos, ids, capture)); state.pos += len(ids)
            return Logit(), None
        return SimpleNamespace(buffers={n: object() for n in range(1,9)},
            state=lambda: SimpleNamespace(pos=0), step=step,
            torch=SimpleNamespace(inference_mode=nullcontext,
                                  cuda=SimpleNamespace(synchronize=lambda: None)))

    def test_prompt_chunks_and_ownership(self):
        be = QwenTargetBackend(self.target(), prefill_rows=8)
        h = RequestHandle('x', 0, 1, 'a'); state = be.create(h)
        row = WorkRow(h, 0, tuple(range(17)), True, True)
        self.assertEqual(be.advance(row, state), 7)
        self.assertEqual([len(c[1]) for c in self.calls], [8, 8, 1])
        self.assertEqual(state.pos, 17)
        with self.assertRaises(BridgeError): be.advance(row, state)
        be.release(h, state)
        with self.assertRaises(BridgeError): be.release(h, state)
        self.assertEqual(be.states, {})

    def test_unsupported_buffers_rejected(self):
        target = self.target(); target.buffers.pop(8)
        with self.assertRaises(ValueError): QwenTargetBackend(target)

    def test_owned_prefill_routes_prompt_only_and_preserves_observer_contract(self):
        target = self.target()
        calls = []
        class Logit:
            def __getitem__(self, key): return self
            def argmax(self): return 11
        def advance(state, ids, capture):
            calls.append((tuple(ids),capture)); state.pos += len(ids)
            return SimpleNamespace(logits=Logit())
        executor = SimpleNamespace(advance=advance)
        backend = QwenTargetBackend(target, prefill_executor=executor)
        h = RequestHandle('x',0,1,'a'); s = backend.create(h)
        self.assertEqual(backend.advance(WorkRow(h,0,tuple(range(17)),True,True),s),11)
        self.assertEqual(calls,[(tuple(range(17)),False)])
        self.assertEqual(self.calls,[])
        self.assertEqual(backend.advance(WorkRow(h,17,(2,),False,False),s),7)
        self.assertEqual(len(self.calls),1)
        backend.release(h,s)
        with self.assertRaisesRegex(ValueError,'observers'):
            QwenTargetBackend(target,observer=object(),prefill_executor=executor)


if __name__ == '__main__': unittest.main()
