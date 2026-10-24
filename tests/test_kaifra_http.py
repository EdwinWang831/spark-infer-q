"""Loopback HTTP checks with a CPU fake backend; no GPU/service speed claim."""
from contextlib import contextmanager
import json
import socket
from threading import Event, Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from spark_infer_q.kaifra_http import make_server
from spark_infer_q.kaifra_runtime import KaifraEngine
from test_kaifra_runtime import ScriptedBridge


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs): return [1, 2]
    def decode(self, tokens, **kwargs): return '中'*len(tokens)


@contextmanager
def server():
    engine = KaifraEngine(ScriptedBridge(), tokenizer=Tokenizer())
    http = make_server(engine, ('127.0.0.1', 0))
    thread = Thread(target=http.serve_forever); thread.start()
    try:
        yield engine, 'http://127.0.0.1:'+str(http.server_port)
    finally:
        http.shutdown(); http.server_close(); thread.join(timeout=5); engine.close()


def post(base, path='/v1/chat/completions', value=None):
    if value is None:
        value = dict(messages=[dict(role='user', content='你好')], max_tokens=2, stream=True)
    return urlopen(Request(base+path, json.dumps(value).encode(),
        {'Content-Type': 'application/json'}), timeout=5)


class HTTPTests(unittest.TestCase):
    def test_sse_has_tokens_terminal_metrics_and_releases(self):
        with server() as (engine, base):
            with post(base) as response:
                body = response.read().decode()
                request_id = response.headers['X-Request-Id']
            items = [json.loads(line[6:]) for line in body.splitlines()
                     if line.startswith('data: {')]
            self.assertEqual(items[0], dict(event='accepted', request_id=request_id))
            self.assertEqual(items[-1]['finish_reason'], 'length')
            self.assertEqual(items[-1]['text'], '中中')
            self.assertEqual(items[-1]['processed_position'], 3)
            self.assertTrue(body.endswith('data: [DONE]\n\n'))
            self.assertEqual(engine.active_requests, 0)
            with urlopen(base+'/health') as response:
                self.assertEqual(json.load(response)['status'], 'ready')

    def test_explicit_cancel_and_busy_rejection(self):
        with server() as (engine, base):
            original, gate, entered = engine.step, Event(), Event()
            def step():
                entered.set()
                if not gate.wait(5): raise RuntimeError('test gate timeout')
                return original()
            engine.step = step
            with post(base) as response:
                request_id = response.headers['X-Request-Id']
                try:
                    self.assertTrue(entered.wait(3))
                    with self.assertRaises(HTTPError) as error: post(base)
                    self.assertEqual(error.exception.code, 409)
                    error.exception.close()
                    with post(base, '/cancel', dict(request_id=request_id)) as cancelled:
                        self.assertTrue(json.load(cancelled)['cancelled'])
                finally:
                    gate.set()
                body = response.read().decode()
            self.assertIn('"finish_reason": "cancelled"', body)
            self.assertNotIn('"token_ids": [3]', body)
            self.assertEqual(engine.active_requests, 0)

    def test_rejects_bad_inputs_and_exposes_quarantine_health(self):
        with server() as (engine, base):
            for value in ([], dict(messages=[], stream=False), dict(messages=[], logprobs=True)):
                with self.assertRaises(HTTPError) as error: post(base, value=value)
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
            engine.bridge.poisoned = True
            with self.assertRaises(HTTPError) as error: urlopen(base+'/health')
            self.assertEqual(error.exception.code, 503)
            error.exception.close()

    def test_disconnect_cancels_and_releases_generation(self):
        with server() as (engine, base):
            original, gate = engine.step, Event()
            cancelled, released = Event(), Event()
            original_cancel = engine.cancel
            original_release = engine.bridge.release
            def step():
                if not gate.wait(5): raise RuntimeError('test gate timeout')
                return original()
            def cancel(handle):
                original_cancel(handle); cancelled.set()
            def release(handle, state):
                original_release(handle, state); released.set()
            engine.step, engine.cancel = step, cancel
            engine.bridge.release = release
            with post(base, value=dict(messages=[dict(role='user', content='hello')],
                                       max_tokens=20)) as response:
                self.assertIn(b'accepted', response.readline())
                response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            gate.set()
            self.assertTrue(cancelled.wait(3))
            self.assertTrue(released.wait(3))

    def test_refuses_public_listener(self):
        with self.assertRaises(ValueError): make_server(None, ('0.0.0.0', 18083))


if __name__ == '__main__': unittest.main()
