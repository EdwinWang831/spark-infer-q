"""Real HTTP requests over the real owner loop, using deterministic device fences."""
import http.client
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from types import SimpleNamespace
import unittest

from spark_infer_q.deepinfer_bridge import BridgeError
from spark_infer_q.kaifra_runtime import KaifraEngine
from spark_infer_q.kaifra_service import make_server, text_delta
from spark_infer_q.kaifra_worker import AsyncWorker
from test_kaifra_worker import DeferredBridge


class EngineApiTests(unittest.TestCase):
    def setUp(self):
        self.bridge = DeferredBridge()
        self.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [1],
                                         decode=lambda ids, **k: '字'*len(ids))
        self.worker = AsyncWorker(lambda: KaifraEngine(self.bridge, tokenizer=self.tokenizer)).start()
        self.server = make_server(self.worker, ('127.0.0.1', 0), protocol='openai', model_id='local-test')
        self.thread = Thread(target=self.server.serve_forever)
        self.thread.start()
        self.connections = []

    def tearDown(self):
        self.bridge.gate.set()
        for connection in self.connections:
            connection.close()
        try:
            self.worker.close(3)
        finally:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(3)

    def request(self, path, body=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=4)
        self.connections.append(connection)
        connection.request('GET' if body is None else 'POST', path,
            body=None if body is None else json.dumps(body), headers={'Content-Type': 'application/json'})
        return connection, connection.getresponse()

    def generate(self, **options):
        body = dict(model='local-test', messages=[dict(role='user', content='你好')], max_tokens=3)
        body.update(options)
        return self.request('/v1/chat/completions', body)

    def test_models_health_nonstream_output_usage_and_metrics(self):
        for path in ('/health', '/ready', '/healthz'):
            _, response = self.request(path)
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())['status'], 'ready')
        _, response = self.request('/v1/models')
        self.assertEqual(json.loads(response.read())['data'][0]['id'], 'local-test')
        self.bridge.gate.set()
        _, response = self.generate()
        self.assertEqual(response.status, 200)
        result = json.loads(response.read())
        self.assertEqual(result['object'], 'chat.completion')
        self.assertEqual(result['choices'][0]['message']['content'], '字字字')
        self.assertEqual(result['choices'][0]['finish_reason'], 'length')
        self.assertEqual(result['usage'], dict(prompt_tokens=1, completion_tokens=3, total_tokens=4))
        self.worker.wait_idle()
        _, response = self.request('/metrics')
        metrics = response.read().decode()
        self.assertIn('kaifra_requests_completed_total 1\n', metrics)
        self.assertIn('kaifra_output_tokens_total 3\n', metrics)
        self.assertIn('kaifra_ttft_seconds_count 1\n', metrics)
        self.assertIn('kaifra_active_requests 0\n', metrics)

    def test_stream_chunks_reconstruct_text_and_include_final_usage(self):
        self.bridge.gate.set()
        _, response = self.generate(stream=True, stream_options=dict(include_usage=True))
        chunks = []
        done = False
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                done = True
                break
            chunks.append(json.loads(data))
        self.assertTrue(done)
        self.assertTrue(all(item['object'] == 'chat.completion.chunk' for item in chunks))
        content = ''.join(choice['delta'].get('content', '') for item in chunks for choice in item['choices'])
        self.assertEqual(content, '字字字')
        self.assertEqual(chunks[0]['choices'][0]['delta']['role'], 'assistant')
        self.assertEqual(chunks[-2]['choices'][0]['finish_reason'], 'length')
        self.assertEqual(chunks[-1]['usage']['completion_tokens'], 3)

    def test_invalid_requests_do_not_reach_the_engine(self):
        for options in (dict(model='other'), dict(temperature=.2), dict(temperature=False),
                        dict(stream='true'), dict(stream_options=dict(include_usage=True)),
                        dict(stream=True, stream_options=dict(include_usage=1)), dict(unknown=1)):
            with self.subTest(options=options):
                _, response = self.generate(**options)
                self.assertEqual(response.status, 400)
                self.assertIn('message', json.loads(response.read())['error'])
        self.assertEqual(self.worker.snapshot()['metrics']['requests_admitted_total'], 0)

    def test_cancelled_stream_updates_metrics_and_keeps_health_ready(self):
        _, response = self.generate(stream=True)
        request_id = response.getheader('X-Request-Id')
        self.assertTrue(self.bridge.started.wait(2))
        _, cancellation = self.request('/cancel', dict(request_id=request_id))
        self.assertEqual(cancellation.status, 200)
        cancellation.read()
        self.bridge.gate.set()
        for line in response:
            if line.strip() == b'data: [DONE]':
                break
        self.worker.wait_idle()
        self.assertEqual(self.worker.snapshot()['metrics']['requests_cancelled_total'], 1)
        self.assertEqual(self.worker.snapshot()['metrics']['output_tokens_total'], 0)
        self.assertEqual(self.worker.snapshot()['status'], 'ready')

    def test_unicode_tail_is_held_until_the_character_is_complete(self):
        delta, previous = text_delta('', '\ufffd')
        self.assertEqual((delta, previous), ('', ''))
        delta, previous = text_delta(previous, '你')
        self.assertEqual((delta, previous), ('你', '你'))
        self.assertEqual(text_delta(previous, '你\ufffd'), ('', '你'))
        self.assertEqual(text_delta(previous, '你好', final=True), ('好', '你好'))
        with self.assertRaises(BridgeError):
            text_delta(previous, '再见')

    def test_full_admission_returns_429_while_existing_streams_survive(self):
        _, first = self.generate(stream=True)
        self.assertTrue(self.bridge.started.wait(2))
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(self.generate, stream=True)
            deadline = time.monotonic()+3
            while self.worker.snapshot()['pending_submissions'] != 1:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.001)
            _, rejected = self.generate(stream=True)
            self.assertEqual(rejected.status, 429)
            self.assertIn('error', json.loads(rejected.read()))
            self.bridge.gate.set()
            _, second = pending.result(3)
        for response in (first, second):
            for line in response:
                if line.strip() == b'data: [DONE]':
                    break
        self.worker.wait_idle()
        self.assertEqual(self.worker.snapshot()['metrics']['requests_completed_total'], 2)
        self.assertEqual(self.worker.snapshot()['metrics']['admission_rejected_total'], 1)


if __name__ == '__main__':
    unittest.main()
