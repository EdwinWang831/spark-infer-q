"""Real HTTP/SSE lifecycle over the deterministic event-fenced worker fixture."""
import http.client
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from types import SimpleNamespace
import socket
import time
import unittest

from spark_infer_q.kaifra_runtime import KaifraEngine
from spark_infer_q.kaifra_worker import AsyncWorker
from spark_infer_q.kaifra_service import make_server
from test_kaifra_worker import DeferredBridge


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.bridge=DeferredBridge()
        tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**k:[1],decode=lambda ids,**k:'x'*len(ids))
        self.worker=AsyncWorker(lambda:KaifraEngine(self.bridge,tokenizer=tokenizer)).start()
        self.server=make_server(self.worker,('127.0.0.1',0));self.thread=Thread(target=self.server.serve_forever);self.thread.start()
        self.address=self.server.server_address;self.connections=[]
    def tearDown(self):
        for conn in self.connections:conn.close()
        self.bridge.gate.set();self.worker.close(3);self.server.shutdown();self.server.server_close();self.thread.join(3)
    def request(self,path,body=None):
        conn=http.client.HTTPConnection(*self.address,timeout=4);self.connections.append(conn)
        conn.request('GET' if body is None else 'POST',path,body=None if body is None else json.dumps(body),headers={'Content-Type':'application/json'})
        return conn,conn.getresponse()
    def stream(self,n=4):
        return self.request('/v1/chat/completions',dict(messages=[dict(role='user',content='hello')],max_tokens=n))
    def wait_for(self,predicate):
        end=time.monotonic()+3
        while not predicate():
            if time.monotonic()>end:self.fail('HTTP lifecycle timeout')
            time.sleep(.005)
    def read_events(self,response):
        events=[]
        for line in response:
            if not line.startswith(b'data: '):continue
            value=line[6:].strip()
            if value==b'[DONE]':return events
            events.append(json.loads(value))
        return events
    def test_concurrent_streams_health_and_cancel_are_independent(self):
        _,first=self.stream(8);first_id=first.getheader('X-Request-Id')
        self.assertTrue(self.bridge.started.wait(2))
        with ThreadPoolExecutor(1) as pool:
            pending=pool.submit(self.stream,3)
            self.wait_for(lambda:self.worker.snapshot()['pending_submissions']==1)
            _,health=self.request('/health');self.assertEqual(health.status,200)
            self.assertEqual(json.loads(health.read())['status'],'ready')
            _,cancel=self.request('/cancel',{'request_id':first_id});self.assertEqual(cancel.status,200);cancel.read()
            self.assertEqual(self.bridge.released,[])
            self.bridge.gate.set();_,second=pending.result(3)
        one=self.read_events(first);two=self.read_events(second)
        self.assertEqual(one[-1]['finish_reason'],'cancelled')
        self.assertFalse(any(e.get('token_ids') for e in one))
        self.assertEqual([t for e in two for t in e.get('token_ids',[])],[3]*3)
        self.assertTrue(all(e['request_id']==first_id for e in one))
        self.assertTrue(all(e['request_id']==second.getheader('X-Request-Id') for e in two))
        self.worker.wait_idle();self.assertEqual(self.worker.snapshot()['open_streams'],0)
    def test_disconnect_registers_cancellation_before_device_reuse(self):
        conn,response=self.stream(8)
        self.assertTrue(self.bridge.started.wait(2));self.assertIn(b'accepted',response.readline())
        response.fp.raw._sock.shutdown(socket.SHUT_RDWR);response.close();conn.close()
        self.wait_for(lambda:self.worker.snapshot()['open_streams']==0)
        self.assertEqual(self.bridge.released,[])
        self.bridge.gate.set();self.worker.wait_idle();self.assertEqual(len(self.bridge.released),1)
    def test_invalid_http_input_does_not_poison_worker(self):
        for body in ({'messages':[]},{'messages':[{'role':'user','content':'x'}],'stream':False},{'unknown':1}):
            _,response=self.request('/v1/chat/completions',body);self.assertEqual(response.status,400);response.read()
        _,response=self.request('/cancel',{'request_id':'missing'});self.assertEqual(response.status,404);response.read()
        self.assertEqual(self.worker.snapshot()['status'],'ready')
        with self.assertRaises(ValueError):make_server(self.worker,('0.0.0.0',0))


if __name__=='__main__':unittest.main()
