"""Concurrent loopback SSE service over the asynchronous GPU owner worker."""
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from queue import Empty
from uuid import uuid4

from .kaifra_worker import WorkerBusy,WorkerFailed
from .deepinfer_bridge import BridgeError


def make_server(worker,address=('127.0.0.1',18084)):
    if address[0] not in ('127.0.0.1','::1','localhost'):raise ValueError('K4 listener is loopback-only')

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup();self.connection.settimeout(30)
        def log_message(self,*args):pass
        def reply(self,code,value):
            payload=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def body(self):
            size=int(self.headers.get('Content-Length','0'))
            if not 1<=size<=65536:raise ValueError('request body must contain 1..65536 bytes')
            value=json.loads(self.rfile.read(size))
            if not isinstance(value,dict):raise ValueError('request body must be an object')
            return value
        def send(self,value):
            self.wfile.write(('data: '+json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n\n').encode());self.wfile.flush()
        def do_GET(self):
            if self.path!='/health':self.reply(404,dict(error='unknown path'));return
            state=worker.snapshot();self.reply(200 if state['status']=='ready' else 503,state)
        def do_POST(self):
            if self.path not in ('/v1/chat/completions','/cancel'):
                self.reply(404,dict(error='unknown path'));return
            sub=None;started=False
            try:
                body=self.body()
                if self.path=='/cancel':
                    if set(body)!={'request_id'} or not isinstance(body['request_id'],str):raise ValueError('request_id required')
                    cancelled=worker.cancel_id(body['request_id'])
                    self.reply(200 if cancelled else 404,dict(cancelled=cancelled));return
                if set(body)-{'messages','max_tokens','temperature','stream','model'}:raise ValueError('unsupported request fields')
                if body.get('stream',True) is not True:raise ValueError('K4 requires stream=true')
                sub=worker.submit(uuid4().hex,body.get('messages'),max_tokens=body.get('max_tokens',128),temperature=body.get('temperature',0))
                self.send_response(200);self.send_header('Content-Type','text/event-stream; charset=utf-8')
                self.send_header('Cache-Control','no-cache');self.send_header('X-Request-Id',sub.handle.request_id);self.end_headers();started=True
                self.send(dict(event='accepted',request_id=sub.handle.request_id))
                while True:
                    try:event=sub.receive(timeout=1.)
                    except Empty:
                        self.wfile.write(b': heartbeat\n\n');self.wfile.flush();continue
                    item=asdict(event);item.pop('handle')
                    self.send(dict(event='output',request_id=sub.handle.request_id,**item))
                    if event.finish_reason is not None:break
                self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
            except (ValueError,UnicodeError,BridgeError) as exc:
                if not started:self.reply(400 if isinstance(exc,(ValueError,UnicodeError)) else 409 if isinstance(exc,WorkerBusy) else 503,dict(error=str(exc)))
                else:
                    try:self.send(dict(event='error',error=str(exc)))
                    except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
            finally:
                if sub is not None:sub.close()

    class Server(ThreadingHTTPServer):
        daemon_threads=False
        allow_reuse_address=True
    return Server(address,Handler)
