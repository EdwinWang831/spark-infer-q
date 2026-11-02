"""Concurrent loopback SSE service over the asynchronous GPU owner worker."""
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import logging
from queue import Empty
import time
from uuid import uuid4

from .kaifra_worker import WorkerBusy,WorkerFailed
from .deepinfer_bridge import BridgeError


def text_delta(previous, current, *, final=False):
    # A byte-level tokenizer can revise an unfinished Unicode suffix. Delay that
    # suffix instead of emitting replacement characters that clients cannot undo.
    stable=current if final else current.rstrip('\ufffd')
    if not stable.startswith(previous):
        raise BridgeError('tokenizer revised text that has already been streamed')
    return stable[len(previous):],stable


def make_server(worker,address=('127.0.0.1',18084),*,protocol='kaifra',
                model_id='spark-infer-q',admission_timeout=30.):
    if address[0] not in ('127.0.0.1','::1','localhost'):raise ValueError('K4 listener is loopback-only')
    if protocol not in ('kaifra','openai'):raise ValueError('unsupported wire protocol')

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup();self.connection.settimeout(30)
        def log_message(self,*args):pass
        def reply(self,code,value):
            if protocol=='openai' and isinstance(value.get('error'),str):
                value=dict(error=dict(message=value['error'],type='invalid_request_error' if code<500
                    else 'server_error',param=None,code=str(code)))
            payload=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def body(self):
            size=int(self.headers.get('Content-Length','0'))
            if not 1<=size<=2*1024**2:raise ValueError('request body must contain 1..2097152 bytes')
            value=json.loads(self.rfile.read(size))
            if not isinstance(value,dict):raise ValueError('request body must be an object')
            return value
        def send(self,value):
            self.wfile.write(('data: '+json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n\n').encode());self.wfile.flush()
        def do_GET(self):
            if self.path=='/v1/models':
                self.reply(200,dict(object='list',data=[dict(id=model_id,object='model',
                    created=0,owned_by='spark-infer-q')]));return
            state=worker.snapshot()
            if self.path in ('/health','/healthz','/ready'):
                self.reply(200 if state['status']=='ready' else 503,state);return
            if self.path=='/metrics':
                values=dict(up=int(state['status']=='ready'),active_requests=state['active_requests'],
                            open_streams=state['open_streams'],pending_submissions=state['pending_submissions'])
                values.update(state.get('metrics',{}))
                payload=''.join(f'kaifra_{key} {value}\n' for key,value in values.items()).encode()
                self.send_response(200);self.send_header('Content-Type','text/plain; version=0.0.4; charset=utf-8')
                self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload);return
            self.reply(404,dict(error='unknown path'))
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
                allowed={'messages','max_tokens','temperature','stream','model'}
                if protocol=='openai':allowed.add('stream_options')
                if set(body)-allowed:raise ValueError('unsupported request fields')
                streaming=body.get('stream',protocol=='kaifra')
                if type(streaming) is not bool:raise ValueError('stream must be boolean')
                if protocol=='kaifra' and not streaming:raise ValueError('K4 requires stream=true')
                if protocol=='openai' and body.get('model',model_id)!=model_id:raise ValueError('unknown model')
                options=body.get('stream_options')
                if options is not None and (not streaming or not isinstance(options,dict)
                    or set(options)-{'include_usage'} or type(options.get('include_usage',False)) is not bool):
                    raise ValueError('only stream_options.include_usage is supported')
                include_usage=bool(options and options.get('include_usage'))
                temperature=body.get('temperature',0)
                if type(temperature) not in (int,float) or temperature!=0:raise ValueError('only temperature=0 is supported')
                sub=worker.submit(uuid4().hex,body.get('messages'),max_tokens=body.get('max_tokens',128),
                    temperature=temperature,timeout=admission_timeout)
                request_id='chatcmpl-'+sub.handle.request_id;created=int(time.time())
                if not streaming:
                    while True:
                        try:event=sub.receive(timeout=1.)
                        except Empty:continue
                        if event.finish_reason is not None:break
                    usage=dict(prompt_tokens=event.prompt_tokens,completion_tokens=event.completion_tokens,
                               total_tokens=event.prompt_tokens+event.completion_tokens)
                    self.reply(200,dict(id=request_id,object='chat.completion',created=created,model=model_id,
                        choices=[dict(index=0,message=dict(role='assistant',content=event.text or ''),
                            finish_reason='stop' if event.finish_reason=='cancelled' else event.finish_reason)],
                        usage=usage,kaifra_finish_reason=event.finish_reason));return
                self.send_response(200);self.send_header('Content-Type','text/event-stream; charset=utf-8')
                self.send_header('Cache-Control','no-cache');self.send_header('X-Request-Id',sub.handle.request_id);self.end_headers();started=True
                def chunk(delta,reason=None):
                    self.send(dict(id=request_id,object='chat.completion.chunk',created=created,model=model_id,
                        choices=[dict(index=0,delta=delta,finish_reason=reason)]))
                if protocol=='kaifra':self.send(dict(event='accepted',request_id=sub.handle.request_id))
                else:chunk(dict(role='assistant'))
                published=''
                while True:
                    try:event=sub.receive(timeout=1.)
                    except Empty:
                        self.wfile.write(b': heartbeat\n\n');self.wfile.flush();continue
                    if protocol=='kaifra':
                        item=asdict(event);item.pop('handle')
                        self.send(dict(event='output',request_id=sub.handle.request_id,**item))
                    else:
                        delta,published=text_delta(published,event.text or '',final=event.finish_reason is not None)
                        if delta:chunk(dict(content=delta))
                        if event.finish_reason is not None:
                            chunk({},'stop' if event.finish_reason=='cancelled' else event.finish_reason)
                            if include_usage:
                                self.send(dict(id=request_id,object='chat.completion.chunk',created=created,
                                    model=model_id,choices=[],usage=dict(prompt_tokens=event.prompt_tokens,
                                        completion_tokens=event.completion_tokens,
                                        total_tokens=event.prompt_tokens+event.completion_tokens)))
                    if event.finish_reason is not None:break
                self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
            except (ValueError,UnicodeError,BridgeError) as exc:
                busy_code=429 if protocol=='openai' else 409
                if not started:self.reply(400 if isinstance(exc,(ValueError,UnicodeError)) else busy_code if isinstance(exc,WorkerBusy) else 503,dict(error=str(exc)))
                else:
                    try:self.send(dict(event='error',error=str(exc)))
                    except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
            except Exception:
                logging.exception('inference request failed')
                if not started:self.reply(500,dict(error='inference request failed'))
                else:
                    try:self.send(dict(error=dict(message='inference request failed',type='server_error')))
                    except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
            finally:
                if sub is not None:sub.close()

    class Server(ThreadingHTTPServer):
        daemon_threads=False
        allow_reuse_address=True
    return Server(address,Handler)
