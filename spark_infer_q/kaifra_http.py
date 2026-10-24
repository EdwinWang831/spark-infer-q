"""Small loopback SSE boundary for K1; one active HTTP generation at a time."""
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Lock
from uuid import uuid4

from .deepinfer_bridge import BridgeError


def make_server(engine, address=('127.0.0.1', 18083)):
    if address[0] not in ('127.0.0.1', '::1', 'localhost'):
        raise ValueError('K1 HTTP listener is loopback-only')
    active_lock = Lock()
    active = {}

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *args):
            pass

        def reply(self, code, value):
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def body(self):
            size = int(self.headers.get('Content-Length', '0'))
            if not 1 <= size <= 65536:
                raise ValueError('request body must contain 1..65536 bytes')
            value = json.loads(self.rfile.read(size))
            if not isinstance(value, dict):
                raise ValueError('request body must be an object')
            return value

        def do_GET(self):
            if self.path != '/health':
                self.reply(404, dict(error='unknown path')); return
            healthy = not engine.closed and not engine.bridge.poisoned
            self.reply(200 if healthy else 503,
                dict(status='ready' if healthy else 'unavailable', capabilities=engine.capabilities))

        def do_POST(self):
            if self.path not in ('/v1/chat/completions', '/cancel'):
                self.reply(404, dict(error='unknown path')); return
            try:
                body = self.body()
            except (ValueError, UnicodeError):
                self.reply(400, dict(error='invalid JSON request')); return
            if self.path == '/cancel':
                with active_lock:
                    handle = active.get(body.get('request_id')) if isinstance(body.get('request_id'), str) else None
                    if handle is None:
                        self.reply(404, dict(error='request is not active')); return
                    try:
                        engine.cancel(handle)
                    except BridgeError:
                        self.reply(409, dict(error='request already finished')); return
                self.reply(200, dict(cancelled=True)); return
            handle, started = None, False
            try:
                with active_lock:
                    if active:
                        self.reply(409, dict(error='K1 HTTP generation slot is busy')); return
                    if set(body)-{'messages', 'max_tokens', 'temperature', 'stream', 'model'}:
                        raise ValueError('unsupported request fields')
                    if body.get('stream', True) is not True:
                        raise ValueError('K1 requires stream=true')
                    handle = engine.submit(uuid4().hex, body.get('messages'),
                        max_tokens=body.get('max_tokens', 128), temperature=body.get('temperature', 0))
                    active[handle.request_id] = handle
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Request-Id', handle.request_id)
                self.end_headers()
                started = True
                def send(value):
                    payload = json.dumps(value, ensure_ascii=False, allow_nan=False)
                    self.wfile.write(('data: '+payload+'\n\n').encode())
                    self.wfile.flush()
                # Snapshot text plus immutable token IDs avoids corrupting UTF-8
                # when a decoder revises its last incomplete character.
                send(dict(request_id=handle.request_id, event='accepted'))
                for event in engine.drain():
                    if event.handle != handle:
                        raise BridgeError('HTTP received another request output')
                    item = asdict(event)
                    item.pop('handle')
                    send(dict(request_id=handle.request_id, event='output', **item))
                self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            except (ValueError, BridgeError) as exc:
                if not started:
                    self.reply(400 if isinstance(exc, ValueError) else 503, dict(error=str(exc)))
                else:
                    send(dict(event='error', error=str(exc)))
            except Exception:
                if started:
                    send(dict(event='error', error='runtime failed; inspect local server log'))
                raise
            finally:
                if handle is not None:
                    try:
                        if engine.active_requests and not engine.bridge.poisoned:
                            engine.cancel(handle)
                            # At a synchronous step boundary no device work is
                            # outstanding; finish cancellation before freeing slot.
                            list(engine.drain())
                    finally:
                        with active_lock:
                            active.pop(handle.request_id, None)

    class Server(ThreadingHTTPServer):
        daemon_threads = False
        allow_reuse_address = True

    return Server(address, Handler)
