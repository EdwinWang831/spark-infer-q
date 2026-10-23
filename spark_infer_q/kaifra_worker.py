"""One owner thread drives native plans and polls GPU completion events.

Clients submit/cancel independently; each generation has a bounded output queue.
No consumer drives the scheduler. Native publication and cancellation have a
serialized boundary, and slot reuse follows the completed device fence.
"""
from collections import deque
from concurrent.futures import Future,CancelledError,TimeoutError as FutureTimeout
from contextlib import nullcontext
from dataclasses import dataclass
from queue import Queue,Empty,Full
from threading import Condition,Thread
import time

from .deepinfer_bridge import BridgeError


class WorkerBusy(BridgeError):pass
class WorkerFailed(BridgeError):pass


@dataclass(frozen=True)
class StreamFailure:
    error: str


@dataclass
class Submission:
    request_id: str
    payload: object
    max_tokens: int
    temperature: float
    token_api: bool
    future: object


class Subscription:
    def __init__(self,worker,handle,capacity):
        self.worker,self.handle=worker,handle
        self.events=Queue(capacity);self.done=False;self.closed=False;self.cancel_requested=False

    def receive(self,timeout=None):
        if self.closed:raise WorkerFailed('stream is closed')
        item=self.events.get(timeout=timeout)
        if isinstance(item,StreamFailure):
            self.close();raise WorkerFailed(item.error)
        if item.finish_reason is not None:self.close()
        return item

    def cancel(self):return self.worker.cancel(self.handle)

    def close(self):self.worker.release(self)

    def __iter__(self):
        while True:
            event=self.receive();yield event
            if event.finish_reason is not None:return


class AsyncWorker:
    def __init__(self,engine_factory,*,context_factory=nullcontext,max_streams=2,
                 event_capacity=64,poll_seconds=.001):
        if (type(max_streams) is not int or max_streams<1 or type(event_capacity) is not int
                or event_capacity<2 or not 0<poll_seconds<=1):raise ValueError('invalid worker bounds')
        self.engine_factory,self.context_factory=engine_factory,context_factory
        self.max_streams,self.event_capacity=max_streams,event_capacity;self.poll_seconds=poll_seconds
        self.condition=Condition();self.pending=deque();self.ids={};self.subscriptions={};self.cancellations=set()
        self.ready=Future();self.thread=None;self.engine=None;self.state='new';self.error=None
        self.active_count=0;self.phase=None;self.capabilities={};self.last_step_metrics=()

    def start(self,timeout=30):
        with self.condition:
            if self.thread is not None:raise WorkerFailed('worker already started')
            self.state='starting';self.thread=Thread(target=self._run,name='kaifra-gpu-owner',daemon=False);self.thread.start()
        self.ready.result(timeout=timeout);return self

    def _submit(self,request_id,payload,max_tokens,temperature,token_api,timeout):
        if not isinstance(request_id,str) or not request_id:raise ValueError('nonempty request ID required')
        future=Future();command=Submission(request_id,payload,max_tokens,temperature,token_api,future)
        with self.condition:
            if self.state!='ready':raise WorkerFailed(self.error or 'worker is not accepting requests')
            if request_id in self.ids:raise WorkerBusy('request ID already has an open stream')
            if len(self.ids)>=self.max_streams:raise WorkerBusy('worker request streams are full')
            self.ids[request_id]=command;self.pending.append(command);self.condition.notify_all()
        try:return future.result(timeout=timeout)
        except CancelledError:
            raise WorkerBusy('request admission was cancelled') from None
        except FutureTimeout:
            if not future.cancel():
                def abandoned(done):
                    if not done.cancelled() and done.exception() is None:done.result().close()
                future.add_done_callback(abandoned)
            with self.condition:self.condition.notify_all()
            raise WorkerBusy('request admission timed out') from None

    def submit(self,request_id,messages,*,max_tokens=128,temperature=0,timeout=30):
        return self._submit(request_id,messages,max_tokens,temperature,False,timeout)

    def submit_tokens(self,request_id,ids,*,max_tokens=128,timeout=30):
        return self._submit(request_id,tuple(ids),max_tokens,0,True,timeout)

    def cancel(self,handle):
        with self.condition:
            sub=self.subscriptions.get(handle)
            if sub is None or sub.done or sub.closed:return False
            sub.cancel_requested=True;self.cancellations.add(handle);self.condition.notify_all();return True

    def cancel_id(self,request_id):
        with self.condition:
            owner=self.ids.get(request_id)
            if isinstance(owner,Subscription):return self.cancel(owner.handle)
            if isinstance(owner,Submission):
                cancelled=owner.future.cancel();self.condition.notify_all();return cancelled
            return False

    def release(self,sub):
        with self.condition:
            if sub.closed:return
            sub.closed=True
            if not sub.done:self.cancellations.add(sub.handle)
            self.subscriptions.pop(sub.handle,None)
            if self.ids.get(sub.handle.request_id) is sub:self.ids.pop(sub.handle.request_id)
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return dict(status=self.state,error=self.error,active_requests=self.active_count,
                open_streams=len(self.ids),pending_submissions=len(self.pending),phase=self.phase,
                capabilities=dict(self.capabilities))

    def wait_idle(self,timeout=30):
        end=time.monotonic()+timeout
        with self.condition:
            while self.active_count or self.pending or self.phase is not None:
                if self.state=='failed':raise WorkerFailed(self.error)
                left=end-time.monotonic()
                if left<=0:raise TimeoutError('worker did not become idle')
                self.condition.wait(min(left,.1))

    def close(self,timeout=30):
        with self.condition:
            if self.thread is None:self.state='closed';return
            if self.state not in ('failed','closed'):
                self.state='closing'
                for sub in self.subscriptions.values():
                    if not sub.done:sub.cancel_requested=True;self.cancellations.add(sub.handle)
                self.condition.notify_all()
        self.thread.join(timeout)
        if self.thread.is_alive():raise TimeoutError('worker still owns in-flight device state; do not release target')
        if self.error:raise WorkerFailed(self.error)

    def _cancel(self):
        for handle in tuple(self.cancellations):
            self.cancellations.discard(handle)
            if handle not in self.engine._requests:continue
            entry=self.engine.bridge.requests.get(handle.request_id)
            if entry is not None and entry['handle']==handle and not entry['cancelled']:
                self.engine.cancel(handle)

    def _admit(self):
        count=len(self.pending)
        for _ in range(count):
            command=self.pending.popleft();future=command.future
            if future.cancelled() or self.state=='closing':
                if not future.done():future.set_exception(WorkerFailed('worker is closing'))
                if self.ids.get(command.request_id) is command:self.ids.pop(command.request_id)
                continue
            if (self.engine.active_requests>=self.engine.bridge.max_requests or
                    command.request_id in self.engine.bridge.requests):
                self.pending.append(command);continue
            if not future.set_running_or_notify_cancel():
                if self.ids.get(command.request_id) is command:self.ids.pop(command.request_id)
                continue
            try:
                if command.token_api:handle=self.engine.submit_tokens(command.request_id,command.payload,max_tokens=command.max_tokens)
                else:handle=self.engine.submit(command.request_id,command.payload,max_tokens=command.max_tokens,temperature=command.temperature)
                sub=Subscription(self,handle,self.event_capacity)
                self.subscriptions[handle]=sub;self.ids[command.request_id]=sub
                future.set_result(sub)
            except (ValueError,BridgeError) as exc:
                if self.ids.get(command.request_id) is command:self.ids.pop(command.request_id)
                future.set_exception(type(exc)(str(exc)))
                if self.engine.bridge.poisoned:raise
            except Exception as exc:
                future.set_exception(WorkerFailed(str(exc)));raise

    def _offer(self,sub,item):
        try:sub.events.put_nowait(item)
        except Full:
            # A failed stream is explicit; it cannot silently drop output and
            # then claim successful completion. Native retirement still fences.
            while True:
                try:sub.events.get_nowait()
                except Empty:break
            sub.events.put_nowait(StreamFailure('slow_consumer'))
            sub.cancel_requested=True;self.cancellations.add(sub.handle)

    def _route(self,events):
        for event in events:
            sub=self.subscriptions.get(event.handle)
            if sub is None:
                if event.finish_reason is None:self.cancellations.add(event.handle)
                continue
            self._offer(sub,event)
            if event.finish_reason is not None:sub.done=True

    def _run(self):
        plan=None
        try:
            with self.context_factory():
                self.engine=self.engine_factory()
                with self.condition:
                    self.capabilities=dict(self.engine.capabilities);self.state='ready';self.ready.set_result(True);self.condition.notify_all()
                try:
                    while True:
                        with self.condition:
                            self._cancel()
                            if plan is None:self._admit()
                            self.active_count=self.engine.active_requests
                            if plan is None and not self.active_count:
                                self.phase=None;self.condition.notify_all()
                                if self.state=='closing':break
                                self.condition.wait(.1);continue
                        if plan is None:
                            plan=self.engine.bridge.prepare()
                            with self.condition:
                                self.phase='prepared';self._cancel()
                            self.engine.bridge.execute_async(plan)
                        ready=self.engine.bridge.poll(plan)
                        with self.condition:
                            self._cancel()
                            if ready:
                                # Cancellation registration cannot race native
                                # publication across this serialized boundary.
                                self._route(self.engine.finish_plan(plan));plan=None;self.phase=None
                                self.last_step_metrics=self.engine.last_step_metrics
                                self.capabilities=dict(self.engine.capabilities)
                            else:self.phase=getattr(getattr(plan,'execution',None),'phase','inflight')
                            self.active_count=self.engine.active_requests;self.condition.notify_all()
                            if not ready:self.condition.wait(self.poll_seconds)
                except Exception:
                    self.engine.bridge.poisoned=True
                    raise
                finally:
                    self.engine.close()
        except Exception as exc:
            with self.condition:
                self.error=type(exc).__name__+': '+str(exc);self.state='failed'
                if not self.ready.done():self.ready.set_exception(WorkerFailed(self.error))
        finally:
            with self.condition:
                if self.state!='failed':self.state='closed'
                for command in self.pending:
                    if not command.future.done():command.future.set_exception(WorkerFailed(self.error or 'worker closed'))
                self.pending.clear()
                for sub in self.subscriptions.values():
                    if not sub.done:
                        self._offer(sub,StreamFailure(self.error or 'worker closed'));sub.done=True
                self.active_count=0;self.phase=None;self.condition.notify_all()
