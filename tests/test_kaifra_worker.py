"""Deterministic owner-thread/fence tests; these fakes are not GPU batch evidence."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import time
import unittest

from spark_infer_q.kaifra_runtime import KaifraEngine
from spark_infer_q.kaifra_worker import AsyncWorker,WorkerBusy,WorkerFailed
from spark_infer_q.native_scheduler import NativeScheduler
from test_kaifra_runtime import ScriptedBridge


class DeferredBridge(ScriptedBridge):
    max_requests=2
    def __init__(self,gate=None):
        super().__init__();self.gate=gate or Event();self.started=Event();self.pending=None;self.batch_sizes=[]
    def prepare(self):
        self._ready()
        if self.pending is not None:raise RuntimeError('duplicate pending plan')
        self.pending=SimpleNamespace(executed=False,metrics=());return self.pending
    def execute_async(self,plan):
        if plan is not self.pending:raise RuntimeError('foreign plan')
        self.batch_sizes.append(len(self.requests));self.started.set()
    def poll(self,plan):
        if plan is not self.pending:raise RuntimeError('foreign plan')
        plan.executed=self.gate.is_set();return plan.executed
    def complete(self,plan):
        if plan is not self.pending or not plan.executed:raise RuntimeError('publish before completion')
        super().step();self.pending=None


class WorkerTests(unittest.TestCase):
    def worker(self,gate=None,capacity=64):
        bridge=DeferredBridge(gate);engine=KaifraEngine(bridge)
        worker=AsyncWorker(lambda:engine,event_capacity=capacity).start()
        self.addCleanup(self.clean,worker,bridge)
        return worker,bridge
    def clean(self,worker,bridge):
        bridge.gate.set()
        try:worker.close(3)
        except WorkerFailed:pass
    def wait_for(self,predicate,timeout=3):
        end=time.monotonic()+timeout
        while not predicate():
            if time.monotonic()>end:self.fail('bounded worker condition timed out')
            time.sleep(.001)

    def test_two_consumers_receive_only_their_native_output(self):
        worker,bridge=self.worker();first=worker.submit_tokens('first',[1],max_tokens=4)
        self.assertTrue(bridge.started.wait(2))
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(worker.submit_tokens,'second',[2],max_tokens=3)
            self.wait_for(lambda:worker.snapshot()['pending_submissions']==1)
            bridge.gate.set();second=future.result(3)
        one,two=list(first),list(second)
        self.assertEqual([t for e in one for t in e.token_ids],[3]*4)
        self.assertEqual([t for e in two for t in e.token_ids],[3]*3)
        self.assertTrue(all(e.handle==first.handle for e in one));self.assertTrue(all(e.handle==second.handle for e in two))
        worker.wait_idle();self.assertIn(2,bridge.batch_sizes);self.assertEqual(worker.snapshot()['open_streams'],0)

    def test_inflight_cancel_holds_state_and_same_id_reuse_has_new_generation(self):
        worker,bridge=self.worker();old=worker.submit_tokens('reuse',[1],max_tokens=8)
        self.assertTrue(bridge.started.wait(2));self.assertTrue(old.cancel())
        self.assertEqual(bridge.released,[]);old.close()
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(worker.submit_tokens,'reuse',[1],max_tokens=2)
            self.wait_for(lambda:worker.snapshot()['pending_submissions']==1)
            self.assertFalse(future.done());bridge.gate.set();new=future.result(3)
        self.assertGreater(new.handle.generation,old.handle.generation)
        self.assertFalse(worker.cancel(old.handle))
        self.assertEqual([t for e in new for t in e.token_ids],[3,3]);worker.wait_idle()
        self.assertEqual(bridge.released.count(old.handle),1)

    def test_slow_consumer_is_failed_and_cancelled_without_blocking_worker(self):
        gate=Event();gate.set();worker,bridge=self.worker(gate,capacity=2)
        sub=worker.submit_tokens('slow',[1],max_tokens=20);worker.wait_idle()
        with self.assertRaisesRegex(WorkerFailed,'slow_consumer'):sub.receive(2)
        self.assertEqual(len(bridge.released),1);self.assertEqual(worker.snapshot()['status'],'ready')
        follow=worker.submit_tokens('next',[1],max_tokens=1);self.assertEqual(list(follow)[0].token_ids,(3,))

    def test_bounded_close_refuses_to_free_an_unfinished_device_generation(self):
        worker,bridge=self.worker();sub=worker.submit_tokens('closing',[1],max_tokens=9)
        self.assertTrue(bridge.started.wait(2))
        with self.assertRaises(TimeoutError):worker.close(.02)
        self.assertEqual(bridge.released,[]);self.assertTrue(worker.thread.is_alive())
        bridge.gate.set();worker.close(3)
        self.assertEqual(list(sub)[-1].finish_reason,'cancelled');self.assertEqual(len(bridge.released),1)

    def test_full_streams_invalid_input_and_worker_failure_are_distinct(self):
        worker,bridge=self.worker();first=worker.submit_tokens('first',[1],max_tokens=2)
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(worker.submit_tokens,'second',[1],max_tokens=2)
            self.wait_for(lambda:worker.snapshot()['pending_submissions']==1)
            with self.assertRaises(WorkerBusy):worker.submit_tokens('third',[1],max_tokens=2)
            bridge.gate.set();second=future.result(3)
        list(first);list(second);worker.wait_idle()
        with self.assertRaises(ValueError):worker.submit_tokens('invalid',[],max_tokens=1)
        self.assertEqual(worker.snapshot()['status'],'ready')
        bridge.fail=True
        bad=worker.submit_tokens('failure',[1],max_tokens=4)
        with self.assertRaisesRegex(WorkerFailed,'injected'):bad.receive(3)
        self.wait_for(lambda:worker.snapshot()['status']=='failed')
        with self.assertRaises(WorkerFailed):worker.submit_tokens('next',[1],max_tokens=1)
        self.assertTrue(worker.engine.closed)

    def test_native_group_cannot_close_beneath_a_live_owner(self):
        native=NativeScheduler('/unused','/unused')
        native._worker_thread=SimpleNamespace(is_alive=lambda:True)
        with self.assertRaisesRegex(RuntimeError,'live worker'):native.close()
        native._worker_thread=SimpleNamespace(is_alive=lambda:False);native.close()

    def test_device_wait_releases_command_lock_and_retains_shutdown_fence(self):
        class WaitingBridge(DeferredBridge):
            def __init__(self):super().__init__();self.waiting=Event()
            def wait_for_device(self,plan):
                if plan is not self.pending:raise RuntimeError('foreign wait')
                self.waiting.set();self.gate.wait();return True
        bridge=WaitingBridge();worker=AsyncWorker(lambda:KaifraEngine(bridge)).start()
        self.addCleanup(self.clean,worker,bridge)
        sub=worker.submit_tokens('device-wait',[1],max_tokens=9)
        self.assertTrue(bridge.waiting.wait(2))
        # Registration must finish while the owner is blocked in the device
        # wait, without exposing output or freeing its native generation.
        with ThreadPoolExecutor(1) as pool:self.assertTrue(pool.submit(sub.cancel).result(1))
        self.assertEqual(bridge.released,[])
        with self.assertRaises(TimeoutError):worker.close(.02)
        self.assertTrue(worker.thread.is_alive());self.assertEqual(bridge.released,[])
        bridge.gate.set();worker.close(3)
        events=list(sub);self.assertFalse(any(e.token_ids for e in events))
        self.assertEqual(events[-1].finish_reason,'cancelled');self.assertEqual(len(bridge.released),1)

    def test_device_wait_failure_quarantines_and_fences_before_releasing(self):
        class BrokenWaitBridge(DeferredBridge):
            def wait_for_device(self,plan):raise RuntimeError('injected event wait failure')
        bridge=BrokenWaitBridge();fences=[]
        bridge.backend.synchronize=lambda:fences.append('fenced')
        worker=AsyncWorker(lambda:KaifraEngine(bridge)).start();self.addCleanup(self.clean,worker,bridge)
        sub=worker.submit_tokens('wait-failure',[1],max_tokens=4)
        with self.assertRaisesRegex(WorkerFailed,'event wait failure'):sub.receive(2)
        self.wait_for(lambda:not worker.thread.is_alive())
        self.assertEqual(worker.snapshot()['status'],'failed');self.assertEqual(fences,['fenced'])
        self.assertEqual(bridge.released,[sub.handle]);self.assertTrue(worker.engine.closed)


if __name__=='__main__':unittest.main()
