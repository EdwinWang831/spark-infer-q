"""Observe competing CUDA contexts without changing other users' processes."""
import os
import threading
import time


class GPUMonitor:
    def __init__(self, device):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        self.pid = os.getpid()
        self.epoch = 0
        self.events = []
        self.samples = 0
        self.last_busy = 0.0
        self.last_poll = None
        self.max_poll_gap_s = 0.0
        self.error = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.poll()
        self.thread.start()

    def poll(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            try:
                processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                others = sorted(process.pid for process in processes if process.pid != self.pid)
                now = time.monotonic()
                if self.last_poll is not None:
                    self.max_poll_gap_s = max(self.max_poll_gap_s, now-self.last_poll)
                self.last_poll = now
                self.samples += 1
                if others:
                    self.epoch += 1
                    self.last_busy = now
                    self.events.append({'unix': time.time(), 'pids': others, 'epoch': self.epoch})
                return self.epoch
            except Exception as error:
                self.error = f'GPU process monitor failed: {error}'
                raise

    def _loop(self):
        while not self.stop.wait(.1):
            try:
                self.poll()
            except Exception:
                return  # The main thread raises the recorded monitoring failure.

    def quiet_checkpoint(self):
        deadline = time.monotonic()+120
        while time.monotonic() < deadline:
            epoch = self.poll()
            with self.lock:
                if time.monotonic()-self.last_busy >= 1.0:
                    return epoch
            time.sleep(.1)
        raise RuntimeError('no quiet GPU interval within 120 seconds')

    def unchanged(self, epoch):
        return self.poll() == epoch

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.nvml.nvmlShutdown()
        return {'interval_s': .1, 'quiet_after_competitor_s': 1.0, 'samples': self.samples,
                'max_poll_gap_s': self.max_poll_gap_s, 'events': self.events, 'error': self.error}
