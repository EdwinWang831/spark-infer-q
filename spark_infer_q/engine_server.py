"""One prepared model, one GPU owner and one bounded local inference service."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import signal
import sys
from threading import Event, Thread, current_thread, main_thread


@dataclass(frozen=True)
class ServiceConfig:
    root: Path
    checkpoint: Path
    port: int = 8000
    model: str = 'spark-infer-q'
    capacity: int = 1536
    mode: str = 'adaptive'
    execution: str = 'graph'
    queue_size: int = 4
    admission_timeout: float = 30.

    def __post_init__(self):
        if type(self.port) is not int or not 0 <= self.port <= 65535:
            raise ValueError('port must be 0..65535')
        if type(self.capacity) is not int or not 16 <= self.capacity <= 1536:
            raise ValueError('v0.1 context capacity must be 16..1536, including output')
        if type(self.queue_size) is not int or not 0 <= self.queue_size <= 128:
            raise ValueError('queue size must be 0..128')
        if not math.isfinite(self.admission_timeout) or not 0 < self.admission_timeout <= 300:
            raise ValueError('admission timeout must be in (0, 300] seconds')
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError('nonempty served model name required')
        if self.mode not in ('adaptive', 'none') or self.execution not in ('graph', 'eager'):
            raise ValueError('unsupported mode or execution backend')


def inspect_environment(config):
    root, checkpoint = config.root.resolve(), config.checkpoint.resolve()
    required = {
        'target_config': checkpoint/'config.json',
        'generation_config': checkpoint/'generation_config.json',
        'target_adapter': root/'e3_target.py',
        'wide_target_adapter': root/'e3_wide_target.py',
        'observed_target_adapter': root/'e12_routes.py',
        'draft_reference': root/'reference/dflash.py',
        'draft_config': root/'checkpoint/config.json',
        'calibration': root/'calibration.json',
        'native_source': root.parent/'DeepInfer',
        'native_extension': root.parent/'build-release/lib/deepinfer',
    }
    if config.mode == 'adaptive':
        required.update(costs=root/'artifacts/e29-frozen-costs.json',
                        linear_model=root/'artifacts/e27-linear-model.json',
                        w8_source=root/'draft_w8_kernel.cu')
    checks = {key: path.exists() for key, path in required.items()}
    checks['q4_extension'] = any((root/'build/qmm').glob('spark_q_e6_qmm*.so'))
    checks['varlen_extension'] = any((root/'build/varlen').glob('spark_q_e3_varlen*.so'))
    if config.mode == 'adaptive':
        checks['w8_extension'] = any((root/'build/draft_w8').glob('spark_q_draft_w8*.so'))
    return dict(ready=all(checks.values()), checks=checks,
        scope='Prepared NVFP4 Qwen / DFlash2 / DeepInfer / TensorFold deployment; no downloads',
        gpu_checked=False, model_loaded=False)


class EngineService:
    def __init__(self, worker, server, metadata):
        self.worker, self.server, self.metadata = worker, server, metadata
        self.thread = Thread(target=server.serve_forever, name='kaifra-http', daemon=False)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        # Complete/cancel device work before waiting for HTTP handlers to leave.
        try:
            self.worker.close()
        finally:
            if self.worker.thread is None or not self.worker.thread.is_alive():
                self.server.server_close()
                self.thread.join(5)
        if self.thread.is_alive():
            raise TimeoutError('HTTP listener did not stop')


@contextmanager
def open_service(session, config):
    """Reuse an existing session; the CLI is responsible for loading it once."""
    root = config.root.resolve()
    for path in (root, root/'followup-deps', root/'python-deps'):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    from .native_scheduler import NativeScheduler
    from .kaifra_optimized import optimized_provider
    from .kaifra_batch import batch_engine
    from .kaifra_batch_target import BatchTarget
    from .kaifra_worker import AsyncWorker
    from .kaifra_service import make_server
    from .screened_head_fused import FusedScreenedHead
    from .prefill import PrefillExecutor
    from .prefill_tensorfold import TensorFoldPrefillProvider
    from e3_wide_target import WideTarget

    target = prefill = worker = service = None
    with NativeScheduler(root.parent/'DeepInfer', root.parent/'build-release/lib/deepinfer',
                         max_drafts=7 if config.mode == 'adaptive' else 0) as native:
        torch = session.torch
        try:
            with torch.inference_mode():
                provider, profile = optimized_provider(session) if config.mode == 'adaptive' else (None, {})
                head = FusedScreenedHead(session.weights.head)
                if not head.static_ok:
                    raise ValueError('target output head does not satisfy the E33 range gate')
                target = BatchTarget(WideTarget(session.weights, capacity=2048), head,
                                     graphs=config.execution == 'graph')
                target.warmup()
                # There is one GPU owner. Reusing the caller's stream also
                # bounds per-stream library workspaces across resident restarts.
                stream = torch.cuda.current_stream()
                with torch.cuda.stream(stream):
                    prefill = PrefillExecutor(TensorFoldPrefillProvider(session.weights, torch,
                        capacity=2048), chunk_rows=128)

            @contextmanager
            def context():
                with torch.inference_mode(), torch.cuda.stream(stream):
                    yield

            def make():
                engine = batch_engine(native, target, provider, eos_ids=session.eos,
                    prior=session.calibration['conditional_prior'], capacity=config.capacity,
                    tokenizer=session.tokenizer, target_only=config.mode == 'none',
                    prefill_executor=prefill, prompt_token_budget=128, interleave_prefill=True)
                engine.capabilities.update(service_version='0.1.0', max_running_requests=2,
                    max_queued_requests=config.queue_size, protocol='openai_chat_subset')
                return engine

            worker = AsyncWorker(make, context_factory=context, max_streams=2+config.queue_size)
            worker.start()
            native.bind_worker(worker.thread)
            server = make_server(worker, ('127.0.0.1', config.port), protocol='openai',
                model_id=config.model, admission_timeout=config.admission_timeout)
            metadata = dict(mode=config.mode, execution=config.execution,
                            capacity=config.capacity, profile=profile, native=native.metadata)
            service = EngineService(worker, server, metadata)
            yield service
        finally:
            try:
                if service is not None:
                    service.close()
                elif worker is not None:
                    worker.close()
            finally:
                # A timed-out owner retains its resources until its fence completes.
                if worker is None or worker.thread is None or not worker.thread.is_alive():
                    try:
                        if prefill is not None:
                            prefill.close()
                    finally:
                        if target is not None:
                            target.close()


def run_service(service):
    stop = Event()
    previous = {}
    if current_thread() is main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: stop.set())
    try:
        print(json.dumps(dict(status='READY', address=service.server.server_address,
            metadata=service.metadata, worker=service.worker.snapshot())), flush=True)
        while not stop.wait(.25):
            if service.worker.snapshot()['status'] == 'failed':
                raise RuntimeError('GPU owner failed; inspect the health endpoint')
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
