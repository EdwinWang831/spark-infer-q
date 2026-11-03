"""One prepared model, one GPU owner and one bounded local inference service."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
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
    capacity: int = 131072
    max_requests: int = 4
    mode: str = 'adaptive'
    execution: str = 'graph'
    queue_size: int = 4
    admission_timeout: float = 30.
    prefill_chunk_rows: int | None = None
    prefill_experts: Path | None = None
    prefill_batch_control: Path | None = None
    feature_history_layout: str = 'mirrored'

    def __post_init__(self):
        if type(self.port) is not int or not 0 <= self.port <= 65535:
            raise ValueError('port must be 0..65535')
        if type(self.capacity) is not int or not 16 <= self.capacity <= 131072:
            raise ValueError('context capacity must be 16..131072, including output')
        if type(self.max_requests) is not int or not 1 <= self.max_requests <= 4:
            raise ValueError('running request capacity must be 1..4')
        if type(self.queue_size) is not int or not 0 <= self.queue_size <= 128:
            raise ValueError('queue size must be 0..128')
        if not math.isfinite(self.admission_timeout) or not 0 < self.admission_timeout <= 300:
            raise ValueError('admission timeout must be in (0, 300] seconds')
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError('nonempty served model name required')
        if self.mode not in ('adaptive', 'none') or self.execution not in ('graph', 'eager'):
            raise ValueError('unsupported mode or execution backend')
        if self.prefill_chunk_rows is not None and (type(self.prefill_chunk_rows) is not int
                or self.prefill_chunk_rows not in (128, 256, 512, 1024)):
            raise ValueError('prefill chunk rows must be 128, 256, 512 or 1024')
        if self.feature_history_layout not in ('mirrored','linear'):
            raise ValueError('feature layout must be mirrored or linear')
        if self.feature_history_layout=='linear' and (self.mode!='adaptive' or self.capacity<=1536 and self.max_requests==2):
            raise ValueError('linear feature history requires long adaptive serving')
        if self.prefill_batch_control is not None and (self.capacity != 131072 or self.max_requests != 4
                or self.mode != 'adaptive' or self.execution != 'graph' or self.prefill_experts is None
                or self.feature_history_layout != 'linear' or self.prefill_chunk_rows not in (None, 1024)):
            raise ValueError('batch prefill requires the four-slot 128K adaptive Graph/E44/linear profile')

    @property
    def prompt_chunk(self):
        if self.prefill_batch_control is not None:
            return 1024
        if self.prefill_chunk_rows is not None:
            return self.prefill_chunk_rows
        if self.capacity > 65536 and self.max_requests == 4:
            return 512
        if self.prefill_experts is not None:
            return 1024
        return 512 if self.capacity > 1536 or self.max_requests != 2 else 128

    @property
    def index_ring_rows(self):
        return 2048 if self.capacity > 65536 else None


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
    if config.prefill_experts is not None:
        from .prefill_experts import E44_BINARY_SHA256
        path = config.prefill_experts.resolve()
        checks['prefill_expert_identity'] = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == E44_BINARY_SHA256
    if config.prefill_batch_control is not None:
        from .prompt_batch import CONTROL_SHA256, HEADER_SHA256
        path = config.prefill_batch_control.resolve()
        checks['prompt_control_identity'] = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == CONTROL_SHA256
        checks['prompt_control_headers'] = all((root.parent/'DeepInfer'/name).is_file()
            and hashlib.sha256((root.parent/'DeepInfer'/name).read_bytes()).hexdigest() == digest
            for name, digest in HEADER_SHA256.items())
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


def memory_snapshot(torch):
    host = {line.split(':')[0]: int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemAvailable:')}
    return dict(allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                host_available_bytes=host['MemAvailable'])


def check_memory(torch):
    snapshot = memory_snapshot(torch)
    if (snapshot['reserved_bytes'] > 88*1024**3 or
            snapshot['host_available_bytes'] < 8*1024**3):
        raise RuntimeError('service memory limit or host reserve exceeded: '+str(snapshot))
    return snapshot


def warm_long_slots(target, provider, torch, *, graphs=True):
    """Charge measured first-slot Graph/state bytes before adding more slots."""
    from .long_capacity import LongCapacityBudget
    base = check_memory(torch)
    physical = getattr(target, 'capacity', 32784)
    initial_reserve = (4 if physical > 65552 else 3 if physical > 32784 else 2)*1024**3
    if (base['allocated_bytes']+initial_reserve > 88*1024**3 or
            base['host_available_bytes']-initial_reserve < 8*1024**3):
        raise RuntimeError('insufficient memory to measure the first request slot')
    samples = []
    feature_bytes = 0
    if provider is not None:
        copies = 2 if getattr(provider,'feature_history_layout','mirrored')=='mirrored' else 1
        feature_bytes = copies*provider.history_window_rows*12800*2 + sum(getattr(provider.worker, name).numel()
            *getattr(provider.worker, name).element_size()
            for name in ('_context_rows', '_keys', '_values'))
    budget = None
    for index, slot in enumerate(target.targets):
        if budget is not None:
            snap = check_memory(torch)
            budget.require(len(target.targets), allocated_bytes=snap['allocated_bytes'],
                reserved_bytes=snap['reserved_bytes'], host_available_bytes=snap['host_available_bytes'])
        if graphs:
            slot.warmup()
        else:
            state = slot.state()
            slot.release_state(state)
        snap = check_memory(torch)
        samples.append(snap)
        if index == 0:
            private = snap['allocated_bytes']-base['allocated_bytes']
            # Also charge allocator slack: it shares physical RAM with host
            # MemAvailable and must not be counted as extra free capacity.
            reserve = max(1024**3, snap['reserved_bytes']-snap['allocated_bytes'])
            budget = LongCapacityBudget(base['allocated_bytes'], private+feature_bytes,
                                       reserve, 88*1024**3, 8*1024**3)
    return dict(budget=vars(budget), slot_samples=samples,
                admission=budget.require(len(target.targets), allocated_bytes=snap['allocated_bytes'],
                    reserved_bytes=snap['reserved_bytes'], host_available_bytes=snap['host_available_bytes']))


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

    target = prefill = worker = service = provider = None
    long_context = config.capacity > 1536 or config.max_requests != 2
    with NativeScheduler(root.parent/'DeepInfer', root.parent/'build-release/lib/deepinfer',
                         max_drafts=7 if config.mode == 'adaptive' else 0) as native:
        torch = session.torch
        try:
            control = None
            if config.prefill_batch_control is not None:
                from .prompt_batch import PromptBatchControl
                control = PromptBatchControl.from_file(config.prefill_batch_control, native)
            with torch.inference_mode():
                profile, admission = {}, None
                if config.mode == 'adaptive':
                    # Resident sessions created before v0.4 may retain separate
                    # copies of the exact parameters already in fused matrices.
                    from .dflash2_reference import share_fused_weights
                    share_fused_weights(session.draft)
                    options = dict(history_window_rows=2040, max_request_context=config.capacity+16,
                        query_graphs=config.execution == 'graph', request_projection_cache=True,
                        feature_history_layout=config.feature_history_layout) if long_context else {}
                    provider, profile = optimized_provider(session, **options)
                # There is one GPU owner. Reusing the caller's stream also
                # bounds per-stream library workspaces across resident restarts.
                stream = torch.cuda.current_stream()
                with torch.cuda.stream(stream):
                    physical = config.capacity+16 if long_context else 2048
                    chunk = config.prompt_chunk
                    expert = None
                    if config.prefill_experts is not None:
                        from .prefill_experts import E44_BINARY_SHA256, PrefillExpertTile
                        expert = PrefillExpertTile.from_file(config.prefill_experts,
                                                           expected_sha256=E44_BINARY_SHA256)
                    executor_class, provider_class = PrefillExecutor, TensorFoldPrefillProvider
                    if control is not None:
                        from .prefill_batch import BatchPrefillExecutor, BatchPrefillProvider
                        executor_class, provider_class = BatchPrefillExecutor, BatchPrefillProvider
                    prefill = executor_class(provider_class(session.weights, torch,
                        capacity=physical, expert_backend=expert,
                        index_ring_rows=config.index_ring_rows), chunk_rows=chunk)
                    if long_context:
                        from .shared_graph import SharedGraphTarget
                        target = SharedGraphTarget.from_session(session, capacity=physical,
                            kv_dtype='bf16', slots=config.max_requests,
                            index_ring_rows=config.index_ring_rows)
                        # Materialize shared storage before measuring private
                        # slots. This is one existing stream, never a reload.
                        prefill.workspace = prefill.provider.allocate(prefill.workspace_rows)
                        if provider is not None and config.execution == 'graph':
                            provider.worker.warmup()
                        if config.execution == 'graph':
                            # These head Graphs are shared by all request slots.
                            # Materialize them before measuring private growth.
                            for width in range(1, 9):
                                target.screen.prepare(width)
                        # Eager mode also allocates stable request state; only
                        # the graph profile records the sixteen private Graphs.
                        admission = warm_long_slots(target, provider, torch,
                                                    graphs=config.execution == 'graph')
                    else:
                        head = FusedScreenedHead(session.weights.head)
                        if not head.static_ok:
                            raise ValueError('target output head does not satisfy the E33 range gate')
                        target = BatchTarget(WideTarget(session.weights, capacity=physical), head,
                                             graphs=config.execution == 'graph')
                        target.warmup()

            @contextmanager
            def context():
                with torch.inference_mode(), torch.cuda.stream(stream):
                    yield

            def make():
                if long_context:
                    if config.mode == 'adaptive':
                        from .kaifra_speculative import speculative_engine
                        engine = speculative_engine(native, target, provider, eos_ids=session.eos,
                            prior=session.calibration['conditional_prior'], capacity=config.capacity,
                            max_requests=config.max_requests, token_budget=chunk if control is None else 4*chunk,
                            tokenizer=session.tokenizer, prompt_control=control,
                            prefill_executor=prefill, output_cache=True,
                            serialized_postdraft=config.max_requests > 1,
                            interleave_prefill=config.max_requests > 1)
                    else:
                        from .shared_graph import shared_graph_engine
                        engine = shared_graph_engine(native, target, eos_ids=session.eos,
                            capacity=config.capacity, token_budget=chunk, tokenizer=session.tokenizer,
                            prefill_executor=prefill, output_cache=True,
                            interleave_prefill=config.max_requests > 1)
                else:
                    engine = batch_engine(native, target, provider, eos_ids=session.eos,
                        prior=session.calibration['conditional_prior'], capacity=config.capacity,
                        tokenizer=session.tokenizer, target_only=config.mode == 'none',
                        prefill_executor=prefill, prompt_token_budget=chunk, interleave_prefill=True)
                engine.capabilities.update(service_version='0.4.1', max_running_requests=config.max_requests,
                    context_capacity=config.capacity, target_kv_dtype='bf16',
                    index_ring_rows=config.index_ring_rows,
                    feature_history_layout=config.feature_history_layout,
                    max_queued_requests=config.queue_size, protocol='openai_chat_subset',
                    http_async_owner=True, memory_admission=admission)
                engine.capabilities.update(prefill_expert_backend='original' if expert is None else expert.name,
                    prefill_expert_sha256=None if expert is None else E44_BINARY_SHA256)
                return engine

            worker = AsyncWorker(make, context_factory=context, max_streams=config.max_requests+config.queue_size)
            worker.start()
            native.bind_worker(worker.thread)
            server = make_server(worker, ('127.0.0.1', config.port), protocol='openai',
                model_id=config.model, admission_timeout=config.admission_timeout)
            metadata = dict(mode=config.mode, execution=config.execution,
                            capacity=config.capacity, max_requests=config.max_requests,
                            profile=profile, native=native.metadata)
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
                        if provider is not None:
                            close = getattr(provider.worker, 'close', None)
                            if close is not None:
                                close()


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
