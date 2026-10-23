"""K1 synchronous target-only runtime, using DeepInfer's actual Scheduler.

There is no replacement scheduling algorithm here: admission, prompt chunks,
decode plans and output accounting belong to SchedulerBridge. The runtime adds
bounded output events, text tokenization, metrics and explicit ownership.
"""
from dataclasses import dataclass, field
from threading import RLock
import time

from .deepinfer_bridge import BridgeError, SchedulerBridge


@dataclass(frozen=True)
class OutputEvent:
    handle: object
    token_ids: tuple[int, ...]
    text: str | None
    finish_reason: str | None
    processed_position: int
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    ttft_seconds: float | None
    decode_seconds: float


@dataclass
class _Request:
    prompt_tokens: int
    max_tokens: int
    submitted: float
    tokens: list[int] = field(default_factory=list)
    first_token: float | None = None


class QwenTargetBackend:
    """Own per-request state while borrowing one loaded target and its buffers.

    Each scheduled request still executes separately. Prefill may use the E8
    validated row chunks; those chunks are not cross-request GPU batching.
    """
    def __init__(self, target, *, prefill_rows=8, observer=None):
        if type(prefill_rows) is not int or prefill_rows not in (1, 2, 4, 8):
            raise ValueError('prefill_rows must be 1, 2, 4 or 8')
        if any(n not in target.buffers for n in range(1, prefill_rows+1)):
            raise ValueError('target lacks requested row buffers')
        self.target, self.prefill_rows, self.observer = target, prefill_rows, observer
        self.states = {}

    def create(self, handle):
        if handle in self.states:
            raise BridgeError('duplicate device state')
        with self.target.torch.inference_mode():
            state = self.target.state()
        self.states[handle] = state
        return state

    def position(self, state):
        return state.pos

    def advance(self, row, state):
        if self.states.get(row.handle) is not state or state.pos != row.position:
            raise BridgeError('device state ownership or position mismatch')
        size = self.prefill_rows if row.prompt else 1
        with self.target.torch.inference_mode():
            for begin in range(0, len(row.token_ids), size):
                ids = list(row.token_ids[begin:begin+size])
                logits, _ = self.target.step(state, ids, capture=False)
                if self.observer is not None:
                    self.observer(row.handle, row.position+begin, ids, logits, state)
            # Target.step includes its GPU completion fence. argmax is read only
            # after all planned rows have been committed to this private state.
            return int(logits[-1].argmax())

    def release(self, handle, state):
        if self.states.get(handle) is not state:
            raise BridgeError('foreign or already released device state')
        del self.states[handle]

    def synchronize(self):
        self.target.torch.cuda.synchronize()


class KaifraEngine:
    """Serialized request API over the existing native scheduler bridge.

    One caller drives step(); submit/cancel are serialized at step boundaries.
    Events contain only native committed output, never backend predictions.
    Consumers own returned events; terminal outputs are not retained forever.
    """
    capabilities = dict(target_only=True, greedy=True, streaming=True,
                        gpu_batch=False, asynchronous=False, speculative=False)

    def __init__(self, bridge, *, tokenizer=None, clock=time.perf_counter):
        self.bridge, self.tokenizer, self.clock = bridge, tokenizer, clock
        self._requests = {}
        self._lock = RLock()
        self.closed = False

    @classmethod
    def from_native(cls, native, target, *, eos_ids, capacity=1024,
                    max_requests=2, token_budget=32, prefill_rows=8,
                    tokenizer=None, observer=None):
        if capacity > target.capacity:
            raise ValueError('scheduler capacity exceeds target capacity')
        backend = QwenTargetBackend(target, prefill_rows=prefill_rows, observer=observer)
        bridge = SchedulerBridge(native.core, native.torch, native.group,
            native.sampling_class, backend, vocab_size=int(target.w.head.b.weight.shape[0]),
            eos_ids=eos_ids, capacity=capacity, max_requests=max_requests,
            token_budget=token_budget)
        return cls(bridge, tokenizer=tokenizer)

    def _ready(self):
        if self.closed:
            raise BridgeError('engine is closed')
        self.bridge._ready()

    @property
    def active_requests(self):
        with self._lock:
            return len(self._requests)

    def submit(self, request_id, messages, *, max_tokens=128, temperature=0):
        with self._lock:
            self._ready()
            if self.tokenizer is None:
                raise ValueError('text requests require a tokenizer')
            if temperature != 0:
                raise ValueError('K1 supports greedy temperature=0 only')
            if (not isinstance(messages, list) or not messages or any(
                    not isinstance(m, dict) or set(m) != {'role', 'content'} or
                    m['role'] not in ('system', 'user', 'assistant') or
                    not isinstance(m['content'], str) for m in messages)):
                raise ValueError('messages must contain text role/content entries')
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            return self.submit_tokens(request_id, ids, max_tokens=max_tokens)

    def submit_tokens(self, request_id, prompt_ids, *, max_tokens=128):
        with self._lock:
            self._ready()
            ids = tuple(prompt_ids)
            submitted = self.clock()
            try:
                handle = self.bridge.add(request_id, ids, max_tokens)
            except (ValueError, BridgeError):
                raise
            except Exception:
                self.bridge.poisoned = True
                raise
            self._requests[handle] = _Request(len(ids), max_tokens, submitted)
            return handle

    def cancel(self, handle):
        with self._lock:
            self._ready()
            try:
                self.bridge.cancel(handle)
            except BridgeError:
                raise
            except Exception:
                self.bridge.poisoned = True
                raise

    def step(self):
        with self._lock:
            self._ready()
            if not self._requests:
                return []
            try:
                self.bridge.step()
                now, events = self.clock(), []
                for handle, record in list(self._requests.items()):
                    done = self.bridge.completed.get(handle)
                    entry = self.bridge.requests.get(handle.request_id)
                    if done is not None:
                        tokens = tuple(done['token_ids'])
                        position = done['processed_position']
                        reason = ('cancelled' if done['cancelled'] else 'stop'
                                  if tokens and tokens[-1] in self.bridge.eos_ids else 'length')
                    else:
                        if entry is None or entry['handle'] != handle:
                            raise BridgeError('native request disappeared')
                        tokens = tuple(entry['seq'].data.get_token_ids()[record.prompt_tokens:])
                        position, reason = entry['seq'].num_cached_tokens, None
                    if (tokens[:len(record.tokens)] != tuple(record.tokens) or
                            len(tokens) > record.max_tokens):
                        raise BridgeError('native output prefix or length contract violated')
                    new = tokens[len(record.tokens):]
                    if new and record.first_token is None:
                        record.first_token = now
                    record.tokens[:] = tokens
                    if new or reason is not None:
                        # Full text is a snapshot: byte-level token decoders may
                        # revise an incomplete Unicode suffix on the next token.
                        text = (self.tokenizer.decode(list(tokens), skip_special_tokens=True)
                                if self.tokenizer is not None else None)
                        events.append(OutputEvent(handle, new, text, reason, position,
                            record.prompt_tokens, len(tokens), now-record.submitted,
                            None if record.first_token is None else record.first_token-record.submitted,
                            0.0 if record.first_token is None else now-record.first_token))
                    if done is not None:
                        del self._requests[handle]
                        del self.bridge.completed[handle]
                if not self._requests:
                    # Native running queues may still contain terminal entries
                    # until one final scheduling pass. Clear that bookkeeping
                    # before reporting the runtime idle or reusing its capacity.
                    self.bridge.drain()
                # The research bridge retains traces; the long-lived engine
                # emits bounded events instead of keeping every past plan.
                self.bridge.events.clear()
                return events
            except Exception:
                self.bridge.poisoned = True
                raise

    def drain(self, *, max_steps=100000):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        for _ in range(max_steps):
            if not self.active_requests:
                return
            yield from self.step()
        if self.active_requests:
            raise BridgeError('runtime exceeded bounded step limit')

    def close(self):
        with self._lock:
            if self.closed:
                return
            if self.bridge.poisoned:
                # No retries after a backend/native failure. Only drop state
                # after a successful fence; CUDA context faults remain explicit.
                self.bridge.backend.synchronize()
                for entry in list(self.bridge.requests.values()):
                    self.bridge.backend.release(entry['handle'], entry['state'])
                self.bridge.requests.clear()
                self.bridge.pending = None
                self.bridge.closed = True
            else:
                self.bridge.close()
            self.bridge.completed.clear()
            self.bridge.events.clear()
            self._requests.clear()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
