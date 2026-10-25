"""Bounded synchronous adapter around the real DeepInfer C++ Scheduler.

Native blocks are admission/accounting handles, not Qwen device KV pages.
The injected backend owns device state and must finish its fence before return.
No native Worker, GPU sampler, speculation, forks or async pipeline is implied.
"""
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4
import os
import time


class BridgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class RequestHandle:
    owner: str
    slot: int
    generation: int
    request_id: str


@dataclass(frozen=True)
class WorkRow:
    handle: RequestHandle
    position: int
    token_ids: tuple[int, ...]
    prompt: bool
    last_prompt_chunk: bool


@dataclass(eq=False)
class Plan:
    owner: str
    serial: int
    rows: tuple[WorkRow, ...]
    metadata: object
    native_output: object
    executed: bool = False
    predictions: tuple[int, ...] = ()


class SchedulerBridge:
    def __init__(self, core, torch, group, sampling_class, backend, *, vocab_size,
                 eos_ids, capacity=128, max_requests=2, token_budget=32,
                 max_drafts=0,prefill_interleave=False,output_cache=False):
        if type(max_drafts) is not int or not 0 <= max_drafts <= 7:
            raise ValueError('max_drafts must be 0..7')
        if os.environ.get('DEEPINFER_USE_NEXTN_DECODE') != str(max_drafts):
            raise BridgeError('native NEXTN setting does not match bridge')
        self.max_drafts = max_drafts
        if (any(type(v) is not int for v in (capacity, max_requests, token_budget)) or
                not (capacity >= 16 and max_requests >= 1 and token_budget >= 16)):
            raise ValueError('unsupported scheduler capacity')
        if type(prefill_interleave) is not bool:
            raise ValueError('prefill_interleave must be boolean')
        if type(output_cache) is not bool:
            raise ValueError('output_cache must be boolean')
        self.output_cache=output_cache
        if prefill_interleave and (max_requests!=2 or group.size()!=1 or max_drafts!=0):
            raise ValueError('prefill interleave requires two single-rank target-only slots')
        self.prefill_interleave=prefill_interleave
        self._last_prompt=False
        self.interleave_hints=dict(decode_after_prompt=0,prompt_after_decode=0,prompt_without_ready_decode=0)
        self.core, self.torch, self.sampling_class, self.backend = core, torch, sampling_class, backend
        self.vocab_size, self.eos_ids = vocab_size, frozenset(eos_ids)
        if not self.eos_ids or any(not 0 <= t < vocab_size for t in self.eos_ids):
            raise ValueError('invalid EOS set')
        self.capacity, self.max_requests = capacity, max_requests
        self.owner, self.serial, self.sequence_id = uuid4().hex, 0, 0
        self.generations = [0]*max_requests
        self.slots = [None]*max_requests
        self.requests, self.completed, self.events = {}, {}, []
        self.pending, self.poisoned, self.closed = None, False, False
        # These block IDs account for private Qwen state; they do not allocate
        # device KV. Native prefill reserves a prompt's complete block table on
        # its first chunk, then refuses further chunks above its 90% watermark.
        # Leave host accounting headroom even when every admitted slot is full,
        # or a valid near-capacity prompt can wait forever after its first chunk.
        accounted_blocks = 1 + max_requests*((capacity+15)//16+2)
        self.block_count = (10*accounted_blocks+8)//9
        self.blocks = core.BlockManager(self.block_count)
        self.pool = core.SequenceMetadataPool()
        # With one rank and one admitted slot there cannot be a second waiting
        # prompt during decode. Polling Gloo for that impossible work every step
        # only adds host latency. Native prefill resets decode_step to zero;
        # restarting this polling phase at admission also makes reuse immediate.
        # This counter is separate from native MTP schedule_cnt/process_cnt.
        self.single_slot = max_requests == 1 and group.size() == 1
        prefill_interval = 2 if prefill_interleave else capacity+1 if self.single_slot else 1
        config = SimpleNamespace(max_model_len=capacity, runtime_max_model_len=capacity+2*max_drafts,
            max_num_batched_tokens=token_budget, max_num_seqs=max_requests,
            stop_preempted_request=True, chunked_prefill=True, engine_type=0,
            prefill_attempt_interval=prefill_interval, prefill_any_attempt_interval=prefill_interval)
        self.native = core.Scheduler(16, self.blocks, None, min(self.eos_ids), group,
            config, 0, 0, capacity*max_requests, None, [], 1, vocab_size, False)

    def _ready(self):
        if self.closed or self.poisoned:
            raise BridgeError('bridge is closed or quarantined')

    def _entry(self, handle):
        if not isinstance(handle, RequestHandle) or handle.owner != self.owner:
            raise BridgeError('foreign handle')
        entry = self.requests.get(handle.request_id)
        if entry is None or entry['handle'] != handle:
            raise BridgeError('stale handle')
        return entry

    def _terminal(self, seq):
        return seq.status not in (self.core.SequenceStatus.WAITING,
            self.core.SequenceStatus.RUNNING, self.core.SequenceStatus.SWAPPED)

    def _prefill_phase(self):
        if not self.prefill_interleave:return
        live=list(self.requests.values())
        waiting=any(e['seq'].status==self.core.SequenceStatus.WAITING for e in live)
        ready=any(e['seq'].status==self.core.SequenceStatus.RUNNING and
                  e['seq'].num_cached_tokens>=e['prompt_length'] for e in live)
        if waiting:
            # Single-rank NEXTN=0 only: choose a supported native polling phase,
            # not a request/token plan. Native admission/queues still select work.
            # An interval of2 makes phase1 a decode-only opportunity and phase0
            # a prompt polling point. Without runnable decode, keep prefill moving.
            yield_decode=ready and self._last_prompt
            self.native.decode_step=1 if yield_decode else 0
            key='decode_after_prompt' if yield_decode else 'prompt_after_decode' if ready else 'prompt_without_ready_decode'
            self.interleave_hints[key]+=1

    def add(self, request_id, prompt_ids, max_tokens):
        import numpy as np
        self._ready()
        ids = tuple(prompt_ids)
        if not isinstance(request_id, str) or not request_id or request_id in self.requests:
            raise BridgeError('empty or duplicate request ID')
        if (not ids or any(type(t) is not int or not 0 <= t < self.vocab_size for t in ids)
                or type(max_tokens) is not int or max_tokens < 1 or len(ids)+max_tokens > self.capacity):
            raise ValueError('invalid token IDs or context/output budget')
        if None not in self.slots:
            raise BridgeError('request slots full')
        slot = self.slots.index(None); self.generations[slot] += 1
        handle = RequestHandle(self.owner, slot, self.generations[slot], request_id)
        self.sequence_id += 1
        params = self.sampling_class(temperature=0, max_tokens=max_tokens,
            ignore_eos=False, eos_id=min(self.eos_ids), seq_seed=17)
        prompt = np.asarray(ids, dtype=np.int32)
        seq = self.core.Sequence(self.sequence_id, prompt, prompt, None, 16,
            request_id, 'e5', params, time.time()*1000, None)
        state = self.backend.create(handle)
        try:
            self.native.add_sequence(seq)
            if self.single_slot:
                self.native.decode_step = 0
        except Exception:
            # Admission may have partly changed native accounting. Never reuse
            # this scheduler after a failed native call, even before dispatch.
            self.poisoned = True
            self.backend.release(handle, state)
            raise
        self.requests[request_id] = dict(handle=handle, seq=seq, state=state,
            cancelled=False, prompt_length=len(ids),published=[])
        self.slots[slot] = handle
        return handle

    def cancel(self, handle):
        self._ready(); entry = self._entry(handle)
        if self._terminal(entry['seq']):
            raise BridgeError('request already terminal')
        entry['cancelled'] = True
        self.native.abort_sequence([handle.request_id])
        # A prepared prompt is no longer in native waiting/running queues.
        entry['seq'].status = self.core.SequenceStatus.FINISHED_ABORTED

    def _retire(self):
        leased = {r.handle for r in self.pending.rows} if self.pending else set()
        for request_id, entry in list(self.requests.items()):
            if not self._terminal(entry['seq']) or entry['handle'] in leased:
                continue
            seq = entry['seq']; handle = entry['handle']
            tokens=seq.data.get_token_ids()[entry['prompt_length']:]
            if self.output_cache and list(tokens)!=entry['published']:
                self.poisoned=True
                raise BridgeError('cached native publication differs at retirement')
            self.completed[handle] = dict(token_ids=tokens,
                processed_position=seq.num_cached_tokens, status=str(seq.status),
                cancelled=entry['cancelled'])
            self.native.free_seq(seq)
            self.backend.release(handle, entry['state'])
            self.slots[handle.slot] = None
            del self.requests[request_id]

    def _cache_published(self,entry,tokens):
        if not self.output_cache:return
        seq=entry['seq'];data=seq.data
        if (data.get_len()-entry['prompt_length']!=len(entry['published'])+len(tokens) or
                (tokens and data.get_last_token_id()!=tokens[-1])):
            raise BridgeError('native publication length/tail differs from cache')
        if tokens and tuple(seq.last_accepted_token)+(int(seq.last_output_token),)!=tuple(tokens):
            raise BridgeError('native accepted/output tokens differ from cache')
        # Called only after native process_outputs and its count/position gates.
        # Read the native bounded last-step fields, not backend guesses.
        # The complete native prefix is independently checked at retirement.
        entry['published'].extend(tokens)

    def output_tokens(self,handle):
        entry=self._entry(handle)
        return tuple(entry['published']) if self.output_cache else tuple(entry['seq'].data.get_token_ids()[entry['prompt_length']:])

    def prepare(self):
        self._ready()
        if self.pending is not None:
            raise BridgeError('previous plan not completed')
        self._retire()
        self._prefill_phase()
        metadata, output = self.native.schedule(self.pool)
        if any(getattr(output, name).size for name in ('clone_states','clone_blocks','zero_states')):
            self.poisoned = True
            raise BridgeError('unsupported native cache clone/zero operation')
        rows = []
        for meta, seq in zip(metadata, output.scheduled_seqs, strict=True):
            entry = self.requests[meta.request_id]
            start = seq.num_cached_tokens
            row = self._work_row(meta, seq, entry)
            if self.backend.position(entry['state']) != start:
                self.poisoned = True
                raise BridgeError('native/device committed position mismatch')
            rows.append(row)
        self.serial += 1
        plan = Plan(self.owner, self.serial, tuple(rows), metadata, output)
        self.pending = plan
        return plan

    def _work_row(self, meta, seq, entry):
        start = seq.num_cached_tokens
        ids = ((seq.data.get_last_token_id(),) if self.output_cache and not meta.is_prompt else
               tuple(seq.data.get_token_ids()[start:start+meta.seqlen_q]))
        if (len(ids) != meta.seqlen_q or meta.seqlen_k != start+len(ids)
                or (not meta.is_prompt and len(ids) != 1)):
            self.poisoned = True
            raise BridgeError('native token/position metadata mismatch')
        return WorkRow(entry['handle'], start, ids, meta.is_prompt, meta.is_last_prompt_chunk)

    def _plan(self, plan):
        self._ready()
        if plan is not self.pending or plan.owner != self.owner:
            raise BridgeError('stale or foreign plan')

    def execute(self, plan):
        self._plan(plan)
        if plan.executed:
            raise BridgeError('plan already executed')
        try:
            predictions = []
            for row in plan.rows:
                entry = self._entry(row.handle)
                if entry['cancelled']:
                    predictions.append(0)
                else:
                    token = self.backend.advance(row, entry['state'])
                    if type(token) is not int or not 0 <= token < self.vocab_size:
                        raise BridgeError('invalid backend token')
                    if self.backend.position(entry['state']) != row.position+len(row.token_ids):
                        raise BridgeError('backend failed to advance exactly the planned rows')
                    predictions.append(token)
            plan.predictions, plan.executed = tuple(predictions), True
        except Exception:
            self.poisoned = True
            raise

    def complete(self, plan):
        self._plan(plan)
        if not plan.executed:
            raise BridgeError('backend fence not completed')
        try:
            accepted = self.torch.empty(0, dtype=self.torch.int32, device='cpu')
            tokens = self.torch.tensor(plan.predictions, dtype=self.torch.int32, device='cpu')
            _, pushed = self.native.process_outputs(plan.native_output, accepted, tokens, None)
            pushed = list(map(int, pushed))
            for row, token, count in zip(plan.rows, plan.predictions, pushed, strict=True):
                entry = self._entry(row.handle); seq = entry['seq']
                if entry['cancelled']:
                    if count != 0:
                        raise BridgeError('cancelled request published output')
                else:
                    if seq.num_cached_tokens != self.backend.position(entry['state']):
                        raise BridgeError('native/device position diverged at completion')
                    if count and token in self.eos_ids:
                        seq.status = self.core.SequenceStatus.FINISHED_STOPPED
                if self.output_cache and count not in (0,1):
                    raise BridgeError('target-only native publication count differs')
                self._cache_published(entry,() if not count else (token,))
                self.events.append(dict(slot=row.handle.slot, generation=row.handle.generation,
                    request_id=row.handle.request_id, prompt=row.prompt,
                    start=row.position, rows=len(row.token_ids), published=count,
                    cancelled=entry['cancelled']))
            self.pool.release(plan.metadata)
            self.pending = None
            self._last_prompt=any(row.prompt for row in plan.rows)
            self._retire()
            return pushed
        except Exception:
            self.poisoned = True
            raise

    def step(self):
        plan = self.prepare(); self.execute(plan); self.complete(plan)
        return plan

    def drain(self, max_steps=2048):
        for _ in range(max_steps):
            if not self.requests and not self.native.has_unfinished_seqs():
                return
            self.step()
        raise BridgeError('scheduler made no bounded progress')

    def close(self):
        self._ready()
        if self.pending is not None:
            raise BridgeError('cannot close with an unfinished plan')
        for entry in list(self.requests.values()):
            if not self._terminal(entry['seq']):
                self.cancel(entry['handle'])
        self.drain()
        if self.blocks.get_num_free_blocks() != self.block_count-1:
            raise BridgeError('native logical blocks leaked')
        self.closed = True
