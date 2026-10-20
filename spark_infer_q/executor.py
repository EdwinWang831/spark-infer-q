"""Bounded target-only request execution, independent of tensor libraries.

Logical commit follows the backend fence. A canceled in-flight slot remains
leased until finish; stale handles and completions cannot target a reused slot.
This is a serialized prototype, not DeepInfer's production scheduler.
"""
from dataclasses import dataclass, field
from threading import RLock
from typing import Any
from uuid import uuid4


class ExecutionError(RuntimeError):
    pass


class StaleHandle(ExecutionError):
    pass


class Busy(ExecutionError):
    pass


class WorkClosed(ExecutionError):
    pass


class Pending(ExecutionError):
    pass


class BackendContractError(ExecutionError):
    pass


@dataclass(frozen=True)
class Handle:
    owner: str
    slot: int
    generation: int
    request_id: str


@dataclass(frozen=True)
class Row:
    handle: Handle | None = None
    position: int | None = None


@dataclass(frozen=True)
class Segment:
    handle: Handle
    indices: tuple[int, ...]
    base_position: int
    revision: int
    layout_version: int


@dataclass(frozen=True)
class Ticket:
    owner: str
    serial: int
    mode: str
    row_count: int
    segments: tuple[Segment, ...]


@dataclass(frozen=True)
class Checkpoint:
    owner: str
    serial: int
    handle: Handle
    position: int


@dataclass(frozen=True)
class Completion:
    output: Any
    row_status: tuple[str, ...]
    committed: tuple[Handle, ...]


@dataclass
class _Slot:
    generation: int = 0
    handle: Handle | None = None
    state: Any = None
    position: int = 0
    revision: int = 0
    lease: int | None = None
    canceled: bool = False


@dataclass
class _Work:
    ticket: Ticket
    launched: bool = False
    fence: Any = None
    output: Any = None
    backups: list[Any] = field(default_factory=list)


class Executor:
    """Backend must supply create/ensure/layout/position/checkpoint/restore,
    validate_inputs/launch/mask_rows/synchronize/release/export_state.

    launch(ticket, states, inputs) returns (output, fence), computes all valid
    rows and advances each private state by exactly that segment's row count.
    Snapshots must independently preserve the prefix, including KV rows.
    """

    def __init__(self, backend, max_requests=2, max_checkpoints=4):
        if max_requests < 1 or max_checkpoints < 0:
            raise ValueError('invalid pool size')
        self.backend = backend
        self.owner = uuid4().hex
        self._slots = [_Slot() for _ in range(max_requests)]
        self._requests = {}
        self._work = {}
        self._checkpoints = {}
        self._serial = 0
        self._active = None
        self._max_checkpoints = max_checkpoints
        self._poisoned = False
        self._lock = RLock()

    def _healthy(self):
        if self._poisoned:
            raise BackendContractError('backend failure left executor quarantined')

    def _slot(self, handle, canceled=False):
        if not isinstance(handle, Handle) or handle.owner != self.owner or not 0 <= handle.slot < len(self._slots):
            raise StaleHandle('foreign or invalid handle')
        slot = self._slots[handle.slot]
        if slot.handle != handle or (slot.canceled and not canceled):
            raise StaleHandle('request canceled, released or replaced')
        return slot

    def _get_work(self, ticket):
        if not isinstance(ticket, Ticket) or ticket.owner != self.owner:
            raise WorkClosed('foreign ticket')
        work = self._work.get(ticket.serial)
        if work is None or work.ticket != ticket:
            raise WorkClosed('ticket expired or modified')
        return work

    def _release(self, slot):
        handle = slot.handle
        self.backend.release(slot.state)
        if self._requests.get(handle.request_id) == handle:
            self._requests.pop(handle.request_id)
        for serial, (token, _) in list(self._checkpoints.items()):
            if token.handle == handle:
                self._checkpoints.pop(serial)
        slot.handle, slot.state = None, None
        slot.position, slot.revision, slot.lease, slot.canceled = 0, 0, None, False

    def admit(self, request_id):
        with self._lock:
            self._healthy()
            if not isinstance(request_id, str) or not request_id:
                raise ValueError('request_id must be a nonempty string')
            if request_id in self._requests:
                raise Busy('request_id already active')
            for index, slot in enumerate(self._slots):
                if slot.handle is None:
                    state = self.backend.create()
                    slot.generation += 1
                    handle = Handle(self.owner, index, slot.generation, request_id)
                    slot.handle, slot.state = handle, state
                    self._requests[request_id] = handle
                    return handle
            raise Busy('all request slots active or awaiting GPU completion')

    def view(self, handle):
        with self._lock:
            slot = self._slot(handle)
            return {'position': slot.position, 'revision': slot.revision,
                    'layout_version': self.backend.layout(slot.state),
                    'busy': slot.lease is not None}

    def prepare(self, rows, mode='decode'):
        with self._lock:
            self._healthy()
            if mode not in ('prefill', 'decode') or not rows:
                raise ValueError('nonempty rows and prefill/decode mode required')
            groups = {}
            for index, row in enumerate(rows):
                if not isinstance(row, Row):
                    raise ValueError('expected Row metadata')
                if row.handle is None:
                    if row.position is not None:
                        raise ValueError('padding has no position')
                    continue
                slot = self._slot(row.handle)
                if slot.lease is not None:
                    raise Busy('request already reserved')
                indices = groups.setdefault(row.handle, [])
                if type(row.position) is not int or row.position != slot.position + len(indices):
                    raise ValueError('rows must continue the committed prefix in request order')
                indices.append(index)
            for handle, indices in groups.items():
                slot = self._slot(handle)
                self.backend.ensure(slot.state, slot.position + len(indices))
            self._serial += 1
            segments = tuple(Segment(h, tuple(indices), self._slot(h).position,
                                     self._slot(h).revision, self.backend.layout(self._slot(h).state))
                             for h, indices in groups.items())
            ticket = Ticket(self.owner, self._serial, mode, len(rows), segments)
            for seg in segments:
                self._slot(seg.handle).lease = ticket.serial
            self._work[ticket.serial] = _Work(ticket)
            return ticket

    def _close(self, work):
        for seg in work.ticket.segments:
            slot = self._slot(seg.handle, canceled=True)
            slot.lease = None
            if slot.canceled:
                self._release(slot)
        self._work.pop(work.ticket.serial)
        if self._active == work.ticket.serial:
            self._active = None

    def discard(self, ticket):
        with self._lock:
            work = self._get_work(ticket)
            if work.launched:
                raise Busy('dispatched work must drain through finish')
            self._close(work)

    def cancel(self, handle):
        with self._lock:
            self._healthy()
            slot = self._slot(handle)
            slot.canceled = True
            if self._requests.get(handle.request_id) == handle:
                self._requests.pop(handle.request_id)
            if slot.lease is None:
                self._release(slot)
            else:
                work = self._work[slot.lease]
                if not work.launched:
                    self._close(work)  # no GPU references; other reservations become ready again

    def _recover(self, work):
        try:
            self.backend.synchronize()
            for seg, backup in zip(work.ticket.segments, work.backups):
                self.backend.restore(self._slot(seg.handle, canceled=True).state, backup)
            self.backend.synchronize()
        except Exception:
            self._poisoned = True
            raise BackendContractError('rollback failed; slots remain quarantined')
        self._close(work)

    def dispatch(self, ticket, inputs):
        with self._lock:
            self._healthy()
            work = self._get_work(ticket)
            if work.launched or self._active is not None:
                raise Busy('one GPU batch may run at a time')
            states = []
            for seg in ticket.segments:
                slot = self._slot(seg.handle)
                if (slot.lease != ticket.serial or slot.revision != seg.revision
                        or slot.position != seg.base_position
                        or self.backend.layout(slot.state) != seg.layout_version):
                    raise StaleHandle('request or storage changed since prepare')
                states.append(slot.state)
            self.backend.validate_inputs(ticket, inputs)
            work.backups = [self.backend.checkpoint(st) for st in states]
            work.launched, self._active = True, ticket.serial
            try:
                work.output, work.fence = self.backend.launch(ticket, states, inputs)
            except Exception:
                self._recover(work)
                raise

    def finish(self, ticket, wait=True):
        with self._lock:
            self._healthy()
            work = self._get_work(ticket)
            if not work.launched:
                raise Pending('work has not been dispatched')
            try:
                if not work.fence.ready():
                    if not wait:
                        raise Pending('GPU work has not completed')
                    work.fence.wait()
                    if not work.fence.ready():
                        raise Pending('backend fence is still pending')
            except Pending:
                raise
            except Exception:
                self._poisoned = True
                raise BackendContractError('completion fence failed; slots quarantined')
            live, statuses, valid = [], ['PADDING'] * ticket.row_count, []
            try:
                for seg in ticket.segments:
                    slot = self._slot(seg.handle, canceled=True)
                    if self.backend.position(slot.state) != seg.base_position + len(seg.indices):
                        raise BackendContractError('backend advanced an unexpected number of rows')
                    for row in seg.indices:
                        statuses[row] = 'CANCELED' if slot.canceled else 'COMMITTED'
                    if not slot.canceled:
                        live.append(seg.handle)
                        valid.extend(seg.indices)
                output = self.backend.mask_rows(work.output, valid)
            except Exception:
                self._recover(work)
                raise
            for seg in ticket.segments:
                slot = self._slot(seg.handle, canceled=True)
                if not slot.canceled:
                    slot.position += len(seg.indices)
                    slot.revision += 1
            self._close(work)
            return Completion(output, tuple(statuses), tuple(live))

    def checkpoint(self, handle):
        with self._lock:
            self._healthy()
            slot = self._slot(handle)
            if slot.lease is not None:
                raise Busy('cannot snapshot reserved state')
            if len(self._checkpoints) >= self._max_checkpoints:
                raise Busy('checkpoint budget exhausted')
            self._serial += 1
            token = Checkpoint(self.owner, self._serial, handle, slot.position)
            self._checkpoints[token.serial] = token, self.backend.checkpoint(slot.state)
            return token

    def restore(self, handle, token):
        with self._lock:
            self._healthy()
            slot = self._slot(handle)
            saved = self._checkpoints.get(token.serial) if isinstance(token, Checkpoint) else None
            if saved is None or saved[0] != token or token.handle != handle or token.owner != self.owner:
                raise StaleHandle('checkpoint belongs to an expired or different request')
            if slot.lease is not None:
                raise Busy('cannot restore reserved state')
            try:
                self.backend.restore(slot.state, saved[1])
            except Exception:
                self._poisoned = True
                raise BackendContractError('restore failed; executor quarantined')
            slot.position, slot.revision = token.position, slot.revision + 1

    def drop_checkpoint(self, token):
        with self._lock:
            saved = self._checkpoints.get(token.serial) if isinstance(token, Checkpoint) else None
            if saved is None or saved[0] != token:
                raise StaleHandle('checkpoint expired')
            self._checkpoints.pop(token.serial)

    def debug_state(self, handle):
        with self._lock:
            self._healthy()
            slot = self._slot(handle)
            if slot.lease is not None:
                raise Busy('cannot inspect unpublished state')
            return self.backend.export_state(slot.state)
