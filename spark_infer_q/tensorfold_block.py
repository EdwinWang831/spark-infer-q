"""Experimental adapter for one externally loaded TensorFold decoder block.

External sources/weights are supplied by the isolated harness, not vendored.
Decode graphs keep commit outside capture and bind to state storage/parity.
"""
from collections import OrderedDict
from dataclasses import dataclass

import torch


class _Fence:
    def __init__(self, stream):
        self.event = torch.cuda.Event()
        self.event.record(stream)

    def ready(self):
        return self.event.query()

    def wait(self):
        self.event.synchronize()


@dataclass
class _Graph:
    graph: object
    buffers: object
    states: tuple
    tensors: tuple


class TensorFoldBlockBackend:
    def __init__(self, weights, layer, *, graphs=True, capacity=8, limit=64, max_rows=32, graph_limit=8):
        if not 1 <= capacity <= limit or max_rows < 1 or graph_limit < 1:
            raise ValueError('invalid state or graph-cache limits')
        from tensorfold.families.qwen4_exp.cuda import forward
        from tensorfold.families.qwen4_exp.cuda.state import Buffers, State
        self.w, self.layer = weights, layer
        self.forward, self.Buffers, self.State = forward, Buffers, State
        self.graphs, self.capacity, self.limit = graphs, capacity, limit
        self.max_rows, self.graph_limit = max_rows, graph_limit
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self._graphs = OrderedDict()
        self.stats = {'graph_captures': 0, 'graph_hits': 0, 'graph_invalidations': 0,
                      'graph_evictions': 0, 'eager_batches': 0, 'model_calls': 0}

    def create(self):
        with torch.cuda.stream(self.stream):
            return self.State(self.w, self.capacity, self.max_rows, 'bf16', limit=self.limit)

    def invalidate(self, state):
        for key, entry in list(self._graphs.items()):
            if any(st is state for st in entry.states):
                self._graphs.pop(key)
                self.stats['graph_invalidations'] += 1

    def ensure(self, state, rows):
        if rows > state.limit:
            raise ValueError('request context exceeds configured limit')
        if rows > state.capacity:
            self.invalidate(state)  # release old pointer bindings before resize drops the old tensors
            with torch.cuda.stream(self.stream):
                state.ensure(rows, step=8)

    def layout(self, state):
        return state.version

    def position(self, state):
        return state.pos

    def checkpoint(self, state):
        with torch.cuda.stream(self.stream):
            return state.clone()  # independent KV prefix plus recurrent/conv state

    def restore(self, state, saved):
        self.invalidate(state)
        with torch.cuda.stream(self.stream):
            state.copy_from(saved)

    def release(self, state):
        self.invalidate(state)

    def validate_inputs(self, ticket, inputs):
        width = self.w.cfg.hidden * self.w.cfg.streams
        if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 2
                or tuple(inputs.shape) != (ticket.row_count, width) or inputs.dtype != torch.bfloat16
                or inputs.device != self.w.device):
            raise ValueError('inputs must be device-local BF16 [metadata rows, four-stream width]')
        if sum(len(s.indices) for s in ticket.segments) > 64 or any(len(s.indices) > self.max_rows for s in ticket.segments):
            raise ValueError('batch or per-request window exceeds probe limit')
        indices = [i for s in ticket.segments for i in s.indices]
        if indices:
            selected = inputs.index_select(0, torch.tensor(indices, device=inputs.device))
            if not bool(torch.isfinite(selected).all()):
                raise ValueError('valid hidden-state rows must be finite')

    @staticmethod
    def _tensor_refs(state):
        refs = []
        def visit(value):
            if isinstance(value, torch.Tensor):
                refs.append(value)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    visit(v)
            elif hasattr(value, '__dict__'):
                for v in vars(value).values():
                    visit(v)
        for value in vars(state).values():
            visit(value)
        return refs

    def _key(self, ticket, states):
        return tuple((s.handle, len(s.indices), st.version, st.capacity, tuple(st.cur), id(st))
                     for s, st in zip(ticket.segments, states))

    def launch(self, ticket, states, inputs):
        self.stream.wait_stream(torch.cuda.current_stream())
        inputs.record_stream(self.stream)  # caller may drop its input after asynchronous dispatch
        indices = [i for seg in ticket.segments for i in seg.indices]
        with torch.cuda.stream(self.stream):
            output = torch.zeros((ticket.row_count, inputs.shape[1]), dtype=inputs.dtype, device=self.w.device)
            if not indices:
                return output, _Fence(self.stream)
            index = torch.tensor(indices, dtype=torch.long, device=inputs.device)
            packed = inputs.index_select(0, index).to(self.w.device).contiguous()
            offsets, at = [], 0
            for st, seg in zip(states, ticket.segments):
                offsets.append((st, at, at + len(seg.indices)))
                at += len(seg.indices)
            rows, context = len(indices), max(st.capacity for st in states)

            def compute(buf):
                pending = self.forward.layer_forward(self.layer, self.w, offsets, buf, rows, None, context=context)
                self.forward._writeback(buf.h, buf, rows, self.w.cfg, pending)

            graph_mode = self.graphs and ticket.mode == 'decode'
            if graph_mode:
                key = self._key(ticket, states)
                entry = self._graphs.get(key)
                if entry is None:
                    buf = self.Buffers(self.w, rows, context)
                    for _ in range(2):
                        buf.h.copy_(packed)
                        compute(buf)  # no commit: only proposed state and KV suffix are written
                    self.stream.synchronize()
                    buf.h.copy_(packed)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self.stream):
                        compute(buf)
                    refs = tuple(t for st in states for t in self._tensor_refs(st))
                    entry = _Graph(graph, buf, tuple(states), refs)
                    self._graphs[key] = entry
                    self.stats['graph_captures'] += 1
                    while len(self._graphs) > self.graph_limit:
                        self._graphs.popitem(last=False)
                        self.stats['graph_evictions'] += 1
                else:
                    self._graphs.move_to_end(key)
                    self.stats['graph_hits'] += 1
                buf = entry.buffers
                buf.h.copy_(packed)
                entry.graph.replay()
            else:
                buf = self.Buffers(self.w, rows, context, prefill=ticket.mode == 'prefill')
                buf.h.copy_(packed)
                compute(buf)
                self.stats['eager_batches'] += 1
            self.stats['model_calls'] += 1
            output.index_copy_(0, index.to(output.device), buf.h)
            for seg, st, (_, offset, _) in zip(ticket.segments, states, offsets):
                n = len(seg.indices)
                self.forward.commit(self.w, st, buf, n, n, at=offset)
            return output, _Fence(self.stream)

    def mask_rows(self, output, valid):
        with torch.cuda.stream(self.stream):
            masked = torch.zeros_like(output)
            if valid:
                idx = torch.tensor(valid, device=output.device)
                masked.index_copy_(0, idx, output.index_select(0, idx))
        consumer = torch.cuda.current_stream()
        consumer.wait_stream(self.stream)
        masked.record_stream(consumer)
        return masked

    def synchronize(self):
        self.stream.synchronize()

    def export_state(self, state):
        self.synchronize()
        out = {'position': state.pos, 'layout_version': state.version, 'capacity': state.capacity,
               'conv': state.conv.cpu().clone()}
        if self.layer.linear:
            out['rec'] = state.rec[state.cur[0], 0].cpu().clone()
        else:
            out.update(k=state.kc[0].k[:state.pos].cpu().clone(), v=state.kc[0].v[:state.pos].cpu().clone(),
                       ik=state.ikc[0][:state.pos].cpu().clone(), pool=state.pooled[0][:state.pos//4].cpu().clone())
        return out
