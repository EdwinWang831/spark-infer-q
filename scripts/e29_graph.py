"""Request-local target graphs with unchanged kernels and DFlash feature taps."""
import copy
import gc

from e3_wide_target import WideTarget


class FeatureGraphTarget(WideTarget):
    """Static state addresses; stage/commit stay eager and no graph is shared by requests."""
    def __init__(self, weights, capacity=2048):
        super().__init__(weights, capacity)
        self.bound = None
        self.graphs = {}
        self.outputs = {}
        self.pool = self.torch.cuda.graph_pool_handle()
        self.captures = 0

    def bind(self, start):
        if self.bound is None:
            self.bound = start.clone()
        else:
            self.bound.copy_from(start)
        return self.bound

    def compute_features(self, state, rows):
        torch, w, f = self.torch, self.w, self.forward
        b = self.buffers[rows]
        segs = [(state, 0, rows)]
        f._embed(w, b.ids[:rows], w.cfg.streams, b.h[:rows])
        pending, features = None, []
        for layer in w.layers:
            pending = f.layer_forward(layer, w, segs, b, rows, pending, context=self.capacity)
            if layer.index in (5, 15, 25, 35, 45):
                shadow = b.h[:rows].clone()
                proxy = copy.copy(b)
                proxy.pss = torch.empty_like(b.pss)
                f._writeback(shadow, proxy, rows, w.cfg, pending)
                features.append(shadow.unflatten(-1, (4, 2560)).mean(-2))
        return f.finish(w, w.mixer, b, rows, pending), torch.cat(features, -1)

    def run(self, st, ids, capture=True):
        if not capture:
            return super().run(st, ids, capture=False)
        if st is not self.bound:
            raise ValueError('graph target requires its bound static state')
        if not 1 <= len(ids) <= 8 or st.pos + len(ids) > self.capacity:
            raise ValueError('unsupported graph window')
        if len(set(st.cur)) != 1:
            raise ValueError('DeltaNet parities must agree')
        torch = self.torch
        rows = len(ids)
        b = self.buffers[rows]
        self.forward.stage(self.w, b, [(st, ids)])
        key = (rows, st.cur[0], self.capacity)
        if key not in self.graphs:
            self.compute_features(st, rows)  # shape-specific eager compilation
            torch.cuda.synchronize()
            gc.collect()
            enabled = gc.isenabled()
            gc.disable()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, pool=self.pool, capture_error_mode='thread_local'):
                    output = self.compute_features(st, rows)
            finally:
                if enabled:
                    gc.enable()
            torch.cuda.synchronize()
            self.graphs[key], self.outputs[key] = graph, output
            self.captures += 1
        self.graphs[key].replay()
        logits, features = self.outputs[key]
        logits = logits.clone()
        torch.cuda.synchronize()
        return logits, features, b

    def warm(self, start, widths=range(1, 9)):
        for parity in (0, 1):
            for width in widths:
                state = self.bind(start)
                state.cur = [parity] * len(state.cur)
                self.run(state, [0] * width)
        return self.bind(start)
