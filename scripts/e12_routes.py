"""Separate, untimed route capture over the E8 target arithmetic."""
import copy

from e3_wide_target import WideTarget


class ObservedTarget(WideTarget):
    """Normal run() is inherited unchanged; observation uses a separate method."""

    def __init__(self, weights, capacity=1024):
        super().__init__(weights, capacity=capacity)
        self.route_storage = {
            rows: self.torch.empty((len(weights.layers), rows, 10), dtype=self.torch.int32,
                                   device=weights.device)
            for rows in self.buffers}

    def observe(self, state, ids):
        torch, w, f = self.torch, self.w, self.forward
        rows = len(ids)
        b, routes = self.buffers[rows], self.route_storage[rows]
        segments = f.stage(w, b, [(state, ids)])
        f._embed(w, b.ids[:rows], w.cfg.streams, b.h[:rows])
        pending, features = None, []
        for index, layer in enumerate(w.layers):
            pending = f.layer_forward(layer, w, segments, b, rows, pending)
            # pick has an additional shared-expert slot, excluded from Top-10.
            routes[index].copy_(b.moe.pick[:rows, :10])
            if layer.index in (5, 15, 25, 35, 45):
                shadow = b.h[:rows].clone()
                proxy = copy.copy(b)
                proxy.pss = torch.empty_like(b.pss)
                f._writeback(shadow, proxy, rows, w.cfg, pending)
                features.append(shadow.unflatten(-1, (4, 2560)).mean(-2))
        logits = f.finish(w, w.mixer, b, rows, pending).clone()
        features = torch.cat(features, -1)
        torch.cuda.synchronize()
        # Only one D2H route copy, after the entire forward, outside timing.
        by_layer = routes.cpu().tolist()
        assert all(len(set(pick)) == 10 and all(0 <= e < 512 for e in pick)
                   for layer in by_layer for pick in layer)
        by_row = [[by_layer[layer][row] for layer in range(len(w.layers))]
                  for row in range(rows)]
        return logits, features, b, by_row
