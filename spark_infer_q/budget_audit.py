"""Cost diagnostics; hindsight paths are never an online scheduling policy."""
import math


MODES = ('serial', 'fixed1', 'fixed2', 'fixed3', 'dynamic')


def mode_order(repetition, task_index=0):
    offset = (repetition+task_index) % len(MODES)
    return MODES[offset:]+MODES[:offset]


def candidate_scores(cumulative_rates, costs_ms):
    """Startup-curve throughput, excluding unobserved runtime slope corrections."""
    if len(costs_ms) != len(cumulative_rates)+1:
        raise ValueError('one cost per anchor-plus-prefix width required')
    previous = 1.
    for rate in cumulative_rates:
        if not math.isfinite(rate) or not 0 <= rate <= previous:
            raise ValueError('rates must be cumulative probabilities')
        previous = rate
    if any(not math.isfinite(c) or c <= 0 for c in costs_ms):
        raise ValueError('costs must be finite and positive')
    expected = 1.
    result = []
    for index, cost in enumerate(costs_ms):
        if index: expected += cumulative_rates[index-1]
        result.append({'width': index+1, 'expected_tokens': expected,
                       'cost_ms': cost, 'tokens_per_ms': expected/cost})
    return result


def hindsight_paths(positions):
    """Shortest path and fixed-width paths over measured, EOS-aware actions.

    Each position describes the same serial prefix, with all four possible
    widths. Costs are independently replayed medians, not end-to-end timings.
    The result uses future acceptance/EOS and cannot be deployed online.
    """
    count = len(positions)
    for index, actions in enumerate(positions):
        if sorted(a['width'] for a in actions) != [1, 2, 3, 4]:
            raise ValueError('every position needs widths 1..4')
        for action in actions:
            emitted, cost = action['emitted_tokens'], action['total_ms']
            if not isinstance(emitted, int) or not 1 <= emitted <= min(action['width'], count-index):
                raise ValueError('invalid action advance')
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError('invalid action cost')
    cost_to_go = [0.]*(count+1)
    chosen = [None]*count
    for index in range(count-1, -1, -1):
        action = min(positions[index], key=lambda a: (a['total_ms']+cost_to_go[index+a['emitted_tokens']], a['width']))
        cost_to_go[index] = action['total_ms']+cost_to_go[index+action['emitted_tokens']]
        chosen[index] = action
    def walk(width=None):
        index, total, path = 0, 0., []
        while index < count:
            action = chosen[index] if width is None else next(a for a in positions[index] if a['width'] == width)
            path.append({'output_offset': index, 'width': action['width'], 'emitted_tokens': action['emitted_tokens']})
            total += action['total_ms']
            index += action['emitted_tokens']
        return {'replayed_cost_ms': total, 'path': path}
    fixed = {str(width): walk(width) for width in range(1, 5)}
    best_width = min(fixed, key=lambda k: fixed[k]['replayed_cost_ms'])
    oracle = walk()
    return {'hindsight': oracle, 'fixed': fixed, 'best_fixed_width': int(best_width),
            'best_fixed_over_hindsight': fixed[best_width]['replayed_cost_ms']/oracle['replayed_cost_ms'] if count else None,
            'scope': 'future-aware replay cost model; not an observed online speedup'}
