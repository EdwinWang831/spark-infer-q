"""Past-only budget/fallback adapter for the frozen collection trajectory helper."""
from collections import deque
import math
import time

from .confidence_features import feature_vector
from .confidence_head import choose_rows
from .verification_budget import LaggedAcceptance


class BudgetedTrajectoryDraft:
    """Choose a prefix after drafting; skip unprofitable drafts using past feedback.

    The next call observes the previous step's committed-prefix advance and its
    pending correction token. It never reads current/future target logits. Four
    initial serial steps measure the actual collector's single-row cost; slow
    drafted windows fall back to sixteen serial steps before trying again.
    """
    def __init__(self, draft, model, costs, cpu_predict_ms, conditional, *, clock=time.perf_counter):
        if len(conditional) != 7 or not math.isfinite(cpu_predict_ms) or cpu_predict_ms < 0:
            raise ValueError('seven-depth prior and finite inference cost required')
        for c in costs.values():
            if len(c['verify_commit_ms']) != 8 or any(not math.isfinite(v) or v <= 0 for v in c['verify_commit_ms']):
                raise ValueError('invalid collector cost curve')
        self.draft, self.model, self.costs = draft, model, costs
        self.cpu_ms, self.conditional, self.clock = cpu_predict_ms, conditional, clock

    def configure(self, task, *, calibrate=False):
        self.cap, self.thinking = task['output_cap'], task['thinking']
        self.early = 8 if calibrate else 0

    def begin(self, request):
        self.draft.begin(request)
        self.policy = LaggedAcceptance(self.conditional)
        self.prompt_rows = None
        self.last = None
        self.serial_ms = None
        self.warm = 4
        self.cooldown = 0
        self.skipped_since_draft = 0
        self.history = deque(maxlen=8)
        self.actual_draft_calls = 0
        self.skip_calls = 0
        self.runtime_fallbacks = 0

    def propose(self, features, anchor):
        now = self.clock()
        n = len(features)
        if self.prompt_rows is None:
            self.prompt_rows = n-self.early
        c = self.costs['long' if n >= 768 else 'short']
        if self.serial_ms is None:
            self.serial_ms = c['verify_commit_ms'][0]
        if self.last is not None:
            previous_n, started, proposals, verified = self.last
            advance = n-previous_n
            elapsed = max(.001, (now-started)*1000)
            assert 1 <= advance <= verified+1
            if proposals:
                if verified:
                    observed = min(advance, verified)
                    accepted = min(advance-1, verified)
                else:
                    # A width-one target step still reveals whether its next
                    # token matches the previously computed first draft.
                    observed, accepted = 1, int(anchor == proposals[0])
                self.policy.observe(observed, accepted)
                self.history.append((advance, elapsed))
            else:
                assert advance == 1
                self.serial_ms = .8*self.serial_ms+.2*elapsed
            if len(self.history) >= 4:
                tokens = sum(v[0] for v in self.history)
                ms = sum(v[1] for v in self.history)
                if tokens/ms <= .98/self.serial_ms:
                    self.cooldown = 16
                    self.history.clear()
                    self.runtime_fallbacks += 1
        remaining = self.cap-(n-self.prompt_rows+1)
        assert remaining >= 1
        limit = min(8, remaining)
        fee = c['native_draft_ms']+c['feature_increment_ms']+self.cpu_ms
        pre_costs = [v+(fee if index else 0.) for index, v in enumerate(c['verify_commit_ms'])]
        pre_width = choose_rows(self.policy.current, pre_costs, limit)
        skip = self.warm > 0 or self.cooldown > 0 or (pre_width == 1 and self.skipped_since_draft < 16)
        if skip:
            if self.warm:
                self.warm -= 1
            if self.cooldown:
                self.cooldown -= 1
            self.skip_calls += 1
            self.skipped_since_draft += 1
            self.last = (n, now, [], 0)
            return [], {}
        proposals, info = self.draft.propose(features, anchor)
        self.actual_draft_calls += 1
        self.skipped_since_draft = 0
        x = feature_vector(info['signals'], n, n-self.prompt_rows+1, remaining, self.thinking)
        rates = self.model.predict(x)
        width = choose_rows(rates, [v+fee for v in c['verify_commit_ms']], limit)
        self.last = (n, now, proposals, width-1)
        return proposals[:width-1], {}
