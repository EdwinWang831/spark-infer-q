"""Lagged conditional-acceptance estimates and balanced benchmark order."""


class LaggedAcceptance:
    def __init__(self, conditional=(.5,.5,.5), strength=4.):
        if strength <= 0 or not conditional or any(not 0 <= p <= 1 for p in conditional):
            raise ValueError('invalid acceptance prior')
        self.success=[p*strength for p in conditional]
        self.trials=[strength]*len(conditional)
        self.current=self._rates()
        self.previous=self.current.copy()

    def _rates(self):
        product=1.
        rates=[]
        for success,trials in zip(self.success,self.trials):
            product *= success/trials
            rates.append(product)
        return rates

    def observe(self, verified, accepted):
        if not 0 <= accepted <= verified <= len(self.success):
            raise ValueError('invalid accepted prefix')
        self.previous=self.current.copy()
        for j in range(verified):
            if j > accepted: break  # later drafts were censored by an earlier rejection
            self.trials[j]+=1
            self.success[j]+=int(j<accepted)
        self.current=self._rates()


def balanced_order(repetition,task_index=0):
    modes=('serial','fixed','dynamic')
    offset=(repetition+task_index)%len(modes)
    return modes[offset:]+modes[:offset]
