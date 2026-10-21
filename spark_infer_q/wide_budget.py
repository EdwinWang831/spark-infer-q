"""Experiment ordering and seven-depth acceptance accounting."""
MODES=('serial','fixed3','fixed3_wide','fixed7','dynamic7')


def order(repetition,task_index=0):
    offset=(repetition+task_index)%len(MODES)
    return MODES[offset:]+MODES[:offset]


def conditional_prior(steps,depths=7):
    success=[1.]*depths;trials=[2.]*depths
    for step in steps:
        observed,accepted=step['observed_drafts'],step['accepted_drafts']
        if not 0<=accepted<=observed<=depths:raise ValueError('invalid observed prefix')
        for depth in range(observed):
            if depth>accepted:break
            trials[depth]+=1;success[depth]+=int(depth<accepted)
    return [s/t for s,t in zip(success,trials)],success,trials
