"""Small greedy verification contract: processed prefix and pending token are distinct."""


def greedy_result(anchor, drafts, target_next, eos_ids=(), max_new=None):
    """Return (new outputs, processed rows, accepted drafts, stopped).

    target_next[i] predicts after input [anchor,*drafts][i]. The final
    correction/bonus is emitted but pending; its state is not committed.
    """
    if len(target_next) != len(drafts)+1:
        raise ValueError('one prediction required for every verified input row')
    if max_new is not None and max_new < 1:
        raise ValueError('max_new must be positive')
    accepted = 0
    for proposed, golden in zip(drafts,target_next):
        if proposed != golden: break
        accepted += 1
    outputs = list(drafts[:accepted]) + [target_next[accepted]]
    stop_at = len(outputs) if max_new is None else min(len(outputs), max_new)
    stopped = False
    for i,t in enumerate(outputs[:stop_at]):
        if t in eos_ids:
            stop_at=i+1
            stopped=True
            break
    outputs=outputs[:stop_at]
    # Commit anchor and all emitted tokens except the final pending output.
    keep=len(outputs)
    return outputs, keep, min(accepted, max(0,keep-1)), stopped
