"""Chunk a prompt through the target's validated decode arithmetic and commit path."""


def prompt_state(target, token_ids, *, chunk_size=8, capture=True):
    """Return committed prompt state, draft features and final-row logits.

    The target owns row buffers and recurrent/KV state. No independent prefill
    arithmetic or speculative prefix cropping is used here.
    """
    if type(chunk_size) is not int or chunk_size not in (1, 2, 4, 8):
        raise ValueError('prompt chunk_size must be 1, 2, 4 or 8')
    if not token_ids:
        raise ValueError('prompt must contain at least one token')
    if len(token_ids)>target.capacity:
        raise ValueError('prompt exceeds target capacity')
    if any(n not in target.buffers for n in range(1,chunk_size+1)):
        raise ValueError('target does not support the requested prompt chunk size')
    state=target.state();features=[]
    for begin in range(0,len(token_ids),chunk_size):
        logits,hidden=target.step(state,token_ids[begin:begin+chunk_size],capture=capture)
        if capture:features.append(hidden)
    context=target.torch.cat(features) if capture else None
    return state,context,logits[-1:]
