"""A tiny real-target/state/drafter check on an already-loaded session."""


def run(session, arguments):
    from spark_infer_q.prompt import prompt_state
    torch, target = session.torch, session.target
    text = arguments.get('prompt', 'Write two sentences explaining how a cache works.')
    ids = session.tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    state, context, logits = prompt_state(target, ids, chunk_size=8)
    anchor = int(logits[-1].argmax())
    proposals = session.draft.propose(context, anchor)[0]
    serial, window = state.clone(), state.clone()
    sl, _ = target.step(serial, [anchor], capture=False)
    wl, _, buffers = target.run(window, [anchor]+proposals[:3])
    assert torch.equal(sl, wl[:1])
    target.commit(window, buffers, 4, 1)
    assert not target.state_difference(serial, window)
    return dict(prompt_tokens=len(ids), draft_tokens=len(proposals),
                first_logit_exact=True, committed_state_exact=True,
                target_object_id=id(session.weights), draft_object_id=id(session.draft))
