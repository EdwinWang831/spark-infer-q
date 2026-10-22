"""Actual 1K--2K bounded generation, sharing the original resident weights."""
from e13_tasks import CONTRACT as BASE_CONTRACT

TEXT = ('An event processor uses a durable log, monotonically increasing sequence numbers, '
        'and periodic snapshots. Retried requests carry unique identifiers. Recovery must '
        'preserve committed results and avoid applying an event twice. ')*200
TEXT += '\nWrite a detailed explanation of this recovery design in about 180 words, with a practical example.'
TASKS = [dict(id='long_'+str(n),split='long_validation',thinking=False,
              prompt='E13_LENGTH='+str(n)+'\n'+TEXT) for n in (1024,1536,1920)]
CONTRACT = dict(BASE_CONTRACT,capacity=2048,prompt_contexts=[1024,1536,1920],
    scope='actual long-prompt bounded greedy decode; excludes prompt, cloning and network; same resident weights')


def run(session, arguments):
    from types import SimpleNamespace
    from e3_wide_target import WideTarget
    from spark_infer_q.draft_head import HeadDraft
    from importlib import reload
    import e13_study
    study=reload(e13_study)
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            marker,text=messages[0]['content'].split('\n',1)
            n=int(marker.split('=')[1])
            ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],**kwargs)
            assert len(ids)>n
            # Preserve the chat opening, full final question and assistant prefix.
            return ids[:n-96]+ids[-96:]
    scaled=SimpleNamespace(**vars(session))
    scaled.target=WideTarget(session.weights,capacity=2048)
    scaled.draft=HeadDraft.attach(session.draft,session.projection)
    scaled.draft.max_context=2048
    scaled.tokenizer=Tokenizer()
    result=study.run(scaled,dict(phase='integrated',tasks=TASKS,contract=CONTRACT,
                                sources=['e13_long_decode.py'],
                                artifact='e13-long-integrated.json'))
    assert session.draft.max_context==1024
    result.update(shared_target_weights=scaled.target.w is session.weights,
                  shared_draft_weights=scaled.draft.weights is session.draft.weights,
                  original_draft_object_id=id(session.draft))
    return result
