"""Prompt adapter contract tests; real numeric/state equivalence needs the GPU study."""
import types
import unittest

from spark_infer_q.prompt import prompt_state


class Target:
    capacity=64
    buffers={n:None for n in range(1,9)}
    torch=types.SimpleNamespace(cat=lambda features:sum(features,[]))

    def __init__(self):self.calls=[]
    def state(self):return []
    def step(self,state,ids,capture=True):
        self.calls.append((list(ids),capture));logits=[];features=[]
        for token in ids:
            state.append(token)
            # Outputs depend on the entire preceding history, not just a token.
            logits.append(sum((i+1)*t for i,t in enumerate(state)))
            features.append(tuple(state))
        return logits,features if capture else None


class PromptTests(unittest.TestCase):
    def test_chunking_preserves_causal_context_and_short_tail(self):
        ids=list(range(19))
        serial=prompt_state(Target(),ids,chunk_size=1)
        target=Target();chunked=prompt_state(target,ids)
        self.assertEqual(chunked,serial)
        self.assertEqual([len(ids) for ids,_ in target.calls],[8,8,3])

    def test_target_only_does_not_collect_features(self):
        target=Target();state,context,lg=prompt_state(target,[7,8,9],capture=False)
        self.assertEqual(state,[7,8,9]);self.assertIsNone(context);self.assertEqual(lg,[50])
        self.assertEqual(target.calls,[([7,8,9],False)])

    def test_invalid_input_is_rejected_before_mutating_state(self):
        for ids,width in (([],8),([1]*65,8),([1],3),([1],True)):
            target=Target()
            with self.assertRaises(ValueError):prompt_state(target,ids,chunk_size=width)
            self.assertEqual(target.calls,[])
        target=Target();target.buffers={1:None}
        with self.assertRaises(ValueError):prompt_state(target,[1],chunk_size=8)
        self.assertEqual(target.calls,[])


if __name__=='__main__':unittest.main()
