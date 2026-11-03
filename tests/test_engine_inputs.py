"""The long-prompt fixture must really reach the requested token count."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


path = Path(__file__).resolve().parents[1]/'scripts/k4batch_engine_long.py'
spec = importlib.util.spec_from_file_location('engine_input_fixture', path)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class Tokenizer:
    def __init__(self):
        self.encoded_lengths = []

    def encode(self, text, **kwargs):
        self.encoded_lengths.append(len(text))
        return [1]*(len(text)//16)

    def decode(self, ids, **kwargs):
        return 'x'*len(ids)

    def apply_chat_template(self, messages, **kwargs):
        return [1]*len(messages[0]['content'])


class EngineInputsTests(unittest.TestCase):
    def test_128k_inputs_extend_filler_for_all_four_request_types(self):
        for index in range(4):
            tokenizer = Tokenizer()
            result = fixture.make_messages(SimpleNamespace(tokenizer=tokenizer), index, prompt_tokens=131008)
            self.assertEqual(result['prompt_tokens'], 131008)
            self.assertGreater(len(tokenizer.encoded_lengths), 1)
            self.assertEqual(len(result['ids']), 131008)


if __name__ == '__main__':
    unittest.main()
