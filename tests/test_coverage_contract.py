from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e24_contract import ConversationTokenizer,CONTRACT,check_seal,sha


class CoverageContractTests(unittest.TestCase):
    def test_actual_roles_are_preserved_without_flattening_or_unexpected_override(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):return messages,kwargs
        messages=[dict(role='user',content='first'),dict(role='assistant',content='reply'),dict(role='user',content='last')]
        adapter=ConversationTokenizer(Tokenizer(),dict(prompt='last',messages=messages))
        result,kwargs=adapter.apply_chat_template([dict(role='user',content='last')],tokenize=True)
        self.assertEqual(result,messages);self.assertTrue(kwargs['tokenize'])
        with self.assertRaises(AssertionError):adapter.apply_chat_template([dict(role='user',content='other')])
        with self.assertRaises(AssertionError):ConversationTokenizer(Tokenizer(),dict(prompt='last',messages=messages[:2]))

    def test_training_inputs_and_runtime_sources_are_guarded(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'artifacts').mkdir();p=root/'source.py';p.write_text('source')
            d=root/'data.json';d.write_text('data')
            value=dict(contract=CONTRACT,source_sha256={'source.py':sha(p)},input_sha256={'data.json':sha(d)})
            (root/'artifacts/e24-seal.json').write_text(json.dumps(value))
            check_seal(root);d.write_text('changed')
            with self.assertRaises(AssertionError):check_seal(root)


if __name__=='__main__':unittest.main()
