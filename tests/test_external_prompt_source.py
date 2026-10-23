from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e17_source_tasks import select_roots,normalized


class ExternalPromptSourceTests(unittest.TestCase):
    def test_only_reviewed_human_roots_and_distinct_text_are_selected(self):
        def row(i,text,lang='en',**changes):
            return dict(message_id=str(i),text=text,lang=lang,parent_id=None,role='prompter',review_result=True,
                        deleted=False,synthetic=False,**changes)
        en1='Describe a practical method for storing handmade paper without moisture damage.'
        en2='Write a Python routine that enumerates integer factors in increasing order.'
        zh1='请解释潮汐为什么每天出现，以及海岸地形会如何影响潮差。'
        zh2='用一个具体的生活例子解释为什么平均数和中位数可能差很多。'
        rows=[row(1,en1),row(2,en1+'!'),row(3,en2),row(4,zh1,'zh'),row(5,zh2,'zh')]
        bad=row(6,'This synthetic request must never become a human training root.');bad['synthetic']=True;rows.append(bad)
        assistant=row(7,'This assistant answer must never be used as an independent question.');assistant['role']='assistant';rows.append(assistant)
        picked=select_roots(rows,per_language=2)
        self.assertEqual(len(picked['en']),2);self.assertEqual(len(picked['zh']),2)
        ids={r['message_id'] for rs in picked.values() for r in rs}
        self.assertNotIn('6',ids);self.assertNotIn('7',ids)
        self.assertEqual(len({normalized(r['text']) for r in picked['en']}),2)
        with self.assertRaises(ValueError):select_roots(rows,per_language=2,exclude=[en2])
