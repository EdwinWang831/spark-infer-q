"""Fail before full weight loading on identity, tokenizer or study-input drift."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
    root=a.root;sys.path[:0]=[str(root/'followup-deps'),str(root/'python-deps'),str(root)]
    sys.path[:0]=[str(root/'build/qmm'),str(root/'build/varlen')]
    import torch,transformers
    import spark_q_e6_qmm,spark_q_e3_varlen
    for module,sha in ((spark_q_e6_qmm,'5e8088b6cbcb09d05bad9472f0616290457e662546dfdfe497dc5532879450dd'),(spark_q_e3_varlen,'722b6ea835fa3e7e349af7f22ac727045a65af487a35f6709f3838e066bd678e')):
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==sha
    from transformers import AutoTokenizer
    from e7_tasks import TASKS
    assert torch.__version__=='2.13.0+cu132' and transformers.__version__=='5.12.1'
    for name,sha in json.loads((root/'native-pin.json').read_text()).items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    vocab=json.loads((a.checkpoint/'config.json').read_text())['text_config']['vocab_size']
    tok=AutoTokenizer.from_pretrained(str(a.checkpoint),local_files_only=True,trust_remote_code=False)
    tasks=[]
    for t in TASKS:
        ids=tok.apply_chat_template([dict(role='user',content=t['prompt'])],tokenize=True,add_generation_prompt=True,enable_thinking=t['thinking'],return_dict=False)
        assert isinstance(ids,list) and 0<len(ids) and all(type(i) is int and 0<=i<vocab for i in ids)
        assert len(ids)+t['limit']<=1017
        tasks.append(dict(id=t['id'],prompt_ids=ids))
    frozen=json.loads((root/'e6-frozen.json').read_text())
    old=json.loads((root/'calibration.json').read_text());e6=frozen['evidence']['calibration']
    assert old['verify_commit_curve_ms']==e6['verify_commit_curve_ms']
    assert old['conditional_prior']==e6['heads']['q4_full']['conditional_prior']
    assert old['draft_median_ms']['8']==e6['heads']['q4_full']['proposer_median_ms']
    result=dict(status='PASS_PREFLIGHT',tasks=tasks,torch=torch.__version__,transformers=transformers.__version__,
        native_pins=json.loads((root/'native-pin.json').read_text()),e6_report_sha256=hashlib.sha256((root/'e6-frozen.json').read_bytes()).hexdigest())
    (root/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],tasks=len(tasks))))


if __name__=='__main__':main()
