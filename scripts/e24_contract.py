"""Separate data-coverage experiment; no E23 test rows enter training."""
import hashlib
import json
from pathlib import Path

CONTRACT=dict(train_prompts=360,calibration_prompts=72,test_prompts=72,learning_shards=9,test_shards=3,
    shard_size=48,output_cap=384,capacity=2048,architecture=[137,64,32,7],seed=20261022,l2=.005,max_iter=80,
    objective='E23 calibration-selected recipe, fixed before new data labels',
    sources=['OASST1 reviewed human questions/conversations','MBPP official train','GSM8K train','authored long-context ledgers'],
    stages='tokenization -> paired producer validation -> learning collection -> fit/calibration -> fresh test if selected',
    limits=['controlled coverage study, not representative user traffic',
            'MBPP solutions and GSM8K answers excluded from prompts and supervision',
            'same old costs and fast backend; no simultaneous cost-curve refitting',
            'synthetic ledger template shared across partitions; no unseen-template generalization claim',
            'fresh test opens only after calibration choice; no test-driven tuning or automatic deployment'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    with Path(path).open('x') as handle:handle.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def check_seal(root):
    seal=json.loads((root/'artifacts/e24-seal.json').read_text());assert seal['contract']==CONTRACT
    for label in ('source_sha256','input_sha256'):
        for name,digest in seal[label].items():assert sha(root/name)==digest,name
    return seal


class ConversationTokenizer:
    """Supply the actual known role history to the unchanged collection core."""
    def __init__(self,tokenizer,task):
        messages=task.get('messages',[dict(role='user',content=task['prompt'])])
        assert messages and len(messages)%2==1 and messages[-1]['content']==task['prompt']
        assert all(m['role']==('user' if i%2==0 else 'assistant') and isinstance(m['content'],str) and m['content']
                   for i,m in enumerate(messages))
        self.tokenizer,self.messages,self.prompt=tokenizer,messages,task['prompt']

    def apply_chat_template(self,messages,**kwargs):
        assert messages==[dict(role='user',content=self.prompt)]
        return self.tokenizer.apply_chat_template(self.messages,**kwargs)
