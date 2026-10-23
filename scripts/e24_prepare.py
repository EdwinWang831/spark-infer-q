"""Prepare a separate coverage cohort; never alter E23 training inputs."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from e17_source_tasks import SOURCE_FILE, SOURCE_SHA256, normalized, trigrams
from e16_tasks import TASKS
from e16_online_tasks import TASKS as ONLINE

SEED='e24-coverage-20261023'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order(value):
    return hashlib.sha256((SEED+str(value)).encode()).hexdigest()


def prepare(oasst_source, added_source, old_manifest, e23_manifest, output):
    old=json.loads(old_manifest.read_text())['tasks']
    e23=json.loads(e23_manifest.read_text())['tasks']
    used={t['group'] for t in old+e23}
    seen={normalized(t['prompt']) for t in old+e23+TASKS+ONLINE}
    grams=[trigrams(t) for t in seen]
    assert sha(oasst_source/SOURCE_FILE)==SOURCE_SHA256
    for folder in ('gsm8k','mbpp'):
        provenance=json.loads((added_source/folder/'provenance.json').read_text())
        for name,entry in provenance['files'].items():
            assert sha(added_source/folder/name)==entry['sha256'],name
    card=json.loads((added_source/'mbpp/dataset-card-provenance.json').read_text())
    assert sha(added_source/'mbpp/dataset-card.md')==card['sha256']
    with gzip.open(oasst_source/SOURCE_FILE,'rt') as handle:
        messages=[json.loads(line) for line in handle]
    tasks=[]
    def distinct(text):
        n=normalized(text);g=trigrams(n)
        if n in seen or any(len(g&v)/len(g|v)>=.82 for v in grams):return False
        seen.add(n);grams.append(g);return True
    def split(index, n):
        return 'train' if index<n*5//7 else 'calibration' if index<n*6//7 else 'test'
    def add_pool(pool, family, language, count=84):
        selected=[]
        for item in sorted(pool,key=lambda x:order(x['id'])):
            if item['group'] in used or not distinct(item['prompt']):continue
            selected.append(item);used.add(item['group'])
            if len(selected)==count:break
        assert len(selected)==count,(family,language,len(selected))
        for i,item in enumerate(selected):
            tasks.append(dict(item,split=split(i,count),family=family,language=language,
                              thinking=False,output_cap=384))
    for lang in ('en','zh'):
        roots=[m for m in messages if m.get('lang')==lang and m.get('parent_id') is None
               and m.get('role')=='prompter' and m.get('review_result') is True
               and not m.get('deleted') and m.get('synthetic') is False
               and (40 if lang=='en' else 16)<=len(m.get('text',''))<=(2400 if lang=='en' else 900)]
        add_pool([dict(id='e24_oa_'+m['message_id'],group=m['message_tree_id'],prompt=m['text'],
                       source='OpenAssistant/oasst1',source_id=m['message_id']) for m in roots],
                 'external_qa',lang)
    mbpp=[json.loads(line) for line in (added_source/'mbpp/mbpp.jsonl').read_text().splitlines()]
    add_pool([dict(id='e24_mbpp_'+str(r['task_id']),group='mbpp:'+str(r['task_id']),
                   prompt=r['text']+'\nReturn Python code that passes:\n'+'\n'.join(r['test_list']),
                   source='MBPP',source_id=str(r['task_id'])) for r in mbpp if 601<=r['task_id']<=974],
             'code','en')
    gsm=[json.loads(line) for line in (added_source/'gsm8k/train.jsonl').read_text().splitlines()]
    add_pool([dict(id='e24_gsm_'+hashlib.sha256(r['question'].encode()).hexdigest()[:16],
                   group='gsm8k:'+hashlib.sha256(r['question'].encode()).hexdigest(),
                   prompt=r['question'],source='GSM8K/train',source_id=str(i)) for i,r in enumerate(gsm)],
             'math','en')
    conversation_source=added_source/'oasst-conversations/2023-04-12_oasst_ready.messages.jsonl.gz'
    provenance=json.loads((conversation_source.parent/'provenance.json').read_text())
    assert sha(conversation_source)==provenance['sha256']
    with gzip.open(conversation_source,'rt') as handle:
        conversations_all=[json.loads(line) for line in handle]
    by_id={m['message_id']:m for m in conversations_all}
    for lang in ('en','zh'):
        # Distinct synthetic ledgers; queried first rows survive documented truncation.
        for i in range(42):
            group=f'e24-ledger-{lang}-{i:03d}'
            rows=[f'ROW {k:04d}: items={(k*13+i*7)%97+1}; unit_price={(k*17+i*11)%83+1}; label={group}-{k:04d}'
                  for k in range(240)]
            intro='A purchase ledger follows.\n' if lang=='en' else '下面是采购记录。\n'
            question=('Using rows 0000 and 0001, compute their combined items*unit_price cost. Show the two calculations.'
                      if lang=='en' else '请根据第0000和0001行，计算两行items乘unit_price的总和，并列出两步计算。')
            tasks.append(dict(id=group,group=group,source='authored arithmetic ledger',source_id=group,
                prompt=intro+'\n'.join(rows)+'\n'+question,language=lang,family='long_context',
                split=split(i,42),thinking=False,output_cap=384,target_prompt_tokens=[768,1024,1536][i%3]))
        conversations=[]
        for m in conversations_all:
            if (m.get('lang')!=lang or m.get('role')!='prompter' or not m.get('parent_id')
                    or m.get('review_result') is not True or m.get('deleted') or m.get('synthetic') is not False):continue
            parent=by_id.get(m['parent_id']);root=by_id.get(parent.get('parent_id')) if parent else None
            if (not root or root.get('parent_id') is not None or root.get('role')!='prompter'
                    or parent.get('role')!='assistant'):continue
            chain=[root,parent,m]
            if any(x.get('review_result') is not True or x.get('deleted') or x.get('synthetic') is not False for x in chain):continue
            if not (40 if lang=='en' else 16)<=len(m['text']):continue
            if sum(len(x['text']) for x in chain)>(1600 if lang=='en' else 900):continue
            conversations.append(dict(id='e24_dialog_'+m['message_id'],group=m['message_tree_id'],
                prompt=m['text'],source='OpenAssistant/oasst1',source_id=m['message_id'],
                messages=[dict(role='assistant' if x['role']=='assistant' else 'user',content=x['text']) for x in chain]))
        add_pool(conversations,'multi_turn',lang,count=42)
    tasks.sort(key=lambda t:order(t['id']+'final'))
    assert len(tasks)==504 and len({t['group'] for t in tasks})==504
    assert Counter(t['split'] for t in tasks)==dict(train=360,calibration=72,test=72)
    files=[oasst_source/SOURCE_FILE,old_manifest,e23_manifest]
    files+=sorted(p for p in added_source.rglob('*') if p.is_file())
    sources={str(p.relative_to(added_source)) if p.is_relative_to(added_source) else p.name:sha(p) for p in files}
    value=dict(status='PREPARED_NOT_TOKENIZED_OR_COLLECTED',seed=SEED,source_sha256=sources,
        counts=dict(split=dict(Counter(t['split'] for t in tasks)),language=dict(Counter(t['language'] for t in tasks)),
                    family=dict(Counter(t['family'] for t in tasks))),
        limits=['official source train portions only for MBPP/GSM8K; no native benchmark score claim',
                'MBPP solutions and GSM8K answers never enter prompts or predictor training targets',
                'OASST conversations have three original reviewed nonsynthetic messages',
                'long contexts are authored ledgers with documented first/tail token cropping, not natural long documents',
                'all old E17/E23 roots excluded; lexical dedup does not establish semantic or pretraining independence',
                'tokenization and new multi-turn collector correctness still pending'],tasks=tasks)
    output.mkdir(parents=True,exist_ok=True)
    for name,ts in [('e24_all_tasks.json',tasks),('e24_learning_tasks.json',[t for t in tasks if t['split']!='test']),
                    ('e24_test_tasks.json',[t for t in tasks if t['split']=='test'])]:
        p=output/name;assert not p.exists();p.write_text(json.dumps(dict(value,tasks=ts),indent=2,ensure_ascii=False)+'\n')
    public={k:v for k,v in value.items() if k!='tasks'}
    public['manifests']={p.name:sha(p) for p in output.glob('e24_*tasks.json')}
    public['index']=[{k:v for k,v in t.items() if k not in ('prompt','messages')} for t in tasks]
    return public


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('oasst_source','added_source','old_manifest','e23_manifest','output'):
        p.add_argument(name,type=Path)
    a=p.parse_args();result=prepare(a.oasst_source,a.added_source,a.old_manifest,a.e23_manifest,a.output)
    (a.output/'source-manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','counts','manifests')}))
