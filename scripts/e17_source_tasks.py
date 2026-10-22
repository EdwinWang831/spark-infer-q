"""Prepare frozen external human prompts and grouped mixed-language derivatives."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata

REVISION='fdf72ae0827c1cda404aff25b6603abec9e3399b'
SOURCE_FILE='2023-04-12_oasst_prompts.messages.jsonl.gz'
SOURCE_SHA256='621ccd86a6ef320ca4e24c137121bd4b39bcc7a0df839f0897fcc965ef2076ed'
SEED='e17-external-human-20261022'


def normalized(text):
    return re.sub(r'[^\w]+','',unicodedata.normalize('NFKC',text).casefold())


def trigrams(text):
    return {text[i:i+3] for i in range(max(1,len(text)-2))}


def key(message,salt):
    return hashlib.sha256((SEED+salt+message['message_id']).encode()).hexdigest()


def select_roots(messages,per_language=240,exclude=()):
    picked={};seen={normalized(t) for t in exclude};grams=[]
    for language in ('en','zh'):
        candidates=[m for m in messages if m.get('lang')==language and m.get('parent_id') is None
            and m.get('role')=='prompter' and m.get('review_result') is True and not m.get('deleted')
            and m.get('synthetic') is False and (40 if language=='en' else 16)<=len(m.get('text',''))<=(2400 if language=='en' else 900)]
        keep=[]
        for m in sorted(candidates,key=lambda m:key(m,'sample')):
            norm=normalized(m['text'])
            if norm in seen:continue
            gs=trigrams(norm)
            if any(len(gs&g)/len(gs|g)>=.82 for g in grams):continue
            keep.append(m);seen.add(norm);grams.append(gs)
            if len(keep)==per_language:break
        if len(keep)!=per_language:raise ValueError(f'insufficient distinct reviewed human {language} roots')
        picked[language]=sorted(keep,key=lambda m:key(m,'split'))
    return picked


def prepare(source,out):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(source/SOURCE_FILE)==SOURCE_SHA256
    with gzip.open(source/SOURCE_FILE,'rt') as handle:messages=[json.loads(line) for line in handle]
    from e16_tasks import TASKS as OLD_TASKS
    from e16_online_tasks import TASKS as OLD_ONLINE
    picked=select_roots(messages,exclude=[t['prompt'] for t in OLD_TASKS+OLD_ONLINE])
    tasks=[]
    for language,roots in picked.items():
        for i,m in enumerate(roots):
            split='train' if i<168 else 'calibration' if i<204 else 'test'
            task=dict(id='oa_'+m['message_id'],group=m['message_tree_id'],source_message_id=m['message_id'],
                      source_text_sha256=hashlib.sha256(m['text'].encode()).hexdigest(),language=language,split=split,
                      prompt=m['text'],thinking=False,output_cap=384,derived=False)
            tasks.append(task)
        for split,count in [('train',42),('calibration',9),('test',9)]:
            options=sorted((t for t in tasks if t['language']==language and t['split']==split and not t['derived']),
                           key=lambda t:hashlib.sha256((SEED+'mixed'+t['id']).encode()).hexdigest())
            for original in options[:count]:
                extra=('\n请用原问题的语言作答，并在最后附上两句中文总结。' if language=='en' else
                       '\n请用中文作答，并在最后附上两句英文总结。')
                tasks.append(dict(original,id=original['id']+'_mixed',language='mixed',prompt=original['prompt']+extra,
                                  derived=True,parent_task=original['id']))
    tasks.sort(key=lambda t:hashlib.sha256((SEED+'order'+t['id']).encode()).hexdigest())
    assert len(tasks)==600
    manifest=hashlib.sha256(json.dumps(tasks,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    summary=dict(status='PREPARED_NOT_COLLECTED',source='OpenAssistant/oasst1',revision=REVISION,
        license='Apache-2.0',source_filename=SOURCE_FILE,source_sha256=SOURCE_SHA256,
        source_license_sha256=sha(source/'LICENSE'),seed=SEED,manifest_sha256=manifest,
        tasks=600,independent_roots=480,original_human_prompts=480,derived_mixed_prompts=120,
        language_counts=dict(Counter(t['language'] for t in tasks)),split_counts=dict(Counter(t['split'] for t in tasks)),
        filters='reviewed true, nondeleted nonsynthetic root prompter; 40..2400 en /16..900 zh chars; normalized exact and character-trigram Jaccard .82 dedup',
        limits=['root-only single-turn human prompts, not real product traffic or a representative US-user sample',
                'character dedup does not prove absence of semantic paraphrases or source contamination in pretrained weights',
                'actual tokenizer capacity preflight still required; no Target labels collected'],
        index=[{k:v for k,v in t.items() if k!='prompt'} for t in tasks])
    out.mkdir(parents=True,exist_ok=True)
    (out/'new-tasks.json').write_text(json.dumps(dict(contract=summary,tasks=tasks),indent=2,ensure_ascii=False)+'\n')
    (out/'source-manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='index'}))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();prepare(a.source,a.out)
