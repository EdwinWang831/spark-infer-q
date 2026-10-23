"""Twelve fresh online roots, selected from existing licensed sources before linear cost/runtime testing."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from e17_source_tasks import SOURCE_FILE, SOURCE_SHA256, normalized, trigrams
from e16_tasks import TASKS
from e16_online_tasks import TASKS as ONLINE

SEED='e27-residual-fresh-20261023'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def order(value):
    return hashlib.sha256((SEED+value).encode()).hexdigest()


def prepare(oasst, added, exclusions, output):
    previous=[t for p in exclusions for t in json.loads(p.read_text())['tasks']]+TASKS+ONLINE
    used={t.get('group',t['id']) for t in previous}
    seen={normalized(t['prompt']) for t in previous}
    grams=[trigrams(t) for t in seen]
    assert sha(oasst/SOURCE_FILE)==SOURCE_SHA256
    sources={'oasst/'+SOURCE_FILE:SOURCE_SHA256}
    for folder in ('gsm8k','mbpp'):
        provenance=json.loads((added/folder/'provenance.json').read_text())
        for n,entry in provenance['files'].items():
            assert sha(added/folder/n)==entry['sha256']
            sources[folder+'/'+n]=entry['sha256']
    dialog=added/'oasst-conversations/2023-04-12_oasst_ready.messages.jsonl.gz'
    assert sha(dialog)==json.loads((dialog.parent/'provenance.json').read_text())['sha256']
    sources['oasst-conversations/'+dialog.name]=sha(dialog)
    tasks=[]

    def add(pool, family, language, count):
        selected=[]
        for t in sorted(pool,key=lambda t:order(t['id'])):
            n=normalized(t['prompt']);g=trigrams(n)
            if t['group'] in used or n in seen or any(len(g&v)/len(g|v)>=.82 for v in grams):continue
            selected.append(t);used.add(t['group']);seen.add(n);grams.append(g)
            if len(selected)==count:break
        assert len(selected)==count,(family,language)
        tasks.extend(dict(t,family=family,language=language,thinking=False,output_cap=384,split='test') for t in selected)

    with gzip.open(oasst/SOURCE_FILE,'rt') as handle:messages=[json.loads(line) for line in handle]
    for lang in ('en','zh'):
        pool=[m for m in messages if m.get('lang')==lang and m.get('parent_id') is None
            and m.get('role')=='prompter' and m.get('review_result') is True and not m.get('deleted')
            and m.get('synthetic') is False and (40 if lang=='en' else 16)<=len(m.get('text',''))<=(2400 if lang=='en' else 900)]
        add([dict(id='e27_oa_'+m['message_id'],group=m['message_tree_id'],prompt=m['text'],source='OpenAssistant/oasst1') for m in pool],
            'external_qa',lang,2)
    mbpp=[json.loads(s) for s in (added/'mbpp/mbpp.jsonl').read_text().splitlines()]
    add([dict(id='e27_mbpp_'+str(r['task_id']),group='mbpp:'+str(r['task_id']),
        prompt=r['text']+'\nReturn Python code that passes:\n'+'\n'.join(r['test_list']),source='MBPP/train')
        for r in mbpp if 601<=r['task_id']<=974],'code','en',2)
    gsm=[json.loads(s) for s in (added/'gsm8k/train.jsonl').read_text().splitlines()]
    add([dict(id='e27_gsm_'+hashlib.sha256(r['question'].encode()).hexdigest()[:16],
        group='gsm8k:'+hashlib.sha256(r['question'].encode()).hexdigest(),prompt=r['question'],source='GSM8K/train') for r in gsm],
        'math','en',2)
    with gzip.open(dialog,'rt') as handle:ready=[json.loads(line) for line in handle]
    by_id={m['message_id']:m for m in ready}
    for lang in ('en','zh'):
        pool=[]
        for m in ready:
            if m.get('lang')!=lang or m.get('role')!='prompter' or not m.get('parent_id'):continue
            parent=by_id.get(m['parent_id']);root=by_id.get(parent.get('parent_id')) if parent else None
            if not root or root.get('parent_id') is not None or root.get('role')!='prompter' or parent.get('role')!='assistant':continue
            chain=[root,parent,m]
            if any(x.get('review_result') is not True or x.get('deleted') or x.get('synthetic') is not False for x in chain):continue
            if not (40 if lang=='en' else 16)<=len(m['text']):continue
            if sum(len(x['text']) for x in chain)>(1600 if lang=='en' else 900):continue
            pool.append(dict(id='e27_dialog_'+m['message_id'],group=m['message_tree_id'],prompt=m['text'],source='OpenAssistant/oasst1',
                messages=[dict(role='assistant' if x['role']=='assistant' else 'user',content=x['text']) for x in chain]))
        add(pool,'multi_turn',lang,1)
    for lang,index,target in [('en',2001,768),('zh',2002,1024)]:
        group=f'e27-ledger-{lang}-{index}'
        rows=[f'ROW {k:04d}: items={(k*13+index*7)%97+1}; unit_price={(k*17+index*11)%83+1}; label={group}-{k:04d}' for k in range(240)]
        intro='A purchase ledger follows.\n' if lang=='en' else '下面是采购记录。\n'
        question=('Using rows 0000 and 0001, compute their combined items*unit_price cost. Show the two calculations.'
                  if lang=='en' else '请根据第0000和0001行，计算两行items乘unit_price的总和，并列出两步计算。')
        tasks.append(dict(id=group,group=group,prompt=intro+'\n'.join(rows)+'\n'+question,source='authored arithmetic ledger',
            family='long_context',language=lang,thinking=False,output_cap=384,split='test',target_prompt_tokens=target))
    assert len(tasks)==len({t['group'] for t in tasks})==12
    assert not {t['group'] for t in tasks}&{t.get('group',t['id']) for t in previous}
    tasks.sort(key=lambda t:order(t['id']+'final'))
    provenance=dict(seed=SEED,source_sha256=sources,
        exclusion_sha256={str(p.name):sha(p) for p in exclusions},
        languages=dict(Counter(t['language'] for t in tasks)),families=dict(Counter(t['family'] for t in tasks)),
        limits=['existing E24 source licenses apply; MBPP CC-BY-4.0, GSM8K MIT, OASST1 Apache-2.0',
                'natural source roots and normalized/trigram prompts excluded against E17/E23/E24 and earlier authored tasks',
                'synthetic ledgers are new values in a familiar template, not unseen-format generalization',
                'new benchmark selected before linear cost/runtime testing; no target outputs used in selection'])
    output.mkdir(parents=True,exist_ok=True)
    with (output/'e27_test_tasks.json').open('x') as h:h.write(json.dumps(dict(provenance=provenance,tasks=tasks),indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(provenance,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('oasst',type=Path);p.add_argument('added',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--exclude',type=Path,nargs='+',required=True);a=p.parse_args();prepare(a.oasst,a.added,a.exclude,a.output)
