"""Twenty fixed source-test roots chosen without reading any Target labels."""
import hashlib
from e17_run_contract import load_tasks


def get_tasks(root):
    candidates=[t for t in load_tasks(root) if t['split']=='test']
    chosen=[];used=set()
    for lang,n in [('mixed',4),('en',8),('zh',8)]:
        ordered=sorted((t for t in candidates if t['language']==lang),key=lambda t:hashlib.sha256(('e17-online-v1'+t['id']).encode()).hexdigest())
        for t in ordered:
            if t['group'] in used:continue
            chosen.append(t);used.add(t['group'])
            if sum(v['language']==lang for v in chosen)==n:break
        assert sum(v['language']==lang for v in chosen)==n
    assert len(chosen)==len(used)==20
    return sorted(chosen,key=lambda t:hashlib.sha256(('e17-online-order'+t['id']).encode()).hexdigest())
