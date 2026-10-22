"""Summarize a finished paired probe without treating prefix oracles as online TPS."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def oracle(positions,allowed):
    rate=0.
    for _ in range(100):
        chosen=[max((a for a in p if a['block'] in allowed),key=lambda a:a['emitted']-rate*a['total_ms']) for p in positions]
        nxt=sum(a['emitted'] for a in chosen)/sum(a['total_ms'] for a in chosen)
        if abs(nxt-rate)<1e-13:break
        rate=nxt
    else:raise RuntimeError('throughput oracle did not converge')
    return dict(tokens_per_ms=nxt,action_counts=dict(Counter(f"{a['block']}:{a['width']}" for a in chosen)),
                emitted=sum(a['emitted'] for a in chosen),cost_ms=sum(a['total_ms'] for a in chosen))


def summarize(raw):
    assert raw['status']=='PASS_E18_DRAFT_PROBE'
    positions=[];draft4=[];draft8=[];cold4=[];cold8=[];same=0;checks=0
    for task in raw['tasks']:
        for rec in task['positions']:
            by_action={}
            assert all(a['exact'] for a in rec['actions'])
            checks+=len(rec['actions']);same+=rec['first_three_equal']
            for a in rec['actions']:by_action.setdefault((a['block'],a['width']),[]).append(a)
            actions=[]
            for (block,width),values in by_action.items():
                assert sorted(a['repetition'] for a in values)==list(range(raw['contract']['repetitions']))
                assert len({a['emitted'] for a in values})==1
                actions.append(dict(block=block,width=width,emitted=values[0]['emitted'],
                                    total_ms=statistics.median(a['total_ms'] for a in values)))
            positions.append(actions)
            draft4.append(rec['drafts']['4']['median_ms']);draft8.append(rec['drafts']['8']['median_ms'])
            cold4.append(rec['drafts']['4']['cold_ms']);cold8.append(rec['drafts']['8']['cold_ms'])
    only8=oracle(positions,{8});joint=oracle(positions,{0,4,8});only4=oracle(positions,{4})
    fixed={}
    for block,width in ((0,1),(4,4),(8,8)):
        chosen=[next(a for a in p if a['block']==block and a['width']==min(width,max(v['width'] for v in p if v['block']==block))) for p in positions]
        fixed[f'{block}:{width}']=dict(emitted=sum(a['emitted'] for a in chosen),total_ms=sum(a['total_ms'] for a in chosen),
                                     tokens_per_ms=sum(a['emitted'] for a in chosen)/sum(a['total_ms'] for a in chosen))
    return dict(status='PASS_E18_FEASIBILITY_ANALYSIS',positions=len(positions),formal_actions=checks,
        warmup_actions=checks//raw['contract']['repetitions'],first_three_equal=same,
        draft=dict(warm_block4_median_ms=statistics.median(draft4),warm_block8_median_ms=statistics.median(draft8),
                   warm_paired_sum_reduction=1-sum(draft4)/sum(draft8),cold_block4_median_ms=statistics.median(cold4),cold_block8_median_ms=statistics.median(cold8)),
        offline_prefix_oracles=dict(block8=only8,block4=only4,joint=joint,
                                   joint_relative_to_block8=joint['tokens_per_ms']/only8['tokens_per_ms']-1),
        fixed_diagnostics=fixed,
        limits=['training prefixes only; hindsight token/cost aggregate oracle is not an online throughput bound or measured decode speed',
                'existing-cache append8 draft cost, initial cache cold timing separate; no gate/feature predictor fee included',
                'draft block4 measured before block8; target actions counterbalanced; independent online timing required',
                'four-row proposals can differ from the first three of seven; seven-depth feature heads cannot be directly reused'],
        contract=raw['contract'],source_sha256=raw['source_sha256'],q4_arithmetic=raw['q4_arithmetic'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('raw',type=Path);parser.add_argument('out',type=Path)
    a=parser.parse_args();raw=json.loads(a.raw.read_text());r=summarize(raw)
    r['raw_sha256']=hashlib.sha256(a.raw.read_bytes()).hexdigest();r['exporter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    r['evidence_tasks']=raw['tasks'];a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='evidence_tasks'}))
