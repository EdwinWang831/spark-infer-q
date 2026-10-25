"""Real pinned native scheduling, deterministic host backend; no GPU timings."""
import importlib


class Backend:
    def __init__(self):self.expected={};self.live={};self.released=[]
    def register(self,name,ids,limit,eos=False):
        history=list(ids);tokens=[]
        for _ in range(limit):
            t=99 if eos and len(history)>=len(ids)+3 else sum(history)%97+1
            tokens.append(t)
            if t==99:break
            history.append(t)
        self.expected[name]=dict(tokens=tokens,inputs=list(ids)+tokens[:-1])
    def create(self,h):self.live[h]=[];return self.live[h]
    def position(self,s):return len(s)
    def advance(self,row,s):
        assert self.live[row.handle] is s
        e=self.expected[row.handle.request_id]
        assert list(row.token_ids)==e['inputs'][row.position:row.position+len(row.token_ids)]
        s.extend(row.token_ids)
        return e['tokens'][len(s)-len(e['inputs'])+len(e['tokens'])-1] if not row.prompt or row.last_prompt_chunk else 1
    def release(self,h,s):assert self.live.pop(h) is s;self.released.append(h)


def execute(session,source,tag,args,load,report):
    helper=load('scripts/e48_run.py');native_mod=importlib.import_module(tag+'.native_scheduler');bridge_mod=importlib.import_module(tag+'.deepinfer_bridge')
    report.update(status='RUNNING_E48_PILOT',cases=[])
    with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=0) as native:
        report['native']=native.metadata
        for name in ('mixed','early_eos','late_admission','cancel_waiting','cancel_prepared_prompt','cancel_prepared_decode'):
            for policy in (False,True):
                be=Backend();b=bridge_mod.SchedulerBridge(native.core,native.torch,native.group,native.sampling_class,be,
                    vocab_size=100,eos_ids=(99,),capacity=16384,max_requests=2,token_budget=1024,prefill_interleave=policy)
                ids_a=[1]*3073;ids_b=[2]*8193;handles={};trace=[];cancelled=None;prepared_cancel=False
                def add(key,ids,limit,eos=False):
                    be.register(key,ids,limit,eos);handles[key]=b.add(key,ids,limit)
                try:
                    add('a',ids_a,24,name=='early_eos')
                    if name!='late_admission':add('b',ids_b,27)
                    if name=='cancel_waiting':b.cancel(handles['b']);cancelled=handles['b']
                    for step in range(160):
                        if name=='late_admission' and 'b' not in handles and any(e['seq'].data.get_len()>e['prompt_length']+1 for e in b.requests.values()):add('b',ids_b,27)
                        if not b.requests:break
                        ready=any(e['seq'].status==native.core.SequenceStatus.RUNNING for e in b.requests.values())
                        waiting=any(e['seq'].status==native.core.SequenceStatus.WAITING for e in b.requests.values())
                        p=b.prepare();assert p.rows,'native empty device plan'
                        b.execute(p)
                        phase='prompt' if p.rows[0].prompt else 'decode'
                        if not prepared_cancel and name=='cancel_prepared_'+phase:
                            victim=p.rows[0].handle;b.cancel(victim);cancelled=victim;prepared_cancel=True
                        b.complete(p)
                        trace.append(dict(phase=phase,ready_and_waiting=ready and waiting,
                            rows=[dict(name=r.handle.request_id,position=r.position,rows=len(r.token_ids),prompt=r.prompt) for r in p.rows]))
                    assert not b.requests and ('b' in handles or name=='late_admission')
                    if name.startswith('cancel_prepared'):assert prepared_cancel
                    for h in handles.values():
                        if h==cancelled:assert b.completed[h]['cancelled']
                        else:assert b.completed[h]['token_ids']==be.expected[h.request_id]['tokens']
                    longest=run=0
                    for t in trace:
                        run=run+1 if t['ready_and_waiting'] and t['phase']=='prompt' else 0;longest=max(longest,run)
                    if policy:assert longest<=1,'prefill starved ready decode'
                    old=handles['a'];add('a',[5]*33,3);b.drain(max_steps=16)
                    assert handles['a'].generation>old.generation and b.completed[handles['a']]['token_ids']==be.expected['a']['tokens']
                    try:b.cancel(old)
                    except bridge_mod.BridgeError:pass
                    else:raise AssertionError('stale generation accepted')
                    b.close();assert not be.live and b.blocks.get_num_free_blocks()==b.block_count-1
                    report['cases'].append(dict(name=name,interleave=policy,trace=trace,max_prompt_run_while_decode_ready=longest,
                        outputs_exact=True,blocks_recovered=True,generation_reuse=True,hints=dict(b.interleave_hints)))
                    helper.save(source,report)
                finally:
                    if b.pending is not None:
                        if not b.pending.executed:b.execute(b.pending)
                        b.complete(b.pending)
                    if not b.closed:b.close()
    report['status']='PASS_E48_PILOT';helper.save(source,report);return report
