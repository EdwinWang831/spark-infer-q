"""Native C++ Scheduler speculative contract checks, optionally on CPU only."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def run(session, arguments):
    root=Path(arguments['source_root']).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    for name,sha in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    tag='_kaifra_k2_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    package=ModuleType(tag);package.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=package
    try:
        module=importlib.import_module(tag+'.kaifra_speculative')
        native_module=importlib.import_module(tag+'.native_scheduler')
        e1=session.root.parent
        with native_module.NativeScheduler(e1/'DeepInfer',e1/'build-release/lib/deepinfer',max_drafts=7) as native:
            result=cpu_suite(native,module)
        result.update(source_sha256=manifest,model_reloaded=False)
        return result
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)


class TorchProxy:
    def __init__(self,torch):
        self.actual=torch
        self.cuda=SimpleNamespace(synchronize=lambda:None)
    def __getattr__(self,name):return getattr(self.actual,name)


class ToyTarget:
    capacity=256
    def __init__(self,torch):
        self.torch=TorchProxy(torch);self.buffers={i:None for i in range(1,9)}
        self.w=SimpleNamespace(head=SimpleNamespace(b=SimpleNamespace(weight=SimpleNamespace(shape=(100,2)))))
    def state(self):return SimpleNamespace(pos=0,history=[])
    def run(self,state,ids,capture=True):
        t=self.torch
        lg=t.full((len(ids),100),-100.0)
        for i,token in enumerate(ids):lg[i,(token+1)%100]=1.0
        ft=t.tensor([[state.pos+i,token] for i,token in enumerate(ids)],dtype=t.float32)
        return lg,ft,(list(ids),)
    def commit(self,state,buf,rows,keep):
        state.history.extend(buf[0][:keep]);state.pos+=keep
    def step(self,state,ids,capture=True):
        lg,ft,buf=self.run(state,ids,capture);self.commit(state,buf,len(ids),len(ids));return lg,ft


class ToyProvider:
    max_context=256
    model=None
    def __init__(self,fault=None):self.fault=fault;self.calls=0
    def propose(self,handle,state,anchor,remaining):
        tokens=[(anchor+i+1)%100 for i in range(7)]
        if self.fault is not None and self.calls%2==0:tokens[self.fault]=(tokens[self.fault]+10)%100
        self.calls+=1
        return tokens,None


def cpu_suite(native,module):
    target=ToyTarget(native.torch);cases=[];checks=0
    def observe(handle,pos,ids,lg,keep,state):
        nonlocal checks
        assert state.history==[(1+i)%100 for i in range(state.pos)] or handle.request_id.startswith('eos')
        assert state.pos==pos+keep;checks+=1
    def make(mode='fixed7',fault=None,prior=None,max_requests=1,eos=(99,)):
        return module.speculative_engine(native,target,ToyProvider(fault),eos_ids=eos,
            prior=prior or [.7]*7,mode=mode,capacity=128,max_requests=max_requests,
            curve=[(r,1.+r*.2) for r in range(1,9)] if mode=='lagged' else None,
            observer=observe)
    def finish(e,name,events):
        assert not e.active_requests and not e.bridge.backend.states
        assert e.bridge.blocks.get_num_free_blocks()==e.bridge.block_count-1
        assert not e.bridge.native.has_unfinished_seqs()
        cases.append(dict(case=name,status='PASS',events=len(events)))
        e.close()
    for depth in (1,3,7):
        for fault in (None,0,3,6) if depth==7 else (None,0):
            e=make('fixed'+str(depth),fault)
            h=e.submit_tokens('sequence',[1],max_tokens=24);events=list(e.drain())
            assert [t for x in events for t in x.token_ids]==list(range(2,26))
            assert events[-1].processed_position==24
            finish(e,'fixed'+str(depth)+'_fault'+str(fault),events)
    e=make(max_requests=2);a=e.submit_tokens('a',list(range(1,34)),max_tokens=16)
    b=e.submit_tokens('b',[1],max_tokens=16);events=list(e.drain())
    assert [t for x in events if x.handle==a for t in x.token_ids]==list(range(34,50))
    assert [t for x in events if x.handle==b for t in x.token_ids]==list(range(2,18))
    finish(e,'two_requests_chunked_prompt',events)
    e=make(eos=(90,99));e.submit_tokens('eos_nonprimary',[97],max_tokens=8)
    events=list(e.drain());assert [t for x in events for t in x.token_ids]==[98,99]
    assert events[-1].finish_reason=='stop';finish(e,'nonprimary_eos_inside_proposal',events)
    e=make();e.submit_tokens('one',[1],max_tokens=1);events=list(e.drain())
    assert events[-1].processed_position==1;finish(e,'max_tokens_one',events)
    for phase in ('waiting','prepared_prompt','prepared_decode'):
        e=make();h=e.submit_tokens('reuse',[1],max_tokens=16)
        if phase=='prepared_decode':e.step()
        if phase!='waiting':p=e.bridge.prepare();e.bridge.execute(p)
        e.cancel(h)
        if phase!='waiting':e.bridge.complete(p)
        events=list(e.drain());assert not any(x.token_ids for x in events)
        n=e.submit_tokens('reuse',[1],max_tokens=8)
        assert n.generation>h.generation
        try:e.cancel(h)
        except RuntimeError:pass
        else:raise AssertionError('stale handle accepted')
        events+=list(e.drain());finish(e,'cancel_'+phase+'_reuse',events)
    class CurrentProvider(ToyProvider):
        model=object()
        budget=object()
        def propose(self,handle,state,anchor,remaining):
            state.next_width=min(remaining, (1,4,8)[self.calls%3])
            return super().propose(handle,state,anchor,remaining)
    for fault in (None,0,3,6):
        e=module.speculative_engine(native,target,CurrentProvider(fault),eos_ids=(99,),
            prior=[.7]*7,mode='adaptive',capacity=128,observer=observe)
        h=e.submit_tokens('sequence',[1],max_tokens=24)
        events=list(e.drain())
        assert [t for x in events for t in x.token_ids]==list(range(2,26))
        finish(e,'postdraft_finalization_fault'+str(fault),events)
    e=make('lagged');e.submit_tokens('dynamic',[1],max_tokens=24)
    events=[];widths=[]
    while e.active_requests:
        events+=e.step();widths += [m['verified_rows'] for m in e.last_step_metrics if not m['prompt']]
    assert 1 in widths and max(widths)>1,widths
    finish(e,'native_dynamic_lag_and_counters',events)
    return dict(status='PASS_K2_NATIVE_CPU',native=native.metadata,cases=cases,
                state_checks=checks,dynamic_widths=widths,cuda_forward=False)
