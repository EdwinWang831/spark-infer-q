"""Instrumented Target Graph stream intervals; separate from service timing."""
import hashlib,importlib,json,sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_k4_target_breakdown_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    torch=session.torch;target=None
    report=dict(status='RUNNING',source_sha256=manifest,model_reloaded=False,records=[],
        contract=dict(scope='instrumented Target Graph only, not service throughput',repetitions=5,
            timing='nonoverlapping CUDA-event stream intervals; includes graph event overhead, not summed active kernel time',
            excluded='prefill, draft, E17, head, commit and staging are outside measured model-loop intervals'))
    def save(active):
        report['active']=active;p=root/'target-breakdown.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    try:
        from e3_wide_target import WideTarget
        cm=importlib.import_module(tag+'.compact_target');hm=importlib.import_module(tag+'.screened_head_fused')
        loader=importlib.import_module(tag+'.qwen_state_kernel');cpu=importlib.import_module(tag+'.worker_cpu')
        qa=importlib.import_module(tag+'.qwen_attention');qp=importlib.import_module(tag+'.qwen_ple')
        component=json.loads(Path('/workspace/e1/k4batch-20261023-target-component-v6/target-component.json').read_text())
        ext=loader.load_state_kernel(component['kernel']['library'],component['kernel']['binary_sha256'])
        target=cm.CompactTarget(WideTarget(session.weights,capacity=2048),hm.FusedScreenedHead(session.weights.head),ext)
        target.warmup();oracle=WideTarget(session.weights,capacity=2048);states=[target.state(),target.state()]
        texts=['Write a Python function to remove duplicates while preserving order. Return only code.',
            'Read this context: '+('A cache stores recent values; a queue orders pending work. '*25)+' Explain the difference in one sentence.']
        seeds=[]
        for st,text in zip(states,texts):
            ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            ref=oracle.state()
            for at in range(0,len(ids),8):oracle.step(ref,ids[at:at+8])
            st.copy_from(ref);seeds.append(ref)
        pending=[];snapshots={};depth=0
        def wrap(fn,name):
            def call(*args,**kwargs):
                nonlocal depth
                if depth:return fn(*args,**kwargs)
                start=torch.cuda.Event(enable_timing=True,external=True)
                end=torch.cuda.Event(enable_timing=True,external=True)
                start.record();depth+=1
                try:return fn(*args,**kwargs)
                finally:
                    depth-=1;end.record();pending.append((name,start,end))
            return call
        class Proxy:
            def __init__(self,base,labels):self.base,self.labels=base,labels
            def __getattr__(self,name):
                fn=getattr(self.base,name)
                return wrap(fn,self.labels[name]) if name in self.labels else fn
        target.forward=Proxy(target.forward,dict(hc_block='HC',moe_block='MoE',_mm='input_projection',
            _out_proj='output_projection',_embed='embedding',_writeback='feature_writeback',finish='finish'))
        target.gdn_io=Proxy(target.gdn_io,dict(front='GDN_front',back='GDN_back'))
        target.state_kernel=Proxy(target.state_kernel,dict(forward='GDN_recurrence'))
        qa.attention=wrap(qa.attention,'QSA_attention');qp.ple_block=wrap(qp.ple_block,'PLE')
        compute=target._compute
        def instrumented(segments,b,rows,context=None):
            nonlocal pending
            pending=[];begin=torch.cuda.Event(enable_timing=True,external=True);end=torch.cuda.Event(enable_timing=True,external=True)
            begin.record();result=compute(segments,b,rows,context);end.record()
            snapshots[rows,context]=(begin,end,pending);return result
        target._compute=instrumented
        shapes=((3,5),(8,8),(3,0),(0,8))
        def trial(shape,diagnostic=False):
            for st,seed in zip(states,seeds):st.copy_from(seed)
            active=[i for i,n in enumerate(shape) if n]
            windows=[(states[i],[61+8*i+j for j in range(shape[i])]) for i in active]
            ticket=target.launch(windows,diagnostic=diagnostic)
            while not target.poll(ticket):ticket.event.synchronize()
            expected={}
            if diagnostic:
                for i,win in zip(active,ticket.windows):
                    ref=seeds[i].clone();lg,ft,b=oracle.run(ref,list(win.ids));a=win.at;n=len(win.ids)
                    assert torch.equal(lg,ticket.full_logits[a:a+n]) and torch.equal(ft,ticket.features[a:a+n])
                    oracle.commit(ref,b,n,min(2,n));expected[i]=ref
            context=min(next(n for n in (64,128,256,512,1024,2048) if n>=max(st.pos+len(ids) for st,ids in windows)),target.capacity)
            begin,end,records=snapshots[sum(shape),context]
            grouped=defaultdict(float);counts=defaultdict(int)
            for name,a,b in records:grouped[name]+=a.elapsed_time(b);counts[name]+=1
            total=begin.elapsed_time(end)
            target.commit(ticket,{id(win.state):min(2,len(win.ids)) for win in ticket.windows}).synchronize();target.retire(ticket)
            for i,ref in expected.items():assert not oracle.state_difference(states[i],ref)
            return dict(shape=shape,model_loop_ms=total,category_ms=dict(grouped),category_calls=dict(counts),
                unattributed_ms=total-sum(grouped.values()),exact=True)
        with cpu.worker_cpu('performance'):
            for shape in shapes:
                save('warmup:'+str(shape))
                for _ in range(3):trial(shape,True)
            for rep in range(5):
                for shape in shapes:
                    row=trial(shape);row['repetition']=rep;report['records'].append(row)
                save('timing:'+str(rep))
        report.update(status='PASS_K4_TARGET_BREAKDOWN',numerical_checks=True);save('done');return report
    except Exception:
        report['status']='FAIL';save(report.get('active'));raise
    finally:
        torch.cuda.synchronize()
        if target is not None:
            if target.inflight is not None:target.discard(target.inflight)
            for st,leased in zip(target.states,target.leased):
                if leased:target.release_state(st)
            target.close()
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.'):sys.modules.pop(n,None)
