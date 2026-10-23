"""Real native scheduler contract gate using deterministic CPU model tickets."""
import hashlib,importlib,json,sys
from pathlib import Path
from types import ModuleType,SimpleNamespace


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_k4_cpu_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    package=ModuleType(tag);package.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=package
    try:
        batch=importlib.import_module(tag+'.kaifra_batch');nm=importlib.import_module(tag+'.native_scheduler')
        spec=importlib.import_module(tag+'.kaifra_speculative')
        # Import fixture from this sealed overlay, never the resident package.
        fixture=importlib.util.spec_from_file_location(tag+'_fixture',root/'scripts/k2_native_validate.py')
        mod=importlib.util.module_from_spec(fixture);fixture.loader.exec_module(mod)
        class Event:
            def query(self):return True
            def synchronize(self):pass
        class Target(mod.ToyTarget):
            capabilities={}
            def __init__(self,torch):super().__init__(torch);self.inflight=None;self.leased=[];self.joint=0
            def state(self):
                st=super().state();self.leased.append(st);return st
            def _slot(self,st):return next(i for i,s in enumerate(self.leased) if s is st)
            def release_state(self,st):
                assert self.inflight is None;self.leased.pop(self._slot(st))
            def launch(self,requests,diagnostic=False):
                assert self.inflight is None
                self.joint+=len(requests)==2;windows=[];fts=[];lgs=[];at=0
                for st,ids in requests:
                    lg,ft,_=self.run(st,ids);windows.append(SimpleNamespace(state=st,ids=ids,at=at));fts.append(ft);lgs.append(lg);at+=len(ids)
                lg=self.torch.cat(lgs)
                self.inflight=SimpleNamespace(windows=windows,features=self.torch.cat(fts),full_logits=lg,
                    cpu_predictions=lg.argmax(-1),head_info={'fallback':True},event=Event(),started=__import__('time').perf_counter())
                return self.inflight
            def poll(self,ticket):assert ticket is self.inflight;return True
            def commit(self,*args):
                if len(args)==4:return super().commit(*args)
                ticket,keeps=args;assert ticket is self.inflight
                for win in ticket.windows:
                    k=keeps.get(id(win.state),0);super().commit(win.state,(win.ids,),len(win.ids),k)
                return Event()
            def retire(self,ticket):assert ticket is self.inflight;self.inflight=None
            discard=retire
        class Provider(batch.RequestDraftProvider):
            deferred_draft=True;model=object();budget=object();max_context=256
            def __init__(self,fault):self.fault=fault;self.widths={}
            def release(self,handle):pass
            def propose(self,handle,state,anchor,remaining):
                q=min(remaining,(1,3,6,8)[(state.output_tokens+(handle.request_id=='b'))%4]);state.next_width=q
                self.widths.setdefault(handle,[]).append(q);state.cumulative=(.9,)*7
                ids=[(anchor+i+1)%100 for i in range(7)]
                if self.fault is not None:ids[self.fault]=(ids[self.fault]+10)%100
                return ids,[.9]*7
        cases=[]
        for depth in (0,7):
            with nm.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=depth) as native:
                for fault in ((None,) if not depth else (None,0,3,6)):
                    target=Target(native.torch);provider=Provider(fault)
                    def make():return batch.batch_engine(native,target,provider,eos_ids=(99,),prior=[.7]*7,capacity=128,target_only=not depth)
                    with make() as engine:
                        hs=[engine.submit_tokens('a',[1],max_tokens=24),engine.submit_tokens('b',[11],max_tokens=17)];out={h:[] for h in hs}
                        traces=[]
                        while engine.active_requests:
                            for ev in engine.step():out[ev.handle].extend(ev.token_ids)
                            traces.extend(engine.last_step_metrics)
                        assert out[hs[0]]==list(range(2,26)),out
                        assert out[hs[1]]==list(range(12,29)),out
                        assert target.joint>0
                        assert not target.leased and not target.inflight
                        assert not engine.bridge.native.has_unfinished_seqs()
                        assert engine.bridge.blocks.get_num_free_blocks()==engine.bridge.block_count-1
                        cases.append(dict(depth=depth,fault=fault,joint=target.joint,widths=sorted({m['verified_rows'] for m in traces if not m['prompt']})))
                    for phase in ('waiting','prepared_prompt','inflight_decode'):
                        with make() as engine:
                            a=engine.submit_tokens('reuse',[1],max_tokens=12);b=engine.submit_tokens('b',[11],max_tokens=12);out={a:[],b:[]}
                            if phase=='inflight_decode':
                                while not all(out.values()):
                                    for ev in engine.step():out[ev.handle].extend(ev.token_ids)
                            prefix=list(out[a]);plan=None
                            if phase!='waiting':
                                plan=engine.bridge.prepare();engine.bridge.execute_async(plan)
                            engine.cancel(a)
                            if plan is not None:
                                while not engine.bridge.poll(plan):pass
                                for ev in engine.finish_plan(plan):out[ev.handle].extend(ev.token_ids)
                            for ev in engine.drain():out[ev.handle].extend(ev.token_ids)
                            assert out[a]==prefix and out[b]==list(range(12,24)),out
                            c=engine.submit_tokens('reuse',[97],max_tokens=8)
                            assert c.generation>a.generation
                            try:engine.cancel(a)
                            except batch.BridgeError:pass
                            else:raise AssertionError('stale cancellation accepted')
                            events=list(engine.drain());assert [t for ev in events for t in ev.token_ids]==[98,99]
                            assert events[-1].finish_reason=='stop'
                            assert not target.leased and not target.inflight
                    print(json.dumps(cases[-1]),flush=True)
        return dict(status='PASS_K4_BATCH_NATIVE_CPU',cases=cases,source_sha256=manifest,cuda_forward=False,model_reloaded=False)
    finally:
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.'):sys.modules.pop(n,None)
