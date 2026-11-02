"""Target-only joint execution over the existing native DeepInfer scheduler."""
import time
from .deepinfer_bridge import BridgeError,SchedulerBridge
from .kaifra_runtime import KaifraEngine,QwenTargetBackend


class JointTargetBackend(QwenTargetBackend):
    def __init__(self,*args,split_decode=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.split_decode=split_decode

    def release(self,handle,state):
        if self.states.get(handle) is not state:raise BridgeError('foreign or released state')
        self.target.release_state(state)
        del self.states[handle]

    def advance_joint(self,work):
        if not 1<=len(work)<=2:raise BridgeError('one or two decode rows required')
        if self.split_decode and len(work)>1:
            result={}
            for item in work:result.update(self.advance_joint([item]))
            return result
        for row,state in work:
            if self.states.get(row.handle) is not state or state.pos!=row.position:
                raise BridgeError('joint state ownership/position mismatch')
            if row.prompt or len(row.token_ids)!=1:
                raise BridgeError('target-only joint rows must each contain one anchor')
        target=self.target
        with target.torch.inference_mode():
            ticket=target.launch([(st,row.token_ids) for row,st in work],diagnostic=self.observer is not None)
            try:
                deadline=time.monotonic()+60
                while not target.poll(ticket):
                    if time.monotonic()>deadline:raise BridgeError('joint target deadline')
                    time.sleep(.0001)
                # Window order belongs to the target; map by owned state rather than input order.
                windows={id(w.state):w for w in ticket.windows}
                tokens={row.handle:int(ticket.cpu_predictions[windows[id(st)].at]) for row,st in work}
                event=target.commit(ticket,{id(st):1 for _,st in work})
                event.synchronize()
                if self.observer is not None:
                    for row,st in work:
                        w=windows[id(st)]
                        self.observer(row.handle,row.position,list(row.token_ids),
                            ticket.full_logits[w.at:w.at+1],st)
                target.retire(ticket)
                return tokens
            except Exception:
                if target.inflight is ticket:target.discard(ticket)
                raise


class JointSchedulerBridge(SchedulerBridge):
    def execute(self,plan):
        self._plan(plan)
        if plan.executed:raise BridgeError('plan already executed')
        try:
            predictions={};decode=[]
            for row in plan.rows:
                entry=self._entry(row.handle)
                if entry['cancelled']:predictions[row.handle]=0
                elif row.prompt:predictions[row.handle]=self.backend.advance(row,entry['state'])
                else:decode.append((row,entry['state']))
            if decode:predictions.update(self.backend.advance_joint(decode))
            for row in plan.rows:
                token=predictions[row.handle];entry=self._entry(row.handle)
                if type(token) is not int or not 0<=token<self.vocab_size:
                    raise BridgeError('invalid joint backend token')
                if not entry['cancelled'] and self.backend.position(entry['state'])!=row.position+len(row.token_ids):
                    raise BridgeError('joint backend position did not match plan')
            plan.predictions=tuple(predictions[r.handle] for r in plan.rows)
            plan.executed=True
        except Exception:
            self.poisoned=True
            raise


def joint_engine(native,target,*,eos_ids,capacity=256,token_budget=32,tokenizer=None,observer=None,joint_decode=True):
    if getattr(native,'max_drafts',None)!=0:raise ValueError('joint target-only engine requires NEXTN=0')
    if capacity>target.capacity:raise ValueError('logical capacity exceeds physical target')
    backend=JointTargetBackend(target,prefill_rows=8,observer=observer,split_decode=not joint_decode)
    bridge=JointSchedulerBridge(native.core,native.torch,native.group,native.sampling_class,
        backend,vocab_size=int(target.w.head.b.weight.shape[0]),eos_ids=eos_ids,
        capacity=capacity,max_requests=2,token_budget=token_budget)
    engine=KaifraEngine(bridge,tokenizer=tokenizer)
    engine.capabilities=dict(engine.capabilities,**target.capabilities)
    engine.capabilities.update(target_only=True,speculative=False,asynchronous=False,
        gpu_batch=True,max_requests=2,joint_decode_policy=joint_decode)
    return engine
