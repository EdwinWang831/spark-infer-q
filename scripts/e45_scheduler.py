"""Reproduce the native watermark stall, then test near-capacity progress.

Real pinned native scheduler, deterministic host backend; no model/GPU timing.
"""
from types import SimpleNamespace


class Tokens:
    def create(self,handle):return []
    def position(self,state):return len(state)
    def advance(self,row,state):
        state.extend(row.token_ids)
        return 3
    def release(self,handle,state):pass


def check(native,bridge_class):
    records=[]
    for old,capacity,slots,length in ((True,4352,1,4096),(False,4352,1,4096),
                                      (False,4352,1,4288),(False,4352,2,4288),
                                      (False,2048,1,1984)):
        core=native.core
        old_count=1+slots*((capacity+15)//16+2)
        if old:
            core=SimpleNamespace(**{n:getattr(core,n) for n in
                ('SequenceMetadataPool','Scheduler','Sequence','SequenceStatus')},
                BlockManager=lambda _:native.core.BlockManager(old_count))
        bridge=bridge_class(core,native.torch,native.group,native.sampling_class,Tokens(),
            vocab_size=100,eos_ids=(99,),capacity=capacity,max_requests=slots,token_budget=1024)
        if old:bridge.block_count=old_count
        record=dict(legacy_pool=old,capacity=capacity,slots=slots,prompt_tokens=length,
                    block_count=bridge.block_count,steps=0,idle_steps=0,prompt_rows=0)
        try:
            handles=[bridge.add(str(i),[1]*length,64) for i in range(slots)]
            limit=8 if old else slots*((length+1023)//1024+64)+4
            for _ in range(limit):
                if not bridge.requests:break
                plan=bridge.step();record['steps']+=1
                record['idle_steps']+=not bool(plan.rows)
                record['prompt_rows']+=sum(len(r.token_ids) for r in plan.rows if r.prompt)
            record.update(active=len(bridge.requests),used_blocks=bridge.block_count-bridge.blocks.get_num_free_blocks())
            if old:
                assert record['prompt_rows']==1024 and record['idle_steps']==7 and len(bridge.requests)==1,record
                assert record['used_blocks']/record['block_count']>.9,record
                bridge.cancel(handles[0])
            else:
                assert not bridge.requests and record['idle_steps']==0,record
                assert record['prompt_rows']==length*slots,record
                assert all(bridge.completed[h]['token_ids']==[3]*64 for h in handles),record
            bridge.close()
            record['blocks_recovered']=bridge.blocks.get_num_free_blocks()==bridge.block_count-1
            assert record['blocks_recovered'];records.append(record)
        finally:
            if not bridge.closed:bridge.close()
    return records
