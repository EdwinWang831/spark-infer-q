"""Actual native scheduler lifecycle checks, optionally with the full Qwen target."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def require_error(fn):
    try: fn()
    except (ValueError, RuntimeError): return
    raise AssertionError('expected rejection')


def suite(make_bridge, backend_factory, prompts, *, eos_ids):
    results=[]
    def start():
        backend=backend_factory(); bridge=make_bridge(backend)
        return bridge,backend
    def add(b,be,name,kind,limit):
        ids=prompts[kind];be.register(name,ids,limit)
        return b.add(name,ids,limit)
    def expected(b,be,h):
        assert b.completed[h]['token_ids']==be.oracles[h.request_id]['tokens']
        assert not b.completed[h]['cancelled']
    def record(name,b,be):
        b.close()
        assert not b.requests and not b.native.has_unfinished_seqs()
        results.append(dict(case=name,status='PASS',events=b.events,
            completed=[dict(slot=h.slot,generation=h.generation,request_id=h.request_id,**v)
                for h,v in b.completed.items()],
            free_native_blocks=b.blocks.get_num_free_blocks(),
            total_native_blocks=b.block_count-1,backend_checks=be.summary()))
        print(json.dumps(dict(completed=name)),flush=True)
    b,be=start();h=add(b,be,'single','arithmetic',32)
    p=b.prepare();require_error(lambda:b.complete(p));require_error(b.prepare)
    b.execute(p);require_error(lambda:b.execute(p));b.complete(p)
    require_error(lambda:b.complete(p));b.drain();expected(b,be,h)
    assert b.completed[h]['token_ids'][-1] in eos_ids
    record('single_eos_and_plan_fences',b,be)

    b,be=start();a=add(b,be,'json','json',48);c=add(b,be,'code','code',48)
    b.drain();expected(b,be,a);expected(b,be,c)
    assert any(e['request_id']=='json' and not e['prompt'] for e in b.events)
    assert any(e['request_id']=='code' and not e['prompt'] for e in b.events)
    assert any(e['prompt'] and e['published']==0 for e in b.events)
    record('two_requests_chunked_prompt_and_interleaved_decode',b,be)

    b,be=start();h=add(b,be,'capped','code',3);b.drain();expected(b,be,h)
    assert len(b.completed[h]['token_ids'])==3
    record('length_cap_prefix_only',b,be)

    for phase in ('waiting','prepared_prompt','prepared_decode'):
        b,be=start();h=add(b,be,'reuse','json',48)
        if phase=='prepared_decode':
            while not b._entry(h)['seq'].data.get_len()>len(prompts['json']):b.step()
        if phase!='waiting':
            p=b.prepare();assert any(r.handle==h for r in p.rows)
            if phase=='prepared_decode':assert not p.rows[0].prompt
            b.execute(p)
            # Completed backend work is still private until complete().
            before=len(b._entry(h)['seq'].data.get_token_ids())
        b.cancel(h)
        require_error(lambda:b.add('reuse',prompts['arithmetic'],32))
        if phase!='waiting':
            b.complete(p)
            assert len(b.completed[h]['token_ids'])==before-len(prompts['json'])
            require_error(lambda:b.complete(p))
        b.drain();assert b.completed[h]['cancelled']
        new=add(b,be,'reuse','arithmetic',32)
        assert new.slot==h.slot and new.generation>h.generation
        require_error(lambda:b.cancel(h))
        b.drain();expected(b,be,new)
        record('cancel_'+phase+'_and_generation_reuse',b,be)

    b,be=start()
    for ids,limit in (([],1),([-1],1),([b.vocab_size],1),([1],0),([1]*b.capacity,1)):
        require_error(lambda ids=ids,limit=limit:b.add('bad',ids,limit))
    a=add(b,be,'a','arithmetic',32);c=add(b,be,'b','arithmetic',32)
    require_error(lambda:b.add('extra',prompts['arithmetic'],32))
    require_error(lambda:b.add('a',prompts['arithmetic'],32))
    b.cancel(a);b.cancel(c);b.drain();record('invalid_admission_and_full_slots',b,be)
    return results


class ToyBackend:
    def __init__(self):
        self.oracles={};self.checks=0;self.releases=0
    def register(self,name,prompt,limit):
        history=list(prompt);tokens=[]
        for _ in range(limit):
            token=99 if history==[9] else sum(history)*7%97
            tokens.append(token)
            if token==99:break
            history.append(token)
        self.oracles[name]=dict(tokens=tokens,inputs=list(prompt)+tokens[:-1])
    def create(self,handle):return dict(handle=handle,history=[])
    def position(self,state):return len(state['history'])
    def advance(self,row,state):
        expected=self.oracles[row.handle.request_id]['inputs'][row.position:row.position+len(row.token_ids)]
        assert list(row.token_ids)==expected
        state['history'].extend(row.token_ids);self.checks+=1
        return 99 if state['history']==[9] else sum(state['history'])*7%97
    def release(self,handle,state):self.releases+=1
    def summary(self):return dict(input_prefix_checks=self.checks,released_states=self.releases)


def checkpoint_inputs(checkpoint):
    from transformers import AutoTokenizer
    from e4_tasks import TASKS
    tok=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    prompts={name:tok.apply_chat_template([dict(role='user',content=next(t['prompt'] for t in TASKS if t['id']==task))],
        tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        for name,task in [('arithmetic','arithmetic_direct'),('json','json_direct'),('code','positive_direct')]}
    raw=json.loads((checkpoint/'generation_config.json').read_text())['eos_token_id']
    eos=set(raw if isinstance(raw,list) else [raw]);eos.add(tok.eos_token_id)
    config=json.loads((checkpoint/'config.json').read_text())
    vocab=config.get('text_config',config)['vocab_size']
    assert all(type(ids) is list and ids and all(type(t) is int and 0<=t<vocab for t in ids) for ids in prompts.values())
    assert all(type(t) is int and 0<=t<vocab for t in eos)
    return prompts,eos,vocab


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--cpu-only',action='store_true')
    p.add_argument('--replay-e4',type=Path);a=p.parse_args()
    if a.replay_e4 and not a.cpu_only:raise ValueError('replay preflight is CPU only')
    root=a.root;e1=root.parent
    report=dict(status='RUNNING',kind='cpu_native_scheduler' if a.cpu_only else 'full_target_native_scheduler',
        limits=['bounded synchronous bridge; no native Worker, GPU sampler, speculation or async pipeline',
            'native block IDs are logical accounting only; Qwen owns actual GDN/QSA state',
            'multiple scheduled requests execute serially on GPU; not a batched throughput benchmark',
            'serial prompt arithmetic retained; prefill numerical discrepancy remains unresolved'],cases=[])
    artifact=root/'artifacts'/('replay.json' if a.replay_e4 else 'cpu.json' if a.cpu_only else 'gpu.json')
    def save():artifact.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    save();started=time.time()
    try:
        os.environ.update(DEEPINFER_USE_NEXTN_DECODE='0',DEEPINFER_ACCELERATOR='nvidia',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        sys.path[:0]=[str(root/'followup-deps'),str(root/'python-deps'),str(root),str(e1/'DeepInfer')]
        import torch
        import torch.distributed as dist
        import deepinfer
        deepinfer.__path__.insert(0,str(e1/'build-release/lib/deepinfer'))
        import deepinfer._C as extension
        from deepinfer._C import engine as core
        from deepinfer.sampling_params import SamplingParams
        from spark_infer_q.deepinfer_bridge import SchedulerBridge
        assert torch.__version__=='2.13.0+cu132'
        sha=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()
        assert sha=='f6aa14b790eb7bc729251eea152fd9c83cf215bc5ab912e25ee19c57d8fc1768'
        rev=subprocess.check_output(['git','-c','safe.directory='+str(e1/'DeepInfer'),
            '-C',str(e1/'DeepInfer'),'rev-parse','HEAD'],text=True).strip()
        assert rev=='774ce244b87cc0eb8d4b94ab9f5de2ab22049f01'
        report.update(core_sha256=sha,upstream_revision=rev,torch_version=torch.__version__,
            harness_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in ('e5_bridge_smoke.py','spark_infer_q/deepinfer_bridge.py','e3_target.py','e3_wide_target.py')})
        with tempfile.TemporaryDirectory(prefix='gloo-',dir=root) as td:
            dist.init_process_group('gloo',init_method='file://'+td+'/store',rank=0,world_size=1)
            try:
                if a.cpu_only:
                    if a.replay_e4:
                        prompts,eos,vocab=checkpoint_inputs(a.checkpoint)
                        old=json.loads(a.replay_e4.read_text())
                        frozen={t['id']:t['correctness'] for t in old['tasks']}
                        token_map={tuple(prompts[k]):frozen[v]['token_ids'] for k,v in
                            [('arithmetic','arithmetic_direct'),('json','json_direct'),('code','positive_direct')]}
                        for k,v in [('arithmetic','arithmetic_direct'),('json','json_direct'),('code','positive_direct')]:
                            assert prompts[k]==frozen[v]['prompt_ids']
                        class Replay(ToyBackend):
                            def register(self,name,prompt,limit):
                                tokens=token_map[tuple(prompt)][:limit]
                                self.oracles[name]=dict(tokens=tokens,inputs=list(prompt)+tokens[:-1],prompt_length=len(prompt))
                            def advance(self,row,state):
                                oracle=self.oracles[row.handle.request_id]
                                assert list(row.token_ids)==oracle['inputs'][row.position:row.position+len(row.token_ids)]
                                state['history'].extend(row.token_ids);self.checks+=1
                                offset=len(state['history'])-oracle['prompt_length']
                                return 0 if offset<0 else oracle['tokens'][offset]
                        factory=Replay;capacity=256
                        report['cpu_input_mode']='real tokenizer and frozen E4 output replay; no GPU forward'
                    else:
                        prompts=dict(arithmetic=[9],json=list(range(1,40)),code=list(range(40,80)))
                        eos={99};vocab=128;factory=ToyBackend;capacity=128
                else:
                    if a.checkpoint is None:raise ValueError('checkpoint required')
                    # Validate real tokenization before paying for a full model load.
                    prompts,eos,vocab=checkpoint_inputs(a.checkpoint)
                    from e3_target import load_target
                    from e3_wide_target import WideTarget
                    weights,load=load_target(root,a.checkpoint,lambda event:print(json.dumps(event),flush=True))
                    report['target']=load;save()
                    target=WideTarget(weights,capacity=256)
                    # tokenizer.vocab_size omits added special tokens such as EOS.
                    # The scheduler must use the executed model head's row count.
                    assert vocab==int(weights.head.b.weight.shape[0]);capacity=256
                    class GPUBackend:
                        def __init__(self):
                            self.oracles={};self.shadow={};self.rows=0;self.states=0;self.releases=[]
                        def register(self,name,prompt,limit):
                            st=target.state();logits=[];inputs=list(prompt);output=[]
                            with torch.inference_mode():
                                for token in prompt:
                                    lg,_=target.step(st,[token],capture=False);logits.append(lg.cpu())
                                for i in range(limit):
                                    token=int(lg[-1].argmax());output.append(token)
                                    if token in eos or i==limit-1:break
                                    inputs.append(token);lg,_=target.step(st,[token],capture=False);logits.append(lg.cpu())
                            self.oracles[name]=dict(tokens=output,inputs=inputs,logits=logits)
                        def create(self,h):
                            self.shadow[h]=target.state();return target.state()
                        def position(self,st):return st.pos
                        def advance(self,row,st):
                            oracle=self.oracles[row.handle.request_id];start=row.position
                            assert list(row.token_ids)==oracle['inputs'][start:start+len(row.token_ids)]
                            with torch.inference_mode():
                                for j,token in enumerate(row.token_ids):
                                    lg,_=target.step(st,[token],capture=False)
                                    assert torch.equal(lg.cpu(),oracle['logits'][start+j])
                                    shadow_lg,_=target.step(self.shadow[row.handle],[oracle['inputs'][start+j]],capture=False)
                                    assert torch.equal(lg,shadow_lg);self.rows+=1
                                assert not target.state_difference(st,self.shadow[row.handle]);self.states+=1
                            return int(lg[-1].argmax())
                        def release(self,h,st):
                            self.releases.append(dict(slot=h.slot,generation=h.generation,last_device_position=st.pos))
                            del self.shadow[h]
                        def summary(self):return dict(full_logit_rows=self.rows,live_state_checks=self.states,released_states=self.releases)
                    factory=GPUBackend
                def make(backend):
                    return SchedulerBridge(core,torch,dist.group.WORLD,SamplingParams,backend,
                        vocab_size=vocab,eos_ids=eos,capacity=capacity,max_requests=2,token_budget=32)
                report['cases']=suite(make,factory,prompts,eos_ids=eos)
                if a.cpu_only:assert not torch.cuda.is_initialized()
                else:
                    torch.cuda.synchronize()
                    report['torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                    report['torch_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
                report.update(status='PASS_BOUNDED_NATIVE_SCHEDULER_BRIDGE',cuda_initialized=torch.cuda.is_initialized())
            finally:dist.destroy_process_group()
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds']=time.time()-started;save()


if __name__=='__main__':main()
