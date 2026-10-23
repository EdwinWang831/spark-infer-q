"""Resident W8A16 draft kernel and shared-prefix proposer validation."""
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
import time


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8<<20):h.update(b)
    return h.hexdigest()


def backend(root):
    build=json.loads((root/'artifacts/e28-build.json').read_text())
    assert sha(root/'draft_w8_kernel.cu')==build['source_sha256']
    sys.path.insert(0,str(root/'build/draft_w8'))
    ext=importlib.import_module('spark_q_draft_w8')
    assert sha(Path(ext.__file__))==build['binary_sha256']
    return ext,build


def run(session,arguments):
    import torch
    from spark_infer_q.draft_backbone import quantize_w8,attach_mlp_w8
    from spark_infer_q.confidence_features import ConfidenceDraft
    stage=arguments['stage'];assert stage in ('kernel','proposer')
    root=session.root;ext,identity=backend(root)
    out=root/f'artifacts/e28-{stage}.json';assert not out.exists()
    sources=['e28_probe.py','draft_w8_kernel.cu','spark_infer_q/draft_backbone.py',
        'spark_infer_q/dflash2_reference.py','spark_infer_q/dflash2_block.py',
        'spark_infer_q/draft_context.py','spark_infer_q/confidence_features.py','spark_infer_q/draft_head.py']
    hashes={n:sha(root/n) for n in sources}
    report=dict(status='RUNNING',stage=stage,started_unix=time.time(),source_sha256=hashes,identity=identity,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),cases=[],
        scope='MLP gate/up/down only; original target, draft Q4 head, BF16 selector/attention unchanged; no whole-model memory saving while BF16 controls coexist')
    def save():
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    save()
    assert torch.cuda.get_device_capability()==(12,1) and not torch.backends.cuda.matmul.allow_tf32
    w8=lambda x,p:ext.matmul(x,p.values,p.scales)
    try:
        if stage=='kernel':
            report['checkpoint_sha256']=sha(root/'checkpoint/model.safetensors')
            assert report['checkpoint_sha256']=='d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f'
            generator=torch.Generator(device='cuda').manual_seed(281023)
            for n,k in [(259,37),(128,2560)]:
                weight=(torch.randn((n,k),generator=generator,device='cuda',dtype=torch.bfloat16)*.02).contiguous()
                packed=quantize_w8(weight);dequant=packed.values.float()*packed.scales[:,None]
                for rows in (1,7,8):
                    x=torch.randn((rows,k),generator=generator,device='cuda',dtype=torch.bfloat16)
                    for stream_name in ('default','separate'):
                        stream=torch.cuda.current_stream() if stream_name=='default' else torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            got=ext.matmul(x,packed.values,packed.scales,True)
                            want=x.float()@dequant.T
                        stream.synchronize()
                        rel=float(torch.linalg.vector_norm(got-want)/torch.linalg.vector_norm(want))
                        absolute=float((got-want).abs().max())
                        assert torch.isfinite(got).all() and rel<3e-5 and absolute<1e-3,(n,k,rows,rel,absolute)
                        bf=w8(x,packed);assert bf.dtype==torch.bfloat16 and torch.equal(bf,got.to(torch.bfloat16))
                        report['cases'].append(dict(n=n,k=k,rows=rows,stream=stream_name,stored_weight_relative_l2=rel,max_abs=absolute))
                        save()
            report['real_matrices']=[]
            for name,weight in [('gate_up',session.draft.fused['layers.0.gate_up']),('down',session.draft.weights['layers.0.mlp.down_proj.weight'])]:
                p=quantize_w8(weight);x=torch.randn((8,weight.shape[1]),generator=generator,device='cuda',dtype=torch.bfloat16)
                ref=x.float()@(p.values.float()*p.scales[:,None]).T;actual=ext.matmul(x,p.values,p.scales,True)
                rel=float(torch.linalg.vector_norm(actual-ref)/torch.linalg.vector_norm(ref));assert rel<3e-5
                bf16=torch.nn.functional.linear(x,weight);got=w8(x,p)
                distortion=float(torch.linalg.vector_norm(got.float()-bf16.float())/torch.linalg.vector_norm(bf16.float()))
                timings={m:[] for m in ('bf16','w8')}
                fns=dict(bf16=lambda:torch.nn.functional.linear(x,weight),w8=lambda:w8(x,p))
                for fn in fns.values():
                    for _ in range(5):fn()
                for rep in range(6):
                    for mode in (('bf16','w8') if rep%2==0 else ('w8','bf16')):
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):fns[mode]()
                        end.record();end.synchronize();timings[mode].append(start.elapsed_time(end)/20)
                report['real_matrices'].append(dict(name=name,shape=list(weight.shape),rows=8,stored_weight_relative_l2=rel,
                    relative_l2_vs_bf16=distortion,packed_bytes=p.nbytes(),bf16_bytes=weight.numel()*weight.element_size(),
                    gpu_ms={m:statistics.median(v) for m,v in timings.items()},timing_repeats_ms=timings))
                save()
            report['status']='PASS_E28_W8_KERNEL'
        else:
            proof=json.loads((root/'artifacts/e28-kernel.json').read_text());assert proof['status']=='PASS_E28_W8_KERNEL'
            from e3_wide_target import WideTarget
            from spark_infer_q.prompt import prompt_state
            from spark_infer_q.draft_head import HeadDraft
            target=WideTarget(session.weights,capacity=2048)
            base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
            dense=ConfidenceDraft.attach(base);quant=attach_mlp_w8(base,w8)
            report['packed_weights']=quant.w8_manifest
            report['incremental_packed_bytes']=sum(v['packed_bytes'] for v in quant.w8_manifest.values())
            tasks=[dict(id='en-short',prompt='Explain in two sentences why caching can reduce latency.',language='en'),
                dict(id='zh-short',prompt='用两句话解释为什么缓存可以降低延迟。',language='zh'),
                dict(id='en-long',prompt='Read this ledger:\n'+'\n'.join(f'Entry {i}: quantity {i%13+1}, price {i%17+1}.' for i in range(180))+'\nCompute the cost of the first two entries.',language='en',target_prompt_tokens=768),
                dict(id='zh-long',prompt='阅读以下记录：\n'+'\n'.join(f'记录{i}：数量{i%13+1}，单价{i%17+1}。' for i in range(180))+'\n计算前两条记录的总金额。',language='zh',target_prompt_tokens=1024)]
            for task in tasks:
                ids=session.tokenizer.apply_chat_template([dict(role='user',content=task['prompt'])],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
                if 'target_prompt_tokens' in task:
                    n=task['target_prompt_tokens'];assert len(ids)>n;ids=ids[:n-128]+ids[-128:]
                state,ctx,lg=prompt_state(target,ids,chunk_size=8);anchor=int(lg[-1].argmax())
                dense.begin((task['id'],'dense'));quant.begin((task['id'],'quant'))
                h0,_=dense.forward_hidden(ctx,anchor);h1,_=quant.forward_hidden(ctx,anchor)
                rel=float(torch.linalg.vector_norm(h1.float()-h0.float())/torch.linalg.vector_norm(h0.float()))
                p0,_=dense.propose(ctx,anchor);p1,_=quant.propose(ctx,anchor)
                v0,c0=session.projection.topk(h0[1:]);v1,c1=session.projection.topk(h1[1:])
                overlap=sum(len(set(a)&set(b))/16 for a,b in zip(c0.tolist(),c1.tolist()))/7
                times={m:[] for m in ('bf16','w8')}
                # begin() resets append-cache identically; otherwise repeated-prefix cache reuse is matched.
                for rep in range(6):
                    for mode,obj in ((('bf16',dense),('w8',quant)) if rep%2==0 else (('w8',quant),('bf16',dense))):
                        obj.begin((task['id'],mode,rep));obj.propose(ctx,anchor)
                        torch.cuda.synchronize();t=time.perf_counter()
                        for _ in range(5):obj.propose(ctx,anchor)
                        torch.cuda.synchronize();times[mode].append((time.perf_counter()-t)*1000/5)
                report['cases'].append(dict(id=task['id'],language=task['language'],prompt_tokens=len(ids),hidden_relative_l2=rel,
                    proposal_equal=sum(a==b for a,b in zip(p0,p1)),proposal_total=7,top16_mean_overlap=overlap,
                    proposer_ms={m:statistics.median(v) for m,v in times.items()},repeats_ms=times))
                assert torch.isfinite(h1).all() and len(p1)==7 and dense.weights is not quant.weights and dense.fused is not quant.fused
                save();del state,ctx,lg,h0,h1
            report['status']='PASS_E28_PROPOSER_DIAGNOSTIC'
        for n,h in hashes.items():assert sha(root/n)==h,n
        assert session.draft.max_context==1024
        report['finished_unix']=time.time();save()
        return dict(status=report['status'],artifact=out.name,target_object_id=id(session.weights),draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());save();raise
