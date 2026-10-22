"""E11 controlled MXFP8 reduction A/B, forked from the frozen E8 executor."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--session',type=int,default=0);p.add_argument('--prefill-rows',type=int,choices=(1,8),default=8);p.add_argument('--port',type=int,default=18081);a=p.parse_args()
    root=a.root
    sys.path[:0]=[str(root/'followup-deps'),str(root/'python-deps'),str(root),str(root/'build/varlen')]
    import torch
    from e11_projection_route import ProjectionRoute
    from transformers import AutoTokenizer
    from spark_infer_q.prompt import prompt_state
    from e3_target import load_target
    from e3_wide_target import WideTarget
    from e3_mechanism_checks import entry,params
    from spark_infer_q.dflash2_reference import upstream_helpers
    from spark_infer_q.dflash2_block import BlockDraft
    from spark_infer_q.draft_head import HeadDraft,make_projection,Q4Backend
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.verification_budget import LaggedAcceptance
    import spark_q_e3_varlen
    assert hashlib.sha256(Path(spark_q_e3_varlen.__file__).read_bytes()).hexdigest()=='722b6ea835fa3e7e349af7f22ac727045a65af487a35f6709f3838e066bd678e'
    manifest=json.loads((root/'ready.json').read_text())
    for name,sha in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    meta={'status':'LOADING','session':a.session,'harness_sha256':manifest}
    metapath=root/'artifacts'/f'ours-server-{a.session}.json'
    def save():metapath.write_text(json.dumps(meta,indent=2)+'\n')
    save()
    weights,loaded=load_target(root,a.checkpoint,lambda e:print(json.dumps(e),flush=True))
    meta['target']=loaded;target=WideTarget(weights,capacity=1024)
    projections=ProjectionRoute.install()
    meta['projection_scratch_bytes']=projections.payload_bytes
    sys.path.insert(0,str(root/'build/qmm'))
    import spark_q_e6_qmm
    from tensorfold.cuda.kernels import qmm
    from tensorfold.families.qwen4_exp.cuda.bf16 import quantize4
    qmm._ext=lambda:spark_q_e6_qmm
    assert hashlib.sha256(Path(spark_q_e6_qmm.__file__).read_bytes()).hexdigest()=='5e8088b6cbcb09d05bad9472f0616290457e662546dfdfe497dc5532879450dd'
    tokenizer=AutoTokenizer.from_pretrained(str(a.checkpoint),local_files_only=True,trust_remote_code=False)
    helpers=upstream_helpers(root/'reference/dflash.py','b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b')
    draft=BlockDraft(root/'checkpoint',weights.embed[0],weights.head.b.weight,helpers,block_size=8,max_context=1024)
    projection=make_projection(weights.head.b.weight,precision='q4',backend=Q4Backend('tensorfold.group32.sm121',
        lambda weight:weight.is_cuda and weight.dtype==torch.bfloat16 and torch.cuda.get_device_capability(weight.device)==(12,1),quantize4,qmm.matmul))
    draft=HeadDraft.attach(draft,projection)
    meta['head']=dict(precision=projection.effective_precision,backend=projection.backend,incremental_payload_bytes=projection.payload_bytes,rows=len(weights.head.b.weight))
    meta['e6_source_sha256']=hashlib.sha256((root/'e6-frozen.json').read_bytes()).hexdigest()
    old=json.loads((root/'calibration.json').read_text())
    curve=old['verify_commit_curve_ms'];prior=old['conditional_prior'];draft_cost=old['draft_median_ms']['8']
    eos_raw=json.loads((a.checkpoint/'generation_config.json').read_text())['eos_token_id']
    eos=set(eos_raw if isinstance(eos_raw,list) else [eos_raw]);eos.add(tokenizer.eos_token_id)
    golden=None;last_result=None;meta['calibration']=old;meta['default_prefill_rows']=a.prefill_rows

    def generate(task,mode,emit=None,capture=False,oracle=None,prefill_rows=None):
        mode,sep,kernel=mode.partition('_kernel_')
        kernel=kernel if sep else 'original'
        projections.begin(kernel)
        torch.cuda.synchronize();request_start=time.perf_counter()
        ids=tokenizer.apply_chat_template([{'role':'user','content':task['prompt']}],tokenize=True,add_generation_prompt=True,
            enable_thinking=task['thinking'],return_dict=False)
        assert len(ids)+task['limit']<=1017
        rows=a.prefill_rows if prefill_rows is None else prefill_rows
        assert rows in (1,8)
        st,context,lg=prompt_state(target,ids,chunk_size=rows,capture=mode!='serial')
        output=[int(lg[-1].argmax())];stop=output[-1] in eos
        prefill_ms=(time.perf_counter()-request_start)*1000
        if emit:emit(output)
        policy=LaggedAcceptance(prior);budget=spark_q_e3_varlen.BudgetProbe()
        budget.curve([(rows,ms+(draft_cost if rows>1 else 0.)) for rows,ms in curve])
        for _ in range(5):budget.runtime(1,1,curve[0][1],0.)
        steps=[];logits=[]
        torch.cuda.synchronize();t0=time.perf_counter()
        while len(output)<task['limit'] and not stop:
            before=len(output)
            if mode=='serial':
                lg,_=target.step(st,[output[-1]],capture=False)
                output.append(int(lg[-1].argmax()));stop=output[-1] in eos
                if capture:logits.append(lg.cpu())
                if oracle:
                    assert output[-1]==oracle['tokens'][before]
                    assert torch.equal(lg.cpu(),oracle['logits'][before-1])
            else:
                width={'fixed7':8}.get(mode)
                if width is None:width=budget.schedule(params(),[entry(policy.current,policy.previous,process=len(steps)+2,schedule=len(steps)+2)])[0]
                k=min(width-1,task['limit']-len(output)-1)
                t=time.perf_counter();proposals=draft.propose(context,output[-1])[0][:k] if k else []
                torch.cuda.synchronize();dm=(time.perf_counter()-t)*1000
                t=time.perf_counter();lg,features,b=target.run(st,[output[-1]]+proposals);vm=(time.perf_counter()-t)*1000
                pred=lg.argmax(-1).tolist()
                out,keep,_,stop=greedy_result(output[-1],proposals,pred,eos_ids=eos,max_new=task['limit']-len(output))
                raw_accept=0
                for proposed,nxt in zip(proposals,pred):
                    if proposed!=nxt:break
                    raw_accept+=1
                observed=min(k,keep) if stop else k;accepted=min(raw_accept,observed)
                t=time.perf_counter();target.commit(st,b,k+1,keep);cm=(time.perf_counter()-t)*1000
                context=torch.cat([context,features[:keep]]);output.extend(out)
                assert len(context)==st.pos==len(ids)+len(output)-1
                if oracle:
                    assert out==oracle['tokens'][before:before+len(out)]
                    for j in range(keep):assert torch.equal(lg[j:j+1].cpu(),oracle['logits'][before-1+j])
                policy.observe(observed,accepted);budget.runtime(1,k+1,dm+vm+cm,0.)
                steps.append(dict(width=width,verified_rows=k+1,accepted=accepted,observed=observed,
                    emitted=len(out),draft_ms=dm,verify_ms=vm,commit_ms=cm))
            if emit:emit(output)
        torch.cuda.synchronize();decode_ms=(time.perf_counter()-t0)*1000
        return dict(tokens=output,state=st,logits=logits,steps=steps,prompt_ids=ids,
            projection_kernel=kernel,projection_counts=dict(projections.counts),
            prefill_rows=rows,prefill_ms=prefill_ms,decode_ms=decode_ms,core_wall_ms=(time.perf_counter()-request_start)*1000,stopped_by_eos=stop)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path=='/e4/resource':
                meta['resource']={'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                for name in ('memory.current','memory.peak','memory.events'):
                    f=Path('/sys/fs/cgroup')/name
                    if f.exists():meta['resource'][name]=f.read_text().strip()
                save()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(b'{"status":"ready"}')
        def do_POST(self):
            nonlocal golden,last_result
            try:
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path=='/e8/last-check':
                    assert golden is not None and last_result is not None
                    assert last_result['tokens']==golden['tokens']
                    assert last_result['prompt_ids']==golden['prompt_ids']
                    assert not target.state_difference(last_result['state'],golden['state'])
                    self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
                    self.wfile.write(json.dumps({'tokens_exact':True,'live_state_exact':True}).encode())
                    last_result=None;return
                if self.path=='/e8/check':
                    golden=None;last_result=None
                    checks=[]
                    with torch.inference_mode():
                        ref=generate(body,'serial',capture=True,prefill_rows=1)
                        for mode,rows in (('serial',8),('fixed7',8),('dynamic7',1),('dynamic7',8),
                                          ('fixed7_kernel_partials',8),('dynamic7_kernel_partials',8)):
                            got=generate(body,mode,oracle=ref,prefill_rows=rows)
                            assert got['tokens']==ref['tokens'] and not target.state_difference(got['state'],ref['state'])
                            checks.append({'mode':mode,'prefill_rows':rows,'kept_logits_exact':True,'live_state_exact':True})
                            del got
                    result={'token_ids':ref['tokens'],'prompt_ids':ref['prompt_ids'],'checks':checks,
                        'stopped_by_eos':ref['stopped_by_eos'],'serial_logit_rows':len(ref['logits'])}
                    golden=ref
                    self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
                    self.wfile.write(json.dumps(result).encode());return
                task=dict(prompt=body['messages'][0]['content'],thinking=body['chat_template_kwargs']['enable_thinking'],limit=body['max_tokens'])
                requested=body.get('e4_mode','dynamic7')
                mode,sep,chunk=requested.partition('_rows')
                rows=int(chunk) if sep else a.prefill_rows
                assert mode in ('serial','fixed7','dynamic7','dynamic7_kernel_original','dynamic7_kernel_partials') and rows in (1,8)
                last_result=None
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                sent=''
                def send(obj):self.wfile.write(('data: '+json.dumps(obj)+'\n\n').encode());self.wfile.flush()
                def emit(tokens):
                    nonlocal sent
                    text=tokenizer.decode(tokens,skip_special_tokens=True)
                    # Defer an incomplete UTF-8 suffix until a subsequent token completes it.
                    text=text.rstrip('\ufffd')
                    if text.startswith(sent) and len(text)>len(sent):
                        send({'choices':[{'delta':{'content':text[len(sent):]}}]});sent=text
                with torch.inference_mode():r=generate(task,mode,emit,prefill_rows=rows)
                raw=tokenizer.decode(r['tokens'],skip_special_tokens=False)
                content=raw.rsplit('</think>',1)[-1] if task['thinking'] else raw
                reasoning=raw.split('</think>',1)[0] if task['thinking'] and '</think>' in raw else raw if task['thinking'] else ''
                if task['thinking'] and '</think>' not in raw:content=''
                import re
                content=re.sub(r'<\|[^>]+\|>','',content).strip()
                runtime=dict(engine='spark-infer-q-e11-q4',projection_kernel=r['projection_kernel'],projection_counts=r['projection_counts'],prefill_rows=r['prefill_rows'],enable_thinking=task['thinking'],prefill_seconds=r['prefill_ms']/1000,
                    decode_seconds=r['decode_ms']/1000,tokens_per_second=(len(r['tokens'])-1)*1000/r['decode_ms'] if r['decode_ms'] else 0,
                    token_sha=hashlib.sha256(','.join(map(str,r['tokens'])).encode()).hexdigest()[:12])
                send({'choices':[{'delta':{},'finish_reason':'stop' if r['stopped_by_eos'] else 'length'}],
                    'usage':{'prompt_tokens':len(r['prompt_ids']),'completion_tokens':len(r['tokens']),'prompt_tokens_details':{'cached_tokens':0}},
                    'tensorfold':runtime,'e4':{'token_ids':r['tokens'],'prompt_ids':r['prompt_ids'],'steps':r['steps'],
                        'core_wall_ms':r['core_wall_ms'],'content':content,'reasoning':reasoning}})
                self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
                last_result=r
            except Exception:
                traceback.print_exc();raise

    meta['status']='READY';save();print(json.dumps({'status':'READY','session':a.session}),flush=True)
    try:HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
    finally:
        meta['resource']={'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        save()


if __name__=='__main__':main()
