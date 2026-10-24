"""Isolated forward snapshots and local projection/recurrence controls."""
import copy
import gc
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest


def tensor_hash(*tensors):
    h = hashlib.sha256()
    for x in tensors:
        y = x.detach().cpu().contiguous()
        h.update(str((str(y.dtype), tuple(y.shape))).encode())
        h.update(y.view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def metrics(a, b):
    import torch
    assert a.shape == b.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    af, bf = a.double(), b.double()
    delta = af-bf
    return dict(relative_l2=float(delta.norm()/bf.norm().clamp_min(1e-30)),
                max_abs=float(delta.abs().max()),
                top1_agreement=float((af.argmax(-1)==bf.argmax(-1)).double().mean()))


def module_at(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Capture:
    def __init__(self, session, forward, mode, ids):
        from tensorfold.families.qwen4_exp.cuda.state import Buffers, State
        torch, w = session.torch, session.weights
        self.state = State(w, 1024, 8, 'bf16', limit=1024)
        n, wide = len(ids), w.cfg.streams*w.cfg.hidden
        self.residual = [torch.empty((n,wide), dtype=torch.bfloat16, device=w.device) for _ in w.layers]
        self.picks = [torch.empty((n,w.cfg.top_k), dtype=torch.int32, device=w.device) for _ in w.layers]
        self.gdn_inputs = {i:torch.empty((n,w.cfg.hidden),dtype=torch.bfloat16,device=w.device) for i in (0,6)}
        self.logits = None
        rows = n if mode=='prefill' else (1 if mode=='serial' else 8)
        b = Buffers(w, rows, 1024, prefill=mode=='prefill')
        original = forward.gdn_block
        current_offset = 0
        def observed(layer, w, segs, buf, count, cuts=()):
            if layer.index in self.gdn_inputs:
                self.gdn_inputs[layer.index][current_offset:current_offset+count].copy_(buf.mixed[:count])
            return original(layer,w,segs,buf,count,cuts)
        forward.gdn_block = observed
        try:
            with torch.inference_mode():
                for at in range(0,n,rows):
                    current_offset = at; chunk = ids[at:at+rows]; count = len(chunk)
                    segs = forward.stage(w,b,[(self.state,chunk)])
                    forward._embed(w,b.ids[:count],w.cfg.streams,b.h[:count])
                    pending = None
                    for layer in w.layers:
                        pending = forward.layer_forward(layer,w,segs,b,count,pending)
                        shadow = b.h[:count].clone(); proxy = copy.copy(b)
                        proxy.pss = torch.empty_like(b.pss)
                        forward._writeback(shadow,proxy,count,w.cfg,pending)
                        self.residual[layer.index][at:at+count].copy_(shadow)
                        self.picks[layer.index][at:at+count].copy_(b.moe.pick[:count,:w.cfg.top_k])
                    self.logits = forward.finish(w,w.mixer,b,count,pending).clone()[-1:]
                    forward.commit(w,self.state,b,count,count)
                torch.cuda.synchronize()
        finally:
            forward.gdn_block = original


def same_trajectory(torch, a, b):
    return dict(residual_exact=all(torch.equal(x,y) for x,y in zip(a.residual,b.residual)),
                picks_exact=all(torch.equal(x,y) for x,y in zip(a.picks,b.picks)),
                last_logits_exact=torch.equal(a.logits,b.logits))


def projection_control(torch, layer, x):
    from tensorfold.cuda import prompt_precision
    assert not prompt_precision.fp8()
    q = layer.gdn.proj
    assert q.kernel=='mx8'
    # Exact stored e4m3 values, e8m0 powers of two, before any path-specific summation.
    dense = torch.ldexp(q.w8_rows().float(), (q.scale_rows().int()-127).repeat_interleave(32,dim=1))
    serial = torch.cat([q(x[i:i+1]).clone() for i in range(len(x))])
    prefill = q.prefill(x).clone()
    reference = x.float() @ dense.T
    key = tensor_hash(x)
    record = dict(component='mxfp8_gdn_projection',input_sha256=key,reference_input_sha256=key,
        reference_kind='fp32_same_stored_weights',tf32_disabled=True,
        compared_modes=dict(serial=metrics(serial,reference),prefill=metrics(prefill,reference)),
        reference_scope='local_operator',reference_boundary='bf16_input_fp32_compute',shared_reference_decoder=True)
    info = dict(control=record,serial_prefill=metrics(prefill,serial),
                serial_matches_rounded_reference=torch.equal(serial,reference.bfloat16()),
                prefill_matches_rounded_reference=torch.equal(prefill,reference.bfloat16()),
                stored_weight_sha256=tensor_hash(q.w8,q.bs),dense_weight_sha256=tensor_hash(dense))
    return info, serial


def recurrence_control(torch, layer, p, reference, eps):
    from tensorfold.families.qwen4_exp.cuda import gdn, gdn_io
    from tensorfold.cuda.kernels import gdn as shared
    n = len(p); g = layer.gdn; device = p.device
    conv = torch.zeros((3,10240),dtype=torch.bfloat16,device=device)
    initial = torch.zeros((48,128,128),dtype=torch.float32,device=device)
    initial_hash = tensor_hash(p,conv,initial)
    scratch = gdn.GDNScratch(n,device)
    final_decode = torch.empty_like(initial)
    decode_out = torch.empty((n,6144),dtype=torch.bfloat16,device=device)
    decode_xs = torch.empty((n,192),dtype=torch.float32,device=device)
    gdn.chain(p,conv,g.conv,initial,g.a_log,g.dt_bias,g.norm,eps,n,scratch,final_decode,decode_out,decode_xs)
    ptr = torch.tensor([conv.data_ptr()],dtype=torch.int64,device=device)
    sid = torch.zeros(n,dtype=torch.int32,device=device)
    windows = (torch.arange(n,dtype=torch.int32,device=device)[:,None]
               +torch.arange(4,dtype=torch.int32,device=device)[None,:]).contiguous()
    q,k,v,decay,beta = gdn_io.front(p,ptr,sid,windows,g.conv,g.a_log,g.dt_bias,16)
    assert all(torch.equal(a,b) for a,b in zip((k,v,decay,beta),(scratch.k,scratch.v,scratch.g,scratch.b))), 'front mismatch'
    final_prefill = torch.empty_like(initial)
    y = shared.chain(q,k,v,decay,beta,initial,final_prefill)
    prefill_out = torch.empty_like(decode_out); prefill_xs = torch.empty_like(decode_xs)
    gdn_io.back(y,p,g.norm,eps,prefill_out,prefill_xs)
    y32,state32 = reference.recurrence(q,k,v,decay,beta,initial)
    y64,state64 = reference.recurrence(q,k,v,decay,beta,initial,dtype=torch.float64)
    ref_out = torch.empty_like(decode_out); ref_xs = torch.empty_like(decode_xs)
    # back is shared; this comparison arbitrates recurrence, not front/back math.
    gdn_io.back(y32.bfloat16(),p,g.norm,eps,ref_out,ref_xs)
    front_hash=tensor_hash(q,k,v,decay,beta,initial)
    record = dict(component='gdn_recurrent_final_state', input_sha256=front_hash,
        reference_input_sha256=front_hash, reference_kind='fp32_same_front_inputs',tf32_disabled=True,
        compared_modes=dict(serial={k:v for k,v in metrics(final_decode,state32).items() if k!='top1_agreement'},
                            prefill={k:v for k,v in metrics(final_prefill,state32).items() if k!='top1_agreement'}),
        reference_scope='local_operator',reference_boundary='mixed_front_inputs_fp32_compute',shared_reference_decoder=True)
    assert tensor_hash(p,conv,initial)==initial_hash, 'local input was mutated'
    return dict(control=record, common_scratch_exact=True,checked_front_fields=['k','v','decay','beta'],
        q_alignment='source-equivalent front normalization; fused decode q is not exposed',
        front_sha256=front_hash,projection_conv_initial_sha256=initial_hash,
        gated_output_vs_fp32=dict(serial=metrics(decode_out,ref_out),prefill=metrics(prefill_out,ref_out)),
        prefill_raw_y_vs_fp32=metrics(y,y32),fp32_vs_fp64_state=metrics(state32,state64),
        state_vs_fp64=dict(serial=metrics(final_decode,state64),prefill=metrics(final_prefill,state64)),
        fp32_vs_fp64_y=metrics(y32,y64),decode_prefill_gated_exact=torch.equal(decode_out,prefill_out),
        initial_state='zero',raw_decode_y_exposed=False,shared_front_and_back=True,
        prefill_y_dtype=str(y.dtype),executed_eps=eps,
        gated_reference_boundary='FP32 recurrence -> BF16 readout -> existing back')


def diagnose(session, source, cases):
    torch = session.torch
    from tensorfold.families.qwen4_exp.cuda import forward as upstream
    from e3_wide_target import WideTarget
    before = torch.cuda.memory_allocated()
    tag = '_e36_forward_'+hashlib.sha256(str(source).encode()).hexdigest()[:10]
    forward = ModuleType(tag); forward.__package__ = upstream.__package__
    forward.__file__ = upstream.__file__; sys.modules[tag] = forward
    exec(compile(Path(upstream.__file__).read_bytes(),upstream.__file__,'exec'),forward.__dict__)
    reference = module_at(source/'scripts/e36_gdn_reference.py',tag+'_reference')
    gate = module_at(source/'spark_infer_q/prefill_evidence.py',tag+'_gate')
    report = dict(status='RUNNING_E36',cases=[],local_controls=[],default_promoted=False,
                  full_model_fp32_reference=False,task_quality_evaluated=False,performance_evaluated=False)
    path = source/'diagnostic-result.json'
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    try:
        # Small CPU equation tests run inside the same installed Torch runtime.
        previous_scripts=sys.modules.get('scripts')
        previous_ref=sys.modules.get('scripts.e36_gdn_reference')
        package=ModuleType('scripts');package.__path__=[str(source/'scripts')]
        sys.modules['scripts']=package;sys.modules['scripts.e36_gdn_reference']=reference
        try:
            tests=module_at(source/'tests/test_e36_gdn_reference.py',tag+'_tests')
            output=io.StringIO()
            checked=unittest.TextTestRunner(stream=output).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
            assert checked.wasSuccessful() and checked.testsRun==4 and not checked.skipped,output.getvalue()
            report['reference_cpu_tests']=dict(run=checked.testsRun,skipped=len(checked.skipped),passed=True)
        finally:
            sys.modules.pop(tag+'_tests',None)
            for name,previous in (('scripts',previous_scripts),('scripts.e36_gdn_reference',previous_ref)):
                if previous is None:sys.modules.pop(name,None)
                else:sys.modules[name]=previous
        with torch.inference_mode():
            for c in cases:
                report['active']=c['name']+':rows8';save()
                chunked = Capture(session,forward,'rows8',c['ids'])
                serial = None
                if c['length']<=64:
                    report['active']=c['name']+':serial';save()
                    serial = Capture(session,forward,'serial',c['ids'])
                    exact = same_trajectory(torch,chunked,serial)
                    difference = WideTarget.state_difference(type('Owner',(),dict(torch=torch))(),chunked.state,serial.state)
                    assert all(exact.values()) and not difference, (c['name'],exact,difference)
                    del serial
                else:exact=None
                report['active']=c['name']+':prefill';save()
                native = Capture(session,forward,'prefill',c['ids'])
                layers=[]
                for i,(a,b) in enumerate(zip(chunked.residual,native.residual)):
                    pick_a,pick_b=chunked.picks[i].sort(-1).values,native.picks[i].sort(-1).values
                    layers.append(dict(layer=i,residual=metrics(b,a),residual_exact=torch.equal(a,b),
                        expert_set_agreement=float((pick_a==pick_b).all(-1).float().mean())))
                report['cases'].append(dict(name=c['name'],length=c['length'],token_sha256=c['token_sha256'],
                    serial_rows8=exact,layers=layers,last_logits=metrics(native.logits,chunked.logits),
                    last_argmax=dict(rows8=int(chunked.logits.argmax()),prefill=int(native.logits.argmax())),
                    first_different_layer=next((i for i,l in enumerate(layers) if not l['residual_exact']),None),
                    first_route_difference=next((i for i,l in enumerate(layers) if l['expert_set_agreement']<1),None)))
                del native
                for index in (0,6):
                    report['active']=c['name']+':local:'+str(index);save()
                    projection,p=projection_control(torch,session.weights.layers[index],chunked.gdn_inputs[index])
                    recurrence=recurrence_control(torch,session.weights.layers[index],p,reference,session.weights.cfg.eps)
                    gate.validate_local_control(projection['control']);gate.validate_local_control(recurrence['control'])
                    report['local_controls'].append(dict(name=c['name'],layer=index,projection=projection,recurrence=recurrence))
                    del p
                del chunked
                gc.collect();torch.cuda.synchronize();save()
                print(json.dumps(dict(case=c['name'],status='PASS_E36_CASE')),flush=True)
        report.update(status='PASS_E36_LOCAL_DIAGNOSIS',allocated_after=torch.cuda.memory_allocated(),allocated_before=before)
        report.pop('active',None);save()
        return report
    finally:
        forward.gdn_block=upstream.gdn_block
        for name in (tag,tag+'_reference',tag+'_gate'):sys.modules.pop(name,None)
