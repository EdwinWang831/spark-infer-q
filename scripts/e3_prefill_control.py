"""Isolate the observed prefill MoE precision policy; no upstream source edits."""


def probe(target):
    import torch
    from tensorfold.families.qwen4_exp.cuda.state import Buffers
    w,f=target.w,target.forward
    tokens=[100,101,102,103]
    decode,_=target.step(target.state(),tokens,capture=False)
    results={}
    for name,force_float in [('native_prefill',False),('prefill_decode_style_moe',True)]:
        state=target.state()
        b=Buffers(w,4,target.capacity,prefill=True)
        original=str(b.moe.y.dtype)
        if force_float:
            b.moe.y=torch.empty_like(b.moe.y,dtype=torch.float32)
        lg=f.forward(w,state,b,tokens).clone()
        f.commit(w,state,b,4,4);torch.cuda.synchronize()
        a,reference=lg[-1].float(),decode[-1].float()
        assert torch.isfinite(lg).all()
        results[name]={'moe_slots_original_dtype':original,'moe_slots_executed_dtype':str(b.moe.y.dtype),
            'last_logits_relative_l2_vs_decode':float((a-reference).norm()/reference.norm().clamp_min(1e-12)),
            'last_logits_bit_exact_vs_decode':torch.equal(lg[-1],decode[-1]),
            'last_argmax_id':int(a.argmax()),'decode_last_argmax_id':int(reference.argmax())}
    return {'tokens':tokens,'cases':results,
            'control':'F32 MoE slots also select decode shared-expert path; two linked policies, not a single-rounding isolation',
            'scope':'four synthetic token IDs; numerical localization only; no quality or performance claim'}
