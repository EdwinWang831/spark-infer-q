"""Compare against the unchanged full z-lab DFlash PyTorch model, externally supplied."""
import hashlib
import importlib.util
import json
from pathlib import Path

OFFICIAL_COMMIT='07ebd93db9f472af339b644bb70221ad8428328a'
OFFICIAL_SHA='f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e'


def official_module(path):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=OFFICIAL_SHA:
        raise ValueError('official DFlash source identity mismatch')
    spec=importlib.util.spec_from_file_location('spark_q_external_dflash',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare(draft, cases, source):
    """FP32 is a mathematical control; BF16 diagnostics retain backend rounding."""
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from transformers import Qwen3Config
    module=official_module(source)
    cfg=Qwen3Config(**draft.config)
    cfg._attn_implementation='sdpa'
    model=module.DFlash2DraftModel(cfg)
    state={k+'.weight' if k in ('candidate_selector.predecessor_codebook','candidate_selector.successor_codebook') else k:v
           for k,v in draft.weights.items()}
    result=model.load_state_dict(state,strict=True,assign=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.requires_grad_(False).eval()
    model.rotary_emb.to('cuda')
    rows=[]
    dtype=next(iter(draft.weights.values())).dtype
    for features,anchor,label in cases:
        features=features.to(dtype)
        trace=[]
        hooks=[model.hidden_norm.register_forward_hook(lambda m,a,o:trace.append(o[0].detach()))]
        for layer in model.layers:
            hooks.append(layer.register_forward_hook(lambda m,a,o:trace.append(o[0].detach())))
        ids=torch.tensor([[anchor]+[248077]*3],device='cuda')
        noise=torch.nn.functional.embedding(ids,draft.embed)
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            expected=model(position_ids=torch.arange(len(features)+4,device='cuda')[None],
                           noise_embedding=noise,target_hidden=features[None],use_cache=False)
            actual,observed=draft.forward_hidden(features,anchor,trace=True)
            expected_logits=torch.nn.functional.linear(expected[:,1:],draft.head)
            actual_logits=torch.nn.functional.linear(actual[None,1:],draft.head)
            tokens,candidates,_=model.propose(expected[:,1:],ids[:,0],lambda h:torch.nn.functional.linear(h,draft.head),0.)
            proposed,details=draft.propose(features,anchor)
        for hook in hooks: hook.remove()
        def difference(a,b):
            a,b=a.float(),b.float()
            return float((a-b).norm()/b.norm().clamp_min(1e-12))
        hidden_l2=difference(actual,expected[0])
        logits_l2=difference(actual_logits,expected_logits)
        overlap=sum(len(set(a)&set(b)) for a,b in zip(details['candidate_ids'],candidates[0].tolist()))/48
        entry={'label':label,'dtype':str(dtype),'prefix_rows':len(features),
               'hidden_relative_l2':hidden_l2,'logits_relative_l2':logits_l2,
               'context_and_completed_layer_relative_l2':[difference(a,b) for a,b in zip(observed,trace)],
               'proposal_ids':proposed,'official_proposal_ids':tokens[0].tolist(),
               'proposal_ids_equal':proposed==tokens[0].tolist(),'candidate_set_overlap':overlap}
        assert len(observed)==len(trace)==6
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        if dtype==torch.float32:
            # Set before execution: tolerate FP32 accumulation/SDPA ordering only.
            assert hidden_l2<1e-3 and logits_l2<1e-3,(label,entry)
            assert overlap>=.95,(label,entry)
        rows.append(entry)
    return {'official_commit':OFFICIAL_COMMIT,'official_source_sha256':OFFICIAL_SHA,
            'backend':'unchanged official PyTorch model, transformers 5.12.1, SDPA MATH',
            'scope':'full draft forward and selector; not complete SGLang runtime',
            'fp32_thresholds':{'hidden_relative_l2_max':1e-3,'logits_relative_l2_max':1e-3,'candidate_overlap_min':.95},
            'bf16_scope':'numerical diagnostic; fused SGLang-style boundaries can differ from HF',
            'cases':rows}
