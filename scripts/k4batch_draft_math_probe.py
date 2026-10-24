"""Read-only causal probe of BF16 matmul row-count arithmetic on real features."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_batch_draft_math';pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    torch=session.torch;flag=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    rows=[]
    try:
        opt=importlib.import_module(tag+'.kaifra_optimized');prompt=importlib.import_module(tag+'.prompt')
        from e3_wide_target import WideTarget
        provider,_=opt.optimized_provider(session)
        ids=session.tokenizer.apply_chat_template([dict(role='user',content='Write a Python function to remove duplicates while preserving order. Return only code.')],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        state,features,logits=prompt.prompt_state(WideTarget(session.weights,capacity=2048),ids,chunk_size=8)
        weight=provider.worker.weights['fc.weight'];linear=torch.nn.functional.linear
        reference=linear(features,weight)
        for reduced in (True,False):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduced
            for count in (1,2,4):
                x=features.repeat((count,1));actual=linear(x,weight)[:len(features)]
                rows.append(dict(reduced=reduced,requests=count,exact=torch.equal(actual,reference),
                    max_abs=float((actual.float()-reference.float()).abs().max()),
                    different=int((actual!=reference).sum()),
                    relative_l2=float((actual.float()-reference.float()).norm()/reference.float().norm())))
        return dict(status='PASS_BATCH_DRAFT_MATH_PROBE',source_sha256=manifest,original_flag=flag,
            rows=rows,model_reloaded=False)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flag
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.'):sys.modules.pop(n,None)
