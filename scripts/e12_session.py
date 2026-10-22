"""Load the pinned E8 target/drafter once for a resident research worker."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def load_session(root, checkpoint, progress):
    root, checkpoint = Path(root), Path(checkpoint)
    sys.path[:0] = [str(root/'followup-deps'), str(root/'python-deps'), str(root),
                   str(root/'build/varlen')]
    import torch
    from transformers import AutoTokenizer
    from e3_target import load_target
    from e12_routes import ObservedTarget
    from spark_infer_q.dflash2_reference import upstream_helpers
    from spark_infer_q.dflash2_block import BlockDraft
    from spark_infer_q.draft_head import HeadDraft, make_projection, Q4Backend

    weights, metadata = load_target(root, checkpoint, progress)
    target = ObservedTarget(weights, capacity=1024)
    sys.path.insert(0, str(root/'build/qmm'))
    import spark_q_e6_qmm
    from tensorfold.cuda.kernels import qmm
    from tensorfold.families.qwen4_exp.cuda.bf16 import quantize4
    assert hashlib.sha256(Path(spark_q_e6_qmm.__file__).read_bytes()).hexdigest() == '5e8088b6cbcb09d05bad9472f0616290457e662546dfdfe497dc5532879450dd'
    qmm._ext = lambda: spark_q_e6_qmm
    helpers = upstream_helpers(root/'reference/dflash.py', 'b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b')
    draft = BlockDraft(root/'checkpoint', weights.embed[0], weights.head.b.weight,
                       helpers, block_size=8, max_context=1024)
    projection = make_projection(weights.head.b.weight, precision='q4',
        backend=Q4Backend('tensorfold.group32.sm121',
            lambda w: w.is_cuda and w.dtype == torch.bfloat16 and torch.cuda.get_device_capability(w.device) == (12, 1),
            quantize4, qmm.matmul))
    draft = HeadDraft.attach(draft, projection)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                               trust_remote_code=False)
    eos_raw = json.loads((checkpoint/'generation_config.json').read_text())['eos_token_id']
    eos = set(eos_raw if isinstance(eos_raw, list) else [eos_raw])
    eos.add(tokenizer.eos_token_id)
    return SimpleNamespace(root=root, checkpoint=checkpoint, torch=torch, weights=weights,
        target=target, draft=draft, projection=projection, tokenizer=tokenizer, eos=eos,
        metadata=metadata, calibration=json.loads((root/'calibration.json').read_text()))
