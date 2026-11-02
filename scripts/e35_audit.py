"""Audit E35 sealed sources and release of temporary graphs/streams."""
import gc
import json
from e28_probe import sha


def run(session,arguments):
    root=session.root
    seal=json.loads((root/'artifacts/e35-seal.json').read_text())
    for key in ('source_sha256','input_sha256'):
        for n,h in seal[key].items():assert sha(root/n)==h,n
    for n,h in seal['external_source_sha256'].items():
        assert sha(root.parent/'e2-load/reference/tensorfold'/n)==h,n
    report=json.loads((root/'artifacts/e35-probe.json').read_text())
    assert report['status']=='REJECT_E35_MOE_COST'
    assert report['target_object_id']==id(session.weights) and report['draft_object_id']==id(session.draft)
    assert len(report['preflight'])==96 and len(report['cases'])==24 and len(report['trajectory'])==8
    for t in report['traces']:assert sha(root/'artifacts'/t['file'])==t['sha256']
    gc.collect();session.torch.cuda.synchronize()
    allocated=session.torch.cuda.memory_allocated()
    assert allocated==77594734592,allocated
    return dict(status='PASS_E35_RESIDENT_AUDIT',target_object_id=id(session.weights),draft_object_id=id(session.draft),
        allocated_bytes=allocated,model_reloaded=False,immutable_sources_verified=True)
