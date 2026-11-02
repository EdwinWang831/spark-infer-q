"""Final seal, model identity, and allocation check on the same resident session."""
import gc
import json
from e28_probe import sha


def run(session, arguments):
    root=session.root
    reports=[]
    for name in ('e34','e34-v2'):
        seal=json.loads((root/'artifacts'/(name+'-seal.json')).read_text())
        for key in ('source_sha256','input_sha256'):
            for n,h in seal[key].items():assert sha(root/n)==h,n
        for n,h in seal['external_source_sha256'].items():
            assert sha(root.parent/'e2-load/reference/tensorfold'/n)==h,n
        report=json.loads((root/'artifacts'/(name+'-probe.json')).read_text())
        assert report['target_object_id']==id(session.weights)
        assert report['draft_object_id']==id(session.draft)
        reports.append(report)
    assert reports[1]['status']=='REJECT_E34_HEAD_COST'
    gc.collect()
    session.torch.cuda.synchronize()
    allocated=session.torch.cuda.memory_allocated()
    assert allocated==77594734592,allocated
    return dict(status='PASS_E34_RESIDENT_AUDIT',target_object_id=id(session.weights),
        draft_object_id=id(session.draft),allocated_bytes=allocated,model_reloaded=False,
        initial_failed_job_preserved=True,immutable_sources_verified=True)
