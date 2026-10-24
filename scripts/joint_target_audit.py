"""Validate every sealed stage and final resident resources without reloading weights."""
import gc
import hashlib
import json
from pathlib import Path
import sys


STAGES=('probe-v1','graph-v1','native-v1','trace-v1','lifecycle-v1')
JOBS=('210-joint-target-probe-v1.json','211-joint-target-graph-v1.json',
      '212-joint-native-v1.json','213-joint-trace-v1.json','214-joint-lifecycle-v1.json')


def run(session,arguments):
    root=session.root
    source=Path(arguments['source_root']);current=json.loads((source/'manifest.json').read_text())
    for n,h in current.items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h,n
    checked=[]
    for stage,name in zip(STAGES,JOBS):
        folder=root.parent/('joint-target-20261023-'+stage)
        manifest=json.loads((folder/'manifest.json').read_text())
        for n,h in manifest.items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==h,n
        job=json.loads((root/'jobs/done'/name).read_text())
        assert job['status']=='PASS' and job['load_count']==1
        assert job['result']['source_sha256']==manifest
        assert job['result']['target_object_id']==id(session.weights)
        assert job['result']['draft_object_id']==id(session.draft)
        checked.append(dict(stage=stage,job=name,status=job['result']['status'],manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()))
    frozen=json.loads((root/'artifacts/e33-online-seal.json').read_text())
    for key in ('source_sha256','input_sha256'):
        for n,h in frozen[key].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    for n,h in frozen['external_source_sha256'].items():
        assert hashlib.sha256((root.parent/'e2-load/reference/tensorfold'/n).read_bytes()).hexdigest()==h,n
    extensions={}
    for name in ('spark_q_moe_linear','spark_q_moe_plan','spark_q_moe_prompt',
                 'spark_q_block_gdn','spark_q_block_gdn_io','spark_q_block_gdn_shared'):
        extensions[name]=hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest()
    assert not session.torch.distributed.is_initialized(),'native process group leaked'
    gc.collect();session.torch.cuda.synchronize()
    allocated=session.torch.cuda.memory_allocated()
    assert allocated==77594734592,allocated
    return dict(status='PASS_JOINT_RESIDENT_AUDIT',checked=checked,source_sha256=current,
        frozen_external_sources_verified=True,extensions_sha256=extensions,native_group_released=True,
        allocated_bytes=allocated,target_object_id=id(session.weights),draft_object_id=id(session.draft),model_reloaded=False)
