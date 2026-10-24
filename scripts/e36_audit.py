"""Post-return prefill diagnostic audit: source, model identity and live resources."""
import gc
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(session, source):
    current=json.loads((source/'manifest.json').read_text())
    for name,h in current.items():assert digest(source/name)==h,name
    root=session.root
    reports=[]
    for version,job in (('preflight-v1','240-e36-preflight-v1.json'),('diagnose-v2','242-e36-diagnose-v2.json'),
                        ('localize-v1','244-e36-localize-v1.json'),('candidate-v1','246-e36-candidate-v1.json')):
        folder=root.parent/('e36-prefill-20261023-'+version)
        manifest=json.loads((folder/'manifest.json').read_text())
        for name,h in manifest.items():assert digest(folder/name)==h,name
        result=json.loads((root/'jobs/done'/job).read_text())
        assert result['status']=='PASS' and result['load_count']==1
        report=result['result'];assert report['source_sha256']==manifest
        assert report['target_object_id']==id(session.weights)
        assert report['draft_object_id']==id(session.draft)
        reports.append(report)
    assert reports[0]['inputs']==reports[1]['inputs']
    assert reports[0]['inputs']==reports[2]['inputs']
    assert reports[0]['inputs']==reports[3]['inputs']
    assert reports[3]['status'] in ('REJECT_E36_CANDIDATE_NUMERICS','PASS_E36_CANDIDATE_PROMPT')
    if reports[3]['status']=='REJECT_E36_CANDIDATE_NUMERICS':
        assert not reports[3]['timings'] and not reports[3]['performance_evaluated']
    assert reports[1]['status']=='PASS_E36_LOCAL_DIAGNOSIS'
    assert reports[1]['reference_cpu_tests']==dict(run=4,skipped=0,passed=True)
    frozen=json.loads((root/'artifacts/e33-online-seal.json').read_text())
    for key in ('source_sha256','input_sha256'):
        for name,h in frozen[key].items():assert digest(root/name)==h,name
    for name,h in frozen['external_source_sha256'].items():
        assert digest(root.parent/'e2-load/reference/tensorfold'/name)==h,name
    from tensorfold.families.qwen4_exp.cuda import forward
    assert forward.gdn_block.__module__==forward.__name__
    assert forward.gdn_block.__globals__ is forward.__dict__
    extensions={name:digest(Path(sys.modules[name].__file__)) for name in
        ('spark_q_moe_linear','spark_q_moe_plan','spark_q_moe_prompt',
         'spark_q_block_gdn','spark_q_block_gdn_io','spark_q_block_gdn_shared')}
    gc.collect();session.torch.cuda.synchronize()
    allocated=session.torch.cuda.memory_allocated()
    baseline=reports[1]['allocated_before']
    assert allocated<=baseline,(allocated,baseline)
    return dict(status='PASS_E36_AUDIT',source_sha256=current,allocated_bytes=allocated,
        baseline_bytes=baseline,extensions_sha256=extensions,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),
        frozen_sources_verified=True,diagnose_job='242-e36-diagnose-v2.json',
        localization_job='244-e36-localize-v1.json',
        candidate_job='246-e36-candidate-v1.json',
        original_forward_restored=True,configured_eps=session.weights.cfg.eps,
        native_group_released=not session.torch.distributed.is_initialized(),model_reloaded=False)
