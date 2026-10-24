"""Independent post-return source, residency and allocation audit."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(session,source,manifest):
    root=session.root;torch=session.torch
    versions={'geometry-v1':'300-e38-geometry-v1.json','geometry-v2':'301-e38-geometry-v2.json',
              'build-v1':'302-e38-build-v1.json','component-v1':'303-e38-component-v1.json',
              'audit-v1':'305-e38-audit-v1.json','memory-v1':'306-e38-memory-v1.json',
              'cleanup-v1':'307-e38-cleanup-v1.json','component-v2':'309-e38-component-v2.json'}
    if (root/'jobs/done/304-e38-measure-v1.json').exists():versions['measure-v1']='304-e38-measure-v1.json'
    records=[];last=None
    for version,job in versions.items():
        path=root/'jobs/done'/job;data=json.loads(path.read_text())
        folder=root.parent/('e38-prefill-20261024-'+version)
        seal=json.loads((folder/'manifest.json').read_text())
        for n,h in seal.items():assert digest(folder/n)==h,n
        wrapper=root/('e38_job_'+version.replace('-','_')+'.py')
        assert digest(wrapper)==seal['scripts/e38_run.py']
        if version=='geometry-v1':
            assert data['status']=='FAIL' and 'NoneType' in data['error']
        elif version=='audit-v1':
            assert data['status']=='FAIL' and '77896724480, 77762506752' in data['error']
        else:
            assert data['status']=='PASS',job
            result=data['result'];assert result['source_sha256']==seal
            assert result['model_reloaded'] is False
            assert result['target_object_id']==id(session.weights) and result['draft_object_id']==id(session.draft)
            assert result==json.loads((folder/'result.json').read_text())
            if 'baseline_bytes' in result:last=result
        records.append(dict(job=job,sha256=digest(path),status=data['status']))
    frozen=json.loads((root/'artifacts/e33-online-seal.json').read_text())
    for key in ('source_sha256','input_sha256'):
        for n,h in frozen[key].items():assert digest(root/n)==h,n
    for n,h in frozen['external_source_sha256'].items():
        assert digest(root.parent/'e2-load/reference/tensorfold'/n)==h,n
    from tensorfold.families.qwen4_exp.cuda import forward,nvfp4_moe
    for module,names in ((forward,('gdn_block','_mm','moe_block','_prefill_chain')),(nvfp4_moe,('moe',))):
        for name in names:
            fn=getattr(module,name);assert fn.__module__==module.__name__ and fn.__globals__ is module.__dict__
    assert not [n for n in sys.modules if n.startswith('_e38_')]
    gc.collect();torch.cuda.synchronize();allocated=torch.cuda.memory_allocated()
    assert last is not None and allocated<=last['baseline_bytes'],(allocated,last['baseline_bytes'])
    build=root.parent/'e38-prefill-20261024-build-v1'
    built=json.loads((build/'result.json').read_text());binary=Path(built['binary'])
    assert digest(binary)==built['binary_sha256']
    assert digest(build/'build/cuda/nvfp4/experts.cu')==built['generated_sha256']
    tool=shutil.which('cuobjdump')
    resource=None
    if tool:
        result=subprocess.run([tool,'--dump-resource-usage',str(binary)],capture_output=True,text=True,check=True)
        path=source/'candidate-resources.txt';path.write_text(result.stdout);resource=digest(path)
    extensions={n:digest(Path(sys.modules[n].__file__)) for n in ('spark_q_moe_linear','spark_q_moe_plan',
        'spark_q_moe_prompt','spark_q_block_gdn','spark_q_block_gdn_io','spark_q_block_gdn_shared')}
    result=dict(status='PASS_E38_AUDIT',source_sha256=manifest,jobs=records,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),model_reloaded=False,
        baseline_bytes=last['baseline_bytes'],allocated_bytes=allocated,original_forward_restored=True,
        frozen_sources_verified=True,native_group_released=not torch.distributed.is_initialized(),
        binary_sha256=built['binary_sha256'],resource_dump_sha256=resource,extensions_sha256=extensions)
    assert result['native_group_released']
    (source/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
