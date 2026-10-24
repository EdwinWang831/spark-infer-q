"""Independent source, object, resource and post-return allocation audit."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(session,source,manifest):
    root=session.root;records=[];results={}
    versions={'build-v1':'320-e39-build-v1.json','component-v1':'321-e39-component-v1.json'}
    if (root/'jobs/done/322-e39-measure-v1.json').exists():versions['measure-v1']='322-e39-measure-v1.json'
    for version,name in versions.items():
        folder=root.parent/('e39-prefill-20261024-'+version);seal=json.loads((folder/'manifest.json').read_text())
        for n,h in seal.items():assert digest(folder/n)==h,n
        wrapper=root/('e39_job_'+version.replace('-','_')+'.py');assert digest(wrapper)==seal['scripts/e39_run.py']
        path=root/'jobs/done'/name;data=json.loads(path.read_text());assert data['status']=='PASS',data
        result=data['result'];assert result==json.loads((folder/'result.json').read_text())
        assert result['source_sha256']==seal and result['model_reloaded'] is False
        assert result['target_object_id']==id(session.weights) and result['draft_object_id']==id(session.draft)
        records.append(dict(job=name,sha256=digest(path),status='PASS'));results[version]=result
    frozen=json.loads((root/'artifacts/e33-online-seal.json').read_text())
    for key in ('source_sha256','input_sha256'):
        for n,h in frozen[key].items():assert digest(root/n)==h,n
    for n,h in frozen['external_source_sha256'].items():assert digest(root.parent/'e2-load/reference/tensorfold'/n)==h,n
    from tensorfold.families.qwen4_exp.cuda import forward,nvfp4_moe
    for mod,names in ((forward,('gdn_block','_mm','moe_block','_prefill_chain')),(nvfp4_moe,('moe',))):
        for name in names:
            fn=getattr(mod,name);assert fn.__module__==mod.__name__ and fn.__globals__ is mod.__dict__
    assert not [n for n in sys.modules if n.startswith('_e39_')]
    gc.collect();session.torch.cuda.synchronize();allocated=session.torch.cuda.memory_allocated()
    last=results.get('measure-v1',results['component-v1'])
    assert allocated<=last['baseline_bytes'],(allocated,last['baseline_bytes'])
    build=root.parent/'e39-prefill-20261024-build-v1';built=results['build-v1'];binary=Path(built['binary'])
    assert digest(binary)==built['binary_sha256'] and digest(build/'build/cuda/nvfp4/experts.cu')==built['generated_sha256']
    tool=shutil.which('cuobjdump');assert tool
    p=source/'candidate-resources.txt';p.write_text(subprocess.run([tool,'--dump-resource-usage',str(binary)],capture_output=True,text=True,check=True).stdout)
    extensions={n:digest(Path(sys.modules[n].__file__)) for n in ('spark_q_moe_linear','spark_q_moe_plan',
        'spark_q_moe_prompt','spark_q_block_gdn','spark_q_block_gdn_io','spark_q_block_gdn_shared')}
    report=dict(status='PASS_E39_AUDIT',source_sha256=manifest,jobs=records,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),model_reloaded=False,
        baseline_bytes=last['baseline_bytes'],allocated_bytes=allocated,original_forward_restored=True,
        frozen_sources_verified=True,native_group_released=not session.torch.distributed.is_initialized(),
        binary_sha256=built['binary_sha256'],resource_dump_sha256=digest(p),extensions_sha256=extensions)
    assert report['native_group_released']
    (source/'result.json').write_text(json.dumps(report,indent=2)+'\n');return report
