"""Export the negative single-tile instruction-pipeline screen."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prefill_kernel_evidence import bundle,clean,require,sha
from scripts.e43_accounting import summarize


def export(raw,build_version,build_job,component_version,component_job,audit_version,audit_job):
    value=bundle(raw,'e43',build_version,build_job,component_version,component_job,audit_version,audit_job,allow_unrecovered=True)
    summary=summarize(value['reports']['component']['cases']);require(summary==value['reports']['component']['summary'],'cost summary differs')
    require(summary['cost_gate_passed'] is False,'E43 archived outcome differs')
    value.update(summary=summary,full_candidate_prompt_validated=False,full_candidate_requests_validated=False,
        exporter_sha256=sha(__file__),evidence_helper_sha256=sha(Path(__file__).with_name('prefill_kernel_evidence.py')))
    if not value['residency']['memory_gate_passed']:value['status']='E43_REJECTED_WITH_UNRESOLVED_MEMORY_AUDIT'
    value['failed_audits']=[]
    for name in ('434-e43-audit-v1.json','440-e43-audit-v2.json'):
        p=Path(raw)/'jobs/done'/name
        if p.exists():
            j=json.loads(p.read_text());require(j['status']=='FAIL','failed audit history differs')
            value['failed_audits'].append(dict(job=name,sha256=sha(p),error=j['error']))
    return clean(value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    for k in ('build-version','build-job','component-version','component-job','audit-version','audit-job'):p.add_argument('--'+k,required=True)
    a=p.parse_args();v=export(a.raw,a.build_version,a.build_job,a.component_version,a.component_job,a.audit_version,a.audit_job)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    print(json.dumps(v['summary'],indent=2))
