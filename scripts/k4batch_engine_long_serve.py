"""Start the unified service with already loaded weights and a sealed source."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType


def run(session, arguments):
    source=Path(arguments['source_root'])
    manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    tag='_engine_live_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    package=ModuleType(tag);package.__path__=[str(source/'spark_infer_q')];sys.modules[tag]=package
    server=importlib.import_module(tag+'.engine_server')
    config=server.ServiceConfig(session.root,session.checkpoint,port=18085)
    spec=importlib.util.spec_from_file_location(tag+'.validation',source/'scripts/k4batch_engine_long.py')
    validation=importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)
    inputs=dict(long=[validation.make_messages(session,index) for index in range(4)],
                short=[validation.make_messages(session,index,False) for index in range(4)])
    (source/'live-inputs.json').write_text(json.dumps(inputs,ensure_ascii=False)+'\n')
    try:
        with server.open_service(session,config) as service:
            status=dict(status='READY',address=service.server.server_address,metadata=service.metadata,
                        source_sha256=manifest,worker=service.worker.snapshot(),model_reloaded=False)
            (source/'service-status.json').write_text(json.dumps(status,indent=2)+'\n')
            server.run_service(service)
        return dict(status='STOPPED')
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
