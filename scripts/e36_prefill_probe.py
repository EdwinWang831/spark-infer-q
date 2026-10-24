"""Resident prefill preflight and same-input numerical diagnostics, never a promotion."""
import hashlib
import json
from pathlib import Path


TEXTS = (
    ('english', 64, 'Explain the difference between a cache and a queue, with a concrete example. '),
    ('chinese', 64, '请解释缓存和队列的区别，并分别给出一个具体例子。'),
    ('code', 512, 'Write a Python function that removes duplicates while preserving order. Explain the loop invariant and test empty inputs. '),
    ('mixed', 512, 'Document: A cache reuses computed results. A queue preserves work for later processing. 缓存保存可复用结果，队列保存待处理任务。 Compare the two mechanisms using the document. '),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(session):
    cases = [dict(name='historical4', ids=[100,101,102,103])]
    for name, length, text in TEXTS:
        ids = session.tokenizer.encode(text * 80, add_special_tokens=False)[:length]
        assert len(ids) == length
        cases.append(dict(name=name, ids=ids, text_sha256=hashlib.sha256(text.encode()).hexdigest()))
    for c in cases:
        c['token_sha256'] = hashlib.sha256(json.dumps(c['ids'], separators=(',', ':')).encode()).hexdigest()
        c['length'] = len(c['ids'])
    return cases


def run(session, arguments):
    source = Path(arguments['source_root'])
    manifest = json.loads((source/'manifest.json').read_text())
    for name, checksum in manifest.items():
        assert sha(source/name) == checksum, name
    frozen = json.loads((session.root/'artifacts/e33-online-seal.json').read_text())
    external = session.root.parent/'e2-load/reference/tensorfold'
    for name, checksum in frozen['external_source_sha256'].items():
        assert sha(external/name) == checksum, name
    torch = session.torch
    assert not torch.backends.cuda.matmul.allow_tf32
    assert torch.get_float32_matmul_precision() == 'highest'
    cases = inputs(session)
    result = dict(status='PASS_E36_PREFLIGHT', source_sha256=manifest,
        external_source_sha256=frozen['external_source_sha256'], inputs=cases,
        target_object_id=id(session.weights), draft_object_id=id(session.draft),
        model_reloaded=False, tf32_disabled=True, allocated_before=torch.cuda.memory_allocated(),
        shape=dict(layers=len(session.weights.layers), hidden=session.weights.cfg.hidden,
                   gdn_layers=[l.index for l in session.weights.layers if l.linear]))
    if arguments['phase'] in ('diagnose','localize','candidate','audit'):
        import importlib.util
        import sys
        phase=arguments['phase']
        tag='_e36_'+phase+'_'+hashlib.sha256(str(source).encode()).hexdigest()[:10]
        runner={'diagnose':'e36_diagnose.py','localize':'e36_localize.py','candidate':'e36_candidate_validate.py','audit':'e36_audit.py'}[phase]
        spec=importlib.util.spec_from_file_location(tag,source/('scripts/'+runner))
        module=importlib.util.module_from_spec(spec);sys.modules[tag]=module
        try:
            spec.loader.exec_module(module)
            result.update(module.audit(session,source) if phase=='audit' else getattr(module,phase)(session,source,cases))
        finally:
            sys.modules.pop(tag,None)
    assert tuple(result[k] for k in ('target_object_id','draft_object_id')) == (id(session.weights),id(session.draft))
    for name, checksum in manifest.items():
        assert sha(source/name) == checksum, name
    return result
