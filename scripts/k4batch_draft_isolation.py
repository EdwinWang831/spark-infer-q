"""Real GPU draft isolation, cache-tail retry, and generation reuse checks."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_k4_draft_isolation';pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    torch=session.torch;cases=[]
    try:
        opt=importlib.import_module(tag+'.kaifra_optimized');batch=importlib.import_module(tag+'.kaifra_batch')
        dm=importlib.import_module(tag+'.draft_batch');loader=importlib.import_module(tag+'.draft_batch_w8')
        component=json.loads((Path(arguments.get('component_root',
            '/workspace/e1/k4batch-20261023-batch-draft-component-v5'))/'batch-draft-result.json').read_text())
        matmul,identity=loader.load_library(component['kernel']['library'],component['kernel']['binary_sha256'])
        base,_=opt.optimized_provider(session)
        own=batch.BatchedRequestDraftProvider(base,matmul,kernel_identity=identity);backend=own.backend
        torch.manual_seed(41)
        features=[torch.randn(n,12800,device='cuda',dtype=torch.bfloat16) for n in (23,61,261)]
        a=dm.DraftInput(('a',1),features[0],17);b=dm.DraftInput(('b',1),features[1],31)
        original,_=backend.forward_hidden([a,b])
        changed=dm.DraftInput(('b',2),features[1]*3,99)
        alternative,_=backend.forward_hidden([a,changed])
        assert torch.equal(original[0],alternative[0]),'companion contents changed another request'
        cases.append(dict(case='companion_content_isolation',hidden_exact=True,unequal_contexts=True))
        handle=('reuse',1);request=dm.DraftInput(handle,features[2][:260],17)
        backend.propose_batch([request]);prefix=backend.caches[handle].keys[:,:260].clone()
        original_linear=backend.worker.linear
        def fail(x,name):
            if name=='layers.2.self_attn.k_proj.weight':raise RuntimeError('injected partial draft cache append')
            return original_linear(x,name)
        backend.worker.linear=fail
        try:
            try:backend.propose_batch([dm.DraftInput(handle,features[2],17)])
            except RuntimeError as exc:assert 'injected partial' in str(exc)
            else:raise AssertionError('injected failure was not raised')
        finally:backend.worker.linear=original_linear
        assert backend.caches[handle].rows==260
        assert torch.equal(prefix,backend.caches[handle].keys[:,:260])
        retried=backend.propose_batch([dm.DraftInput(handle,features[2],17)])
        assert backend.caches[handle].rows==261
        assert torch.equal(prefix,backend.caches[handle].keys[:,:260])
        cases.append(dict(case='partial_append_retry',old_prefix_exact=True,published_rows=261))
        # Replacing the same ID with a new generation and different prefix must
        # agree with a fresh backend, never with the released cache contents.
        backend.release(handle);assert not backend.caches
        request=dm.DraftInput(('reuse',2),features[2].flip(0).contiguous(),17)
        reused=backend.propose_batch([request])
        fresh=batch.BatchedRequestDraftProvider(base,matmul,kernel_identity=identity)
        expected=fresh.backend.propose_batch([request])
        assert reused==expected
        cases.append(dict(case='same_id_new_generation',proposals_and_signals_exact=True))
        before={h:c.rows for h,c in backend.caches.items()}
        try:backend.propose_batch([dm.DraftInput(('new',1),features[2],1),dm.DraftInput(('bad',1),features[2][:0],1)])
        except ValueError:pass
        else:raise AssertionError('invalid later context accepted')
        assert {h:c.rows for h,c in backend.caches.items()}==before
        cases.append(dict(case='prevalidate_entire_batch',cache_unchanged=True))
        for h in list(backend.caches):backend.release(h)
        requests=[dm.DraftInput((str(i),1),features[2][:260],17) for i in range(4)]
        backend.propose_batch(requests);assert len(backend.caches)==4
        fifth=dm.DraftInput(('fifth',1),features[2][:260],17)
        try:backend.propose_batch([fifth])
        except ValueError:pass
        else:raise AssertionError('unbounded cache admission')
        backend.release(requests[0].handle);backend.propose_batch([fifth]);assert len(backend.caches)==4
        for h in list(backend.caches):backend.release(h)
        for h in list(fresh.backend.caches):fresh.release(h)
        cases.append(dict(case='bounded_cache_admission',maximum_requests=4,released_slots_reusable=True))
        return dict(status='PASS_K4_BATCH_DRAFT_ISOLATION',cases=cases,source_sha256=manifest,
            kernel=identity,model_reloaded=False,synthetic_features=True)
    finally:
        torch.cuda.synchronize()
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.'):sys.modules.pop(n,None)
