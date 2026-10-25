"""Measure the actual native token binding; no GPU inference or fitting."""
import importlib
import statistics
import time


def execute(session,source,tag,args,load,report):
    native_mod=importlib.import_module(tag+'.native_scheduler')
    rows=[]
    with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=0) as native:
        for n in (2048,261120):
            data=native.core.SequenceData([1001]*n+[17]*512)
            for mode in ('full_binding_then_slice','output_only_python_cache'):
                samples=[]
                cached=[17]*512
                for rep in range(5):
                    start=time.perf_counter()
                    for _ in range(100):
                        out=tuple(data.get_token_ids()[n:]) if mode=='full_binding_then_slice' else tuple(cached)
                        assert len(out)==512 and out[-1]==17
                    samples.append((time.perf_counter()-start)*1000/100)
                rows.append(dict(prompt_tokens=n,mode=mode,median_ms_per_read=statistics.median(samples),samples_ms=samples))
    report.update(status='PASS_E50_NATIVE_READ_COST',reads=rows,scope='native binding CPU microbenchmark; cached publication still requires engine correctness validation')
    return report
