"""Optional Linux owner-thread placement; never change clocks or other threads."""
from contextlib import contextmanager
from pathlib import Path
import os


@contextmanager
def worker_cpu(policy='default',*,system=None,frequency_root=Path('/sys/devices/system/cpu')):
    if policy not in ('default','performance'):raise ValueError('unknown worker CPU policy')
    if policy=='default':
        yield dict(worker_cpu_policy='default');return
    system=os if system is None else system
    if not all(hasattr(system,n) for n in ('sched_getaffinity','sched_setaffinity')):
        raise RuntimeError('performance CPU placement requires Linux thread affinity')
    previous=set(system.sched_getaffinity(0))
    if not previous:raise RuntimeError('worker has no allowed CPUs')
    try:
        frequencies={cpu:int((Path(frequency_root)/('cpu'+str(cpu))/'cpufreq/cpuinfo_max_freq').read_text()) for cpu in previous}
    except (OSError,ValueError) as exc:
        raise RuntimeError('cannot read allowed CPU maximum frequencies') from exc
    if min(frequencies.values())<=0:raise RuntimeError('invalid CPU maximum frequency')
    selected={cpu for cpu,freq in frequencies.items() if freq==max(frequencies.values())}
    applied=False
    try:
        # Linux pid0 means this calling owner thread, not the whole process.
        system.sched_setaffinity(0,selected);applied=True
        yield dict(worker_cpu_policy='performance',worker_cpu_affinity=sorted(selected))
    finally:
        if applied:system.sched_setaffinity(0,previous)
