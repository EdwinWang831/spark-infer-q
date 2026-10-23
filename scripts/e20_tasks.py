"""Eight frozen training-only collector probes; never read evaluation labels."""
import hashlib

from e16_tasks import TASKS
from e17_run_contract import load_tasks


def get_tasks(root):
    chosen, used = [], set()
    sources = [('external', 'en', 384), ('external', 'zh', 384),
               ('code', 'en', 192), ('code', 'mixed', 192),
               ('extract', 'zh', 128), ('extract', 'mixed', 128),
               ('long_context', 'en', 128), ('long_context', 'zh', 128)]
    external = load_tasks(root)
    for family, language, cap in sources:
        pool = external if family == 'external' else [t for t in TASKS if t['family'] == family]
        eligible = [t for t in pool if t['split'] == 'train' and t['language'] == language and t['group'] not in used]
        task = min(eligible, key=lambda t: hashlib.sha256(('e20-collector-v1'+t['id']).encode()).hexdigest())
        used.add(task['group'])
        chosen.append(dict(task, output_cap=cap, probe_calibration=family in ('external', 'long_context')))
    assert len(chosen) == len(used) == 8
    return chosen
