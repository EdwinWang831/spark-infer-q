"""Two diagnosis cases plus four fresh training-only collector confirmation roots."""
import hashlib

from e16_tasks import TASKS
from e17_run_contract import load_tasks
from e20_tasks import get_tasks as first_probe


def get_tasks(root):
    earlier = first_probe(root)
    chosen = earlier[:2]
    used = {t['group'] for t in earlier}
    external = load_tasks(root)
    for family, language, cap in [('external', 'en', 192), ('external', 'zh', 192),
                                   ('code', 'mixed', 192), ('long_context', 'zh', 128)]:
        pool = external if family == 'external' else [t for t in TASKS if t['family'] == family]
        eligible = [t for t in pool if t['split'] == 'train' and t['language'] == language and t['group'] not in used]
        task = min(eligible, key=lambda t: hashlib.sha256(('e21-collector-v1'+t['id']).encode()).hexdigest())
        used.add(task['group'])
        chosen.append(dict(task, output_cap=cap, probe_calibration=family == 'long_context'))
    assert len(chosen) == len({t['group'] for t in chosen}) == 6
    return chosen
