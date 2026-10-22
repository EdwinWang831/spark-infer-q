"""Frozen E7 comparison: E6 tasks plus two earlier capped-output controls."""
from e6_tasks import TASKS as E6_TASKS
from e4_tasks import TASKS as E4_TASKS

TASKS=[dict(t,arm='natural') for t in E6_TASKS]+[dict(t) for t in E4_TASKS if t['arm']=='capped']
MODES={'native':('serial','native_mtp'),'ours':('serial','fixed7','dynamic7')}
SESSIONS=(('native',0),('ours',0),('ours',1),('native',1))


def ordered_modes(engine,rep,task_index):
    modes=MODES[engine]
    if engine=='native':return modes if (rep+task_index)%2==0 else modes[::-1]
    offset=(rep+task_index)%len(modes)
    return modes[offset:]+modes[:offset]
