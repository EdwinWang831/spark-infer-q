"""New E4 prompts frozen before inspecting their GPU outputs."""

BASE = [
    ('arithmetic', 'Compute 17 times 8. Return only the integer.', 'integer', 136),
    ('json', 'Return only a JSON object with exactly these fields: name is otter, count is integer 5, enabled is false.', 'json', None),
    ('positive', 'Write a pure Python function sum_positive(nums) that sums only values greater than zero. Return 0 for an empty list. Output only code, without imports or examples.', 'positive', None),
    ('unique', 'Write a pure Python function unique_ordered(items) that removes duplicates while preserving their first occurrence. Inputs are integers. Output only code, without imports or examples.', 'unique', None),
    ('prose', 'Explain the difference between caching model weights and caching a conversation prefix. Use one paragraph of four sentences and include one practical tradeoff.', 'prose', None),
    ('word_math', 'A nursery has 17 trays with 23 seedlings in each tray. It sells 41 seedlings. How many remain? End your answer with the remaining integer.', 'last_integer', 350),
]

TASKS = []
for name, prompt, check, expected in BASE:
    for thinking in (False, True):
        TASKS.append(dict(id=f'{name}_{"think" if thinking else "direct"}', prompt=prompt,
            thinking=thinking, limit=512 if thinking else 256, arm='natural', check=check, expected=expected))

TASKS += [
    dict(id='length_prose', prompt='Write a detailed tutorial of at least 1200 words on designing a reliable key-value cache, with sections on eviction, invalidation, concurrency, and testing. Start the tutorial immediately.', thinking=False, limit=256, arm='capped', check='prose', expected=None),
    dict(id='length_code', prompt='Write a complete Python implementation of a small LRU cache with get, put, delete, capacity validation, clear, and at least ten unittest test methods. Output only the full Python code. Include explanatory docstrings.', thinking=False, limit=256, arm='capped', check='prefix', expected=None),
]

MODES = ('serial', 'fixed6', 'fixed7', 'dynamic7')


def ordered_modes(repetition, task_index=0):
    offset = (repetition + task_index) % len(MODES)
    return MODES[offset:] + MODES[:offset]
