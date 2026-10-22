"""Inputs frozen before E6 GPU outputs; no vocabulary fitting on these tasks."""
CALIBRATION=[
    dict(id='cal_en',prompt='Explain in two sentences why a cache needs an eviction policy.',thinking=False,limit=96),
    dict(id='cal_zh',prompt='用两句话解释什么是先进先出队列。',thinking=False,limit=96),
]
TASKS=[
    dict(id='zh_prose',prompt='用两句话解释：为什么备份不能代替数据库事务？不要分点。',thinking=False,limit=128,check='prose'),
    dict(id='zh_json',prompt='仅输出一个 JSON 对象，恰好包含两个字段：城市为上海，数量为整数3。不要解释。',thinking=False,limit=128,check='json'),
    dict(id='latex',prompt=r'写出 ax^2+bx+c=0（a\ne 0）的求根公式。只给出 LaTeX，不要代码块。',thinking=False,limit=128,check='latex'),
    dict(id='even',prompt='Write a pure Python function count_even(nums) that counts the even integers in a list. An empty list returns 0. Output only code, without imports or examples.',thinking=False,limit=128,check='even'),
    dict(id='merge',prompt='Write a pure Python function merge_counts(left, right). Both arguments map strings to integers. Return a new dictionary with the sum for each key, without changing either input. Output only code, without imports or examples.',thinking=False,limit=160,check='merge'),
    dict(id='integer',prompt='Compute (37 times 19) minus 108. Return only the integer.',thinking=False,limit=64,check='integer',expected=595),
    dict(id='en_prose',prompt='In two sentences, explain why a hash table can become slower when many keys collide. Mention one mitigation.',thinking=False,limit=128,check='prose'),
    dict(id='thinking_math',prompt='A shop has 23 boxes with 17 pens each and sells 86 pens. How many pens remain? End your answer with the remaining integer.',thinking=True,limit=256,check='last_integer',expected=305),
]
HEADS=('bf16_full','bf16_subset','q4_full','q4_subset')
POLICIES=('fixed7','dynamic7')


def order(rep,task_index):
    offset=(rep+task_index)%4
    heads=HEADS[offset:]+HEADS[:offset]
    policies=POLICIES if (rep+task_index)%2==0 else POLICIES[::-1]
    return [(head,policy) for head in heads for policy in policies]
