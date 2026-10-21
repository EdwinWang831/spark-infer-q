"""Prompt splits frozen before collecting local-policy labels or test outputs."""


def task(name,prompt,thinking=False):
    return {'id':name,'prompt':prompt,'thinking':thinking,'limit':512 if thinking else 256}


TRAIN = [
    task('train_sum','只输出 31 加 46 的结果。'),
    task('train_translation','Translate See you tomorrow into Chinese. Output only the translation.'),
    task('train_city','Name the capital of Italy in one short sentence.'),
    task('train_json','Return only JSON with fields color equal to blue and enabled equal to true.'),
    task('train_square','Write a Python function square(x). Output only the function.'),
    task('train_reverse','Write a Python function that reverses a string. Output only code.'),
    task('train_loop','Write a Python function that sums a list using a loop. Output only code.'),
    task('train_dict','Write a short Python example counting repeated words with a dictionary.'),
    task('train_bandwidth','用两句话解释为什么内存带宽会限制推理速度。'),
    task('train_batch','Explain in two sentences how batching can improve GPU utilization.'),
    task('train_kv','用两句话解释 KV cache 的用途和代价。'),
    task('train_experts','Explain sparse mixture-of-experts routing in two short sentences.'),
    task('train_bf16','用两句话说明 BF16 和 FP32 的区别。'),
    task('train_speculation','Explain why speculative decoding can reject draft tokens. Use two sentences.'),
    task('train_reason_math','A train has 6 cars with 18 seats each. There are 79 passengers. How many seats remain? Give the final number after reasoning.',True),
    task('train_reason_code','Write a short Python function returning the smaller of two values without min(). Include an example.',True),
]

CALIBRATION = [
    task('cal_product','只输出 6 乘以 7 的结果。'),
    task('cal_translation','将 Welcome back 翻译成中文，只输出译文。'),
    task('cal_json','Return only a JSON object with city set to Rome and score set to 9.'),
    task('cal_code','Write Python code for a function cube(x). Output only code.'),
    task('cal_context','用两句话解释为什么更长的上下文会占用更多缓存。'),
    task('cal_reason','A shelf has 5 rows of 16 books. Thirteen are borrowed. How many remain? Give the final number after reasoning.',True),
]

TEST = [
    task('test_product','只输出 8 乘以 9 的结果，不要解释。'),
    task('test_city','What is the capital of Japan? Answer in one short sentence.'),
    task('test_translation','将 Good night 翻译成中文，只输出译文。'),
    task('test_json','只输出合法 JSON，字段 animal 为 cat，字段 age 为整数 4。'),
    task('test_code','Write a Python function clamp(x, low, high) without using min or max. Output only the function.'),
    task('test_explanation','用两句话解释为什么投机解码有时会比逐 token 解码更慢。'),
    task('test_reason_math','A garden has 9 rows of 11 plants. Seventeen plants are removed. How many remain? Give the final number after reasoning.',True),
    task('test_reason_code','Write a short Python function that returns the middle of three distinct numbers without sorting. Include one example.',True),
]
