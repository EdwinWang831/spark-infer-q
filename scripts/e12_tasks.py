"""E12 prompts and splits frozen before observing routes or timings."""

TASKS = [
    dict(id='cal_cache', split='train', thinking=False,
         prompt='用中文写一段约150字的说明，解释浏览器缓存的有效期、验证请求和失效机制，并举一个具体例子。'),
    dict(id='cal_json', split='train', thinking=False,
         prompt='Return only a JSON array of eight fictional library books. Each object must have title, author, year, and available fields. Use varied values.'),
    dict(id='cal_code', split='train', thinking=False,
         prompt='Write a Python function merge_intervals(intervals) with a docstring and input validation. Sort intervals and merge overlaps. Output code only.'),
    dict(id='cal_reason', split='train', thinking=True,
         prompt='A workshop makes 37 parts each day for 8 days. It discards 19 defective parts and ships 126. Explain the calculation and determine how many parts remain.'),
    dict(id='cal_english', split='train', thinking=False,
         prompt='Write roughly 120 words explaining why database indexes can speed up reads but slow down writes. Include a concrete example and one tradeoff.'),
    dict(id='cal_long', split='train', thinking=False,
         prompt=('A service records events from sensors. Each event has a device ID, timestamp, sequence number, and value. '
                 'Devices can reconnect and resend old events. Network delivery can be delayed or duplicated. '
                 'The server keeps a durable event log and a current-value table. Operators need to recover after a crash. '
                 'Given these constraints, write a detailed design of approximately 200 words covering deduplication, ordering, persistence, and recovery.')),
    dict(id='test_zh', split='test', thinking=False,
         prompt='用中文写一段约150字的说明，解释数据库事务的原子性和隔离性，用转账举例，并指出一个并发风险。'),
    dict(id='test_json', split='test', thinking=False,
         prompt='Return only a JSON array of eight fictional weather stations. Each object must have city, temperature, humidity, and online fields. Use varied values.'),
    dict(id='test_code', split='test', thinking=False,
         prompt='Write a Python function group_anagrams(words) with a docstring and input validation. Preserve the order of groups and words. Output code only.'),
    dict(id='test_reason', split='test', thinking=True,
         prompt='A school buys 29 boxes of 16 pencils. It distributes 187 pencils and receives another 53. Explain the calculation and determine how many pencils remain.'),
    dict(id='test_english', split='test', thinking=False,
         prompt='Write roughly 120 words explaining how a message queue handles retries and why duplicate processing can occur. Include a concrete example and one tradeoff.'),
    dict(id='test_long', split='test', thinking=False,
         prompt=('An application uploads photographs from mobile devices. Each upload has an account ID, content hash, creation time, and retry counter. '
                 'Devices can lose connectivity and resend an upload. The server uses object storage and a metadata database. '
                 'An upload may finish storing its object before the database transaction completes. Operators need to recover after a crash. '
                 'Given these constraints, write a detailed design of approximately 200 words covering deduplication, authorization, persistence, and recovery.')),
]

CONTRACT = dict(widths=list(range(1, 9)), repetitions=3, positions_per_prompt=6,
                behavior_width=4, max_new_tokens=160, capacity=1024,
                split='six frozen calibration prompts and six held-out prompts; never split repeated steps randomly',
                model_gate_relative_mae_reduction=0.25)
