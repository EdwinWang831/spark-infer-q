"""Frozen before the online confidence trial; no fitting on these outputs."""
TASKS = [
    dict(id='zh_storage', thinking=False, prompt='用约200字解释数据库的预写日志如何帮助崩溃恢复，区分已经提交和未提交的事务，并举一个转账的例子。'),
    dict(id='json_weather', thinking=False, prompt='Return only a JSON object for a fictional weather station. Include name, location, sensors as eight objects with id, type and unit, and a maintenance schedule with four entries.'),
    dict(id='code_merge', thinking=False, prompt='Write a Python merge_intervals(intervals) function. Include a docstring, empty input, touching intervals, and six example assertions. Explain its complexity in a code comment. Output only code.'),
    dict(id='math_train', thinking=True, prompt='Two trains leave the same station. The first travels at 72 km/h and departs 25 minutes earlier. The second travels at 90 km/h. Find when and where the second catches the first, showing the calculation.'),
    dict(id='en_cache', thinking=False, prompt='Explain cache invalidation in about 200 words. Compare time-based expiration with explicit invalidation, and describe one concrete race condition and its remedy.'),
    dict(id='zh_network', thinking=False, prompt='用约200字解释TCP拥塞控制和接收端流量控制分别解决什么问题，说明窗口如何限制发送，并举一个慢接收端的例子。'),
]
BACKGROUNDS = [
    ('long_1024', 1024, 'A warehouse records stock changes as immutable events. Each event has an item ID, quantity, timestamp, and unique operation ID. Inventory snapshots are taken hourly. ', 'Explain a safe recovery procedure after a crash, including deduplication, snapshots and replay, in about 200 words.'),
    ('long_1536', 1536, 'A weather sensor records measurements with a station ID and sequence number. Connections can fail and retransmit packets. A central service computes hourly aggregates and retains original measurements. ', 'Describe how to ingest delayed and duplicate measurements without corrupting hourly aggregates, in about 200 words.'),
    ('long_1792', 1792, 'A document service stores immutable revisions and a mutable pointer to the latest revision. Multiple editors can submit changes concurrently. Clients retry requests after network timeouts. ', 'Describe a concurrency-safe revision protocol with retries and conflict detection, in about 200 words.'),
]
TASKS += [dict(id=name, thinking=False, prompt=body*180+'\n'+question, target_prompt_tokens=n)
          for name, n, body, question in BACKGROUNDS]
CONTRACT = dict(modes=['native_cached', 'depth_budget', 'margin_budget'], repetitions=4,
                output_cap=128, max_depth=5, capacity=2048, prefill_rows=8,
                cost='frozen E6 verify+commit curve; post-draft arms add same frozen draft cost to every width, including width1',
                acceptance='frozen E13 train-only median/Beta table; no online refitting',
                scope='greedy decode; cache build/append, signal GPU work/transfer, budget and runtime calls included; prompt, clones, oracle excluded')
