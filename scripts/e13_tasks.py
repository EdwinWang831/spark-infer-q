"""Frozen new prompts; E12 test prompts are not reused for tuning."""
TASKS = [
    dict(id='cal_zh',split='train',thinking=False,prompt='用约150字解释幂等性，并说明支付重试为什么需要幂等键，给出一个具体例子。'),
    dict(id='cal_json',split='train',thinking=False,prompt='Return only a JSON object describing a fictional spacecraft. Include name, crew as an array of four objects, instruments as six objects, and a boolean ready flag.'),
    dict(id='cal_code',split='train',thinking=False,prompt='Write a Python binary_search_left(values, key) function returning the first matching index or -1. Include a docstring, empty-input handling and five assertions. Output only code.'),
    dict(id='cal_math',split='train',thinking=True,prompt='A tank is initially 3/5 full. Adding 24 liters makes it 4/5 full. Then 18 liters are removed. Explain how to find its capacity and final water volume.'),
    dict(id='cal_en',split='train',thinking=False,prompt='Explain backpressure in a streaming pipeline in about 150 words. Discuss bounded queues, slow consumers and one practical tradeoff.'),
    dict(id='cal_context',split='train',thinking=False,prompt=('A replicated log stores ordered commands. Followers acknowledge durable writes. A leader can fail during a retry. Clients attach unique request identifiers. '*12)+'Write a 150-word recovery design addressing leader changes, deduplication and committed results.'),
    dict(id='test_zh',split='test',thinking=False,prompt='用约150字解释写时复制如何支持快照，说明修改数据时的过程，并指出一个空间开销。'),
    dict(id='test_json',split='test',thinking=False,prompt='Return only a JSON object describing a fictional museum. Include name, staff as an array of four objects, exhibits as six objects, and a boolean open flag.'),
    dict(id='test_code',split='test',thinking=False,prompt='Write a Python top_k_frequent(items, k) function using a frequency dictionary. Include a docstring, tie handling and five assertions. Output only code.'),
    dict(id='test_math',split='test',thinking=True,prompt='A machine completes 7 tasks every 12 minutes. It runs for 96 minutes, then another machine completes 19 tasks. Explain the calculation and find the total.'),
    dict(id='test_en',split='test',thinking=False,prompt='Explain DNS time-to-live in about 150 words. Discuss recursive resolvers, address changes and one practical tradeoff.'),
    dict(id='test_context',split='test',thinking=False,prompt=('An event consumer stores offsets and a materialized view. Messages can be repeated after a network outage. Handlers update durable records. Each event carries a unique identifier. '*12)+'Write a 150-word recovery design addressing duplicate events, transactions and checkpoints.'),
]
CONTRACT = dict(output_cap=96,repetitions=4,prefill_rows=8,capacity=1024,
                modes=['original','cached'], primary_signal='edge_margin',
                confidence_rule='per-depth train-median split, Beta(1,1) smoothing',
                confidence_evaluation='conditional observed-depth Brier and log-loss; first rejection/EOS/cap censored',
                scope='bounded greedy decode from shared prompt state; excludes prompt/network/state cloning/oracle')
