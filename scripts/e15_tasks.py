"""New synthetic pilot groups, frozen before collection. No E3--E14 prompt reuse."""
import random

TOPICS = {
 'zh': ['B+树索引与哈希索引适用的查询', '布隆过滤器的误判及删除限制', '多版本并发控制中的旧版本回收',
        '纠删码在对象存储里的恢复过程', '令牌桶如何允许突发流量', '向量时钟如何判断并发事件',
        '固态硬盘写放大的来源', 'TLS握手中证书验证与密钥协商'],
 'en': ['monotonic clocks and wall clocks in timeout logic', 'read skew under database isolation levels',
        'leader leases and clock uncertainty', 'quorum intersection in replicated storage',
        'consistent hashing when adding a server', 'columnar compression and predicate pushdown',
        'event-time watermarks and late data', 'head-of-line blocking in multiplexed transports'],
 'code': ['iteratively flatten a nested list without recursion', 'find shortest paths with Dijkstra using a heap',
          'implement a small LRU cache with get and put', 'parse quoted CSV fields using a state machine',
          'rotate an array in place by k positions', 'implement a trie with prefix lookup',
          'encode and decode runs in a sequence', 'topologically sort a graph and detect cycles'],
 'json': ['a planetarium observing schedule', 'an international supply shipment', 'a bicycle repair shop work order',
          'a community festival program', 'a planetary rover command sequence', 'a chemical reagent catalog',
          'a harbor berth assignment', 'a laboratory experiment configuration'],
 'math': ['A 12-liter solution is 20% salt. How much pure water makes it 15% salt?',
          'A deposit of 800 grows by 5% annually for three years. Find the final value.',
          'A bag has 5 red and 7 blue balls. Draw three without replacement. Find the probability all are blue.',
          'An arithmetic sequence starts at 7 with difference 4. Find the sum of its first 30 terms.',
          'A boat travels 36 km downstream in 2 hours and upstream in 3 hours. Find boat and current speeds.',
          'How many five-digit codes have distinct digits and do not start with zero?',
          'Find the smallest positive integer that leaves remainder 2 modulo 5 and remainder 3 modulo 7.',
          'One pump fills a pool in 6 hours, another in 9 hours. How long do they take together?'],
 'structured': ['a Markdown decision table comparing three backup methods', 'a CSV inventory with item, quantity, unit_price and supplier',
                'a YAML deployment with two services and a shared network', 'a TOML configuration for a fictional data processor',
                'an XML catalog containing six fictional books', 'a Mermaid state diagram for a parcel delivery lifecycle',
                'a shell script that safely summarizes file sizes by extension', 'a numbered checklist for validating a regular expression'],
}
TASKS=[]
for family_index,(family,topics) in enumerate(TOPICS.items()):
    shuffled=list(range(8));random.Random(20261021+family_index).shuffle(shuffled)
    splits={i:'train' if rank<5 else 'calibration' if rank<7 else 'test' for rank,i in enumerate(shuffled)}
    for i,topic in enumerate(topics):
        for variant in range(2):
            if family=='zh': prompt=f'请解释{topic}，给出具体例子和一个局限，约180字。'
            elif family=='en': prompt=f'Explain {topic} in about 180 words, with a concrete example and a limitation.'
            elif family=='code': prompt=f'Write Python code to {topic}. Include a docstring, edge cases and four assertions. Output code only.'
            elif family=='json': prompt=f'Return only a JSON object describing {topic}. Include a title, metadata, six detailed entries and a boolean finalized flag.'
            elif family=='math': prompt=topic+' Explain the reasoning and verify the numerical result.'
            else: prompt=f'Produce {topic}. Use concrete fictional values and at least six meaningful entries.'
            if variant: prompt+='\nOrganize the answer carefully and include one additional practical detail.'
            task=dict(id=f'{family}_{i}_{variant}',group=f'{family}_{i}',family=family,split=splits[i],
                      prompt=prompt,thinking=family=='math',output_cap=64 if not variant else 96)
            if variant and i in (1,5):
                task['prompt']=('Reference notes describe independent components, explicit identifiers, bounded resources, '
                                'and reproducible checks. Correctness depends on clear assumptions and well-defined boundaries. ')*180+'\n'+prompt
                task['target_prompt_tokens']=1024 if family_index%2==0 else 1536
            TASKS.append(task)
for i,topic in enumerate(['rank sales by region with window functions','find customers with no purchases using an anti-join',
                         'calculate a seven-day moving average','deduplicate rows while retaining the latest timestamp',
                         'compute session boundaries from event gaps','find gaps in a sequence of dates',
                         'aggregate a bill of materials with a recursive query','compare monthly retention cohorts']):
    TASKS.append(dict(id=f'ood_sql_{i}',group=f'ood_sql_{i}',family='sql',split='ood',thinking=False,
                      output_cap=96,prompt=f'Write standard SQL to {topic}. Define a small illustrative schema and explain one edge case.'))

CONTRACT=dict(tasks=104,block_size=8,depths=7,capacity=2048,prefill_rows=8,
              groups=['f0','f1'],l2=.005,max_iter=80,projection_seed=20261021,
              train_split='60 prompts',calibration_split='24 prompts',test_split='12 prompts',ood_split='8 SQL prompts',
              gate_capture=.30,bootstrap_samples=2000,
              scope='pilot; uniform serial prefixes; new synthetic prompts; grouped splits; frozen target and draft weights')
