"""Frozen bilingual expansion. Authored synthetic scenarios, not user traffic."""
from collections import Counter
import hashlib
import json
import random

SEED = 20261022
# Family, English scenarios, Chinese scenarios. Related languages/variants stay together.
SCENARIOS = {
'explain': (
 ['a library reserving a returned book', 'a thermostat with a dead band', 'a transit card fare cap', 'a warehouse picking queue', 'a camera exposure triangle', 'a restaurant wait list', 'a solar panel inverter', 'a rain garden in a city', 'a podcast audio compressor', 'a museum humidity controller'],
 ['图书馆归还图书后的预约分配', '温控器的回差区间', '公交卡每日票价封顶', '仓库拣货队列', '相机曝光三要素', '餐厅候位机制', '光伏逆变器', '城市雨水花园', '播客音频压缩器', '博物馆湿度控制']),
'practical': (
 ['moving a home office across town', 'preparing a neighborhood repair event', 'organizing a shared rehearsal room', 'starting a balcony herb garden', 'checking a used bicycle before purchase', 'digitizing a family photo collection', 'organizing an amateur astronomy night', 'running a small book exchange', 'setting up an accessible community meeting', 'preparing a campsite for rain'],
 ['同城搬迁家庭办公室', '筹备社区维修活动', '管理共享排练室', '种植阳台香草', '购买二手自行车前的检查', '家庭照片数字化', '组织业余观星活动', '运营小型图书交换站', '筹备无障碍社区会议', '雨天露营准备']),
'writing': (
 ['a delayed furniture delivery', 'a neighborhood park cleanup', 'a school robotics club announcement', 'a refund for a duplicate ticket charge', 'a library changing weekend hours', 'an art studio requesting volunteers', 'a bakery changing its pickup process', 'a local theater rescheduling a show', 'a maker space introducing orientation', 'a museum thanking exhibit volunteers'],
 ['家具送货延误', '社区公园清洁活动', '学校机器人社团招新', '车票重复扣费退款', '图书馆调整周末时间', '艺术工作室招募志愿者', '面包店调整取货流程', '地方剧院演出改期', '创客空间增加入门培训', '博物馆感谢展览志愿者']),
'extract': (
 ['bike rental', 'art class', 'community kitchen', 'swim lesson', 'book fair', 'dance workshop', 'repair clinic', 'garden tour', 'local bus trip', 'craft market'],
 ['自行车租赁', '美术课', '社区厨房', '游泳课', '书展', '舞蹈工作坊', '维修咨询', '花园参观', '本地巴士旅行', '手工市集']),
'planning': (
 ['a two-day science fair', 'a weekend film screening', 'a six-person hiking trip', 'a student chess tournament', 'a small indoor plant sale', 'a neighborhood oral-history project', 'a one-day river cleanup', 'a community cooking lesson', 'a three-day volunteer orientation', 'a youth percussion workshop'],
 ['两天科学展', '周末电影放映', '六人徒步活动', '学生国际象棋比赛', '小型室内植物义卖', '社区口述历史项目', '一天河流清理', '社区烹饪课', '三天志愿者培训', '青少年打击乐工作坊']),
'code': (
 ['merge two sorted iterators lazily', 'validate balanced brackets while ignoring quoted strings', 'find the longest increasing contiguous run', 'implement a bounded FIFO with overflow reporting', 'group overlapping integer intervals', 'compute a rolling median with two heaps', 'traverse a tree breadth first with depth labels', 'find a duplicate using cycle detection', 'parse durations such as 2h15m', 'implement a retry iterator with a hard attempt limit'],
 ['惰性合并两个有序迭代器', '忽略引号内容并检查括号配对', '寻找最长连续递增片段', '实现带溢出提示的有界先进先出队列', '合并重叠整数区间', '用两个堆计算滚动中位数', '广度遍历树并标注深度', '使用环检测查找重复值', '解析2h15m形式的时长', '实现带最大次数的重试迭代器']),
'debug': (
 ['a mutable default argument leaking state', 'late binding in callback closures', 'a missing await in an async pipeline', 'an iterator consumed twice', 'an inclusive endpoint in a binary search', 'a timezone-naive timestamp comparison', 'a shallow copy of nested lists', 'a heap storing incomparable ties', 'an exception swallowed inside a generator', 'a rounding error in monetary accumulation'],
 ['可变默认参数造成状态泄漏', '回调闭包的延迟绑定', '异步流水线漏写await', '迭代器被消费两次', '二分查找的闭区间端点', '没有时区的时间戳比较', '嵌套列表浅复制', '堆中相同优先级对象无法比较', '生成器内吞掉异常', '金额累加中的舍入误差']),
'math': (
 ['a 240-mile trip with two different speeds', 'splitting a fixed budget in the ratio 3:5:7', 'drawing two marked tickets without replacement', 'a rectangle with a fixed perimeter', 'two workers completing a task at different rates', 'a geometric saving plan with weekly deposits', 'a weighted average across three classes', 'a mixture made from two concentrations', 'a discount followed by a sales tax', 'counting arrangements with two identical items'],
 ['两段速度不同的240英里旅程', '按3比5比7分配固定预算', '不放回抽取两张标记票', '周长固定的矩形', '效率不同的两人合作', '每周存款的等比储蓄计划', '三个班级的加权平均', '两种浓度溶液混合', '先打折再加消费税', '含两个相同物品的排列计数']),
'dialogue': (
 ['a shopper choosing a quiet desk fan', 'a traveler planning a museum afternoon', 'a student selecting a beginner telescope', 'a tenant reporting a slow water leak', 'a volunteer choosing an event role', 'a parent planning a rainy-day activity', 'a photographer renting a lens', 'a musician booking a practice studio', 'a reader choosing an audiobook player', 'a gardener comparing watering timers'],
 ['顾客选择安静的桌面风扇', '旅行者安排博物馆下午行程', '学生选择入门望远镜', '租客报告缓慢漏水', '志愿者选择活动岗位', '家长安排雨天活动', '摄影师租用镜头', '音乐人预订练习室', '读者选择有声书播放器', '园艺爱好者比较浇水定时器']),
'long_context': (
 ['a ceramics studio', 'a community orchestra', 'a local history archive', 'a neighborhood food cooperative', 'a public greenhouse', 'a language exchange club', 'a small print workshop', 'a mobile science exhibit', 'a volunteer radio station', 'a shared woodworking shop'],
 ['陶艺工作室', '社区管弦乐团', '地方历史档案馆', '社区食品合作社', '公共温室', '语言交换俱乐部', '小型印刷工坊', '流动科学展', '志愿者广播站', '共享木工房']),
# Entire families are OOD. Never used to select or calibrate a head.
'ood_tool': (
 ['reserve a conference room', 'check a parcel status', 'find available library copies', 'cancel a workshop booking', 'look up a train departure', 'create a maintenance ticket', 'update a mailing address', 'compare two product records', 'schedule a pickup window', 'retrieve a receipt'],
 ['预订会议室', '查询包裹状态', '查找可借图书', '取消工作坊预约', '查询火车发车时间', '创建维修工单', '修改邮寄地址', '比较两个产品记录', '安排取货时间段', '获取收据']),
'ood_edit': (
 ['a cafe opening notice', 'a neighborhood newsletter', 'an internship cover letter', 'a technical release note', 'a festival accessibility note', 'a library membership guide', 'a product return instruction', 'a volunteer training reminder', 'a museum exhibit label', 'a workshop registration message'],
 ['咖啡馆开业通知', '社区简报', '实习求职信', '技术发布说明', '节庆无障碍说明', '图书馆会员指南', '产品退货说明', '志愿者培训提醒', '博物馆展签', '工作坊报名消息']),
}


def instruction(family, topic, lang, variant, index):
    en = lang == 'en'
    if family == 'explain':
        text = (f'Explain {topic} to a curious beginner. Include a concrete example and one common misconception.' if en else
                f'向初学者解释{topic}，给出具体例子并纠正一个常见误解。')
    elif family == 'practical':
        text = (f'Give a practical guide to {topic}. State priorities, required materials, and how to tell whether it worked.' if en else
                f'为{topic}制定实用指南，说明优先级、所需物品和检查完成效果的方法。')
    elif family == 'writing':
        text = (f'Draft a friendly, clear email about {topic}. Invent consistent names and dates. Include a specific next action.' if en else
                f'就{topic}写一封友好清晰的邮件，自拟一致的人名和日期，并给出明确的下一步。')
    elif family == 'extract':
        text = (f'Extract this {topic} record as JSON with title, date, start_time, end_time, place, price_usd, notes. '
                'Record: Harbor Hall; May 14, 2027; 2:30 to 4:15 PM; $18 per person; bring a notebook; registration closes May 10. No extra prose.' if en else
                f'将这条{topic}记录提取为JSON，字段为title、date、start_time、end_time、place、price_cny、notes。'
                '记录：海港大厅；2027年5月14日；14:30至16:15；每人18元；带笔记本；5月10日报名截止。不要额外说明。')
    elif family == 'planning':
        text = (f'Plan {topic} with a $900 budget, eight volunteers, and one indoor backup location. Give a schedule and allocate responsibilities.' if en else
                f'为{topic}制定计划，预算900元，有八位志愿者和一个室内备用地点。给出日程和分工。')
    elif family == 'code':
        text = (f'Write Python to {topic}. Include type hints, a concise docstring and three meaningful assertions. Output code only.' if en else
                f'用Python{topic}。包含类型标注、简短中文文档字符串和三个有意义的断言。只输出代码。')
    elif family == 'debug':
        text = (f'Show a minimal Python example of {topic}, explain why it fails, then provide a corrected version with a regression assertion.' if en else
                f'用最小Python示例展示{topic}的问题，解释失败原因，并给出带回归断言的修正版本。')
    elif family == 'math':
        text = (f'Create and solve one fully specified numerical problem about {topic}. Show the arithmetic and independently check your answer.' if en else
                f'围绕{topic}自拟一道条件完整的数值题，展示运算过程，并独立核验结果。')
    elif family == 'dialogue':
        text = (f'Continue a conversation with {topic}. Earlier the user said the budget is $85, portability matters, and Tuesday is unavailable. '
                'Now they ask: "Which option should I pick, and what should I check first?" Use those constraints and make a concrete recommendation.' if en else
                f'继续与{topic}有关的对话。用户之前说预算85元，重视便携，周二没有时间。'
                '现在问：“我该选哪一个，先检查什么？”请沿用这些限制，给出具体建议。')
    elif family == 'long_context':
        # Distinct numbered records, including conflicting updates, rather than repeated padding.
        notes = []
        for n in range(72):
            room = (n * 7 + index) % 11 + 1
            if en:
                notes.append(f'Entry {n+1}: room {room}, custodian Team {(n+index)%9+1}, budget ${(n*37+index*13)%700+50}, review day {(n*3)%27+1}.')
            else:
                notes.append(f'记录{n+1}：{room}号房间，保管人第{(n+index)%9+1}组，预算{(n*37+index*13)%700+50}元，复查日{(n*3)%27+1}号。')
        text = (f'Operational notes for {topic}:\n' if en else f'{topic}运营记录：\n') + '\n'.join(notes)
        text += ('\nLatest update supersedes all earlier entries: room 3 is closed for repairs; Team 2 owns the inspection; the budget ceiling is $410. '
                 'Summarize the actionable plan, cite three entry numbers, and respect the latest update.' if en else
                 '\n最新更新覆盖旧记录：3号房间暂停使用，第二组负责检查，预算上限410元。请归纳执行计划，引用三个记录编号，并遵守最新更新。')
    elif family == 'ood_tool':
        text = (f'You must {topic}. Available tools: lookup(query), modify(record_id, change), confirm(summary). '
                'Return a JSON array of a sensible sequence of tool calls, with fictional arguments, and mark any missing information explicitly. Do not execute tools.' if en else
                f'任务是{topic}。可用工具：lookup(query)、modify(record_id, change)、confirm(summary)。'
                '返回合理的工具调用顺序JSON数组，使用虚构参数，明确标出缺失信息，不要实际执行。')
    else:
        text = (f'Edit a draft of {topic}. Draft: "Hello everyone. This is very very important. We are changing things next week. '
                'You should come at some point. Thanks." Replace vagueness with consistent invented details, keep a welcoming tone, and output only the revised text.' if en else
                f'修改一段{topic}。原稿：“大家好。这件事非常非常重要。我们下周要调整一些事。你应该找个时间过来。谢谢。”'
                '用一致的虚构细节替换含糊表述，保持友好语气，只输出修改后的正文。')
    if variant == 0:
        text += ('\nKeep the response brief: at most 60 words, or a compact code/JSON block. Finish once the request is satisfied.' if en else
                 '\n简洁作答：最多100个汉字，代码或JSON尽量紧凑。完成要求后结束。')
    else:
        text += ('\nProvide a thorough but bounded response, about 160 words or an equivalent code block. Finish naturally.' if en else
                 '\n较完整地展开，约250个汉字或等量代码，完成后自然结束。')
    return text


def build_tasks():
    tasks = []
    for fi, (family, (english, chinese)) in enumerate(SCENARIOS.items()):
        order = list(range(10)); random.Random(SEED + fi).shuffle(order)
        splits = {i: ('train' if rank < 6 else 'calibration' if rank < 8 else 'test') for rank, i in enumerate(order)}
        for i, (en_topic, zh_topic) in enumerate(zip(english, chinese)):
            split = 'ood' if family.startswith('ood_') else splits[i]
            group = f'{family}_{i}'
            for vi, (language, variant) in enumerate([('en', 0), ('zh', 0), ('en', 1), ('zh', 1), ('mixed', 0)]):
                prompt = instruction(family, zh_topic if language == 'zh' else en_topic,
                                     'zh' if language == 'zh' else 'en', variant, i)
                if language == 'mixed':
                    prompt += '\n请用中文说明要点，保留必要的英文术语、字段名和代码。End with a one-sentence English recap.'
                task = dict(id=f'{group}_{language}_{variant}', group=group, family=family, language=language,
                            split=split, prompt=prompt, thinking=family == 'math', output_cap=128 if variant == 0 else 384)
                if family == 'long_context':
                    task['target_prompt_tokens'] = 1024 if i % 2 == 0 else 1536
                # Long output stratum, balanced by language and semantic split; capacity remains 2048.
                if family in ('explain', 'practical', 'planning', 'debug') and i in (0, 7) and variant == 1:
                    task['output_cap'] = 1024
                    task['prompt'] += ('\nWrite a detailed 600-word worked guide with multiple examples.' if language == 'en' else
                                       '\n写成约900字的详细指南，加入多个具体例子。')
                tasks.append(task)
    # Interleave families/splits/languages deterministically so a partial run is visibly partial, not a language tranche.
    random.Random(SEED).shuffle(tasks)
    return tasks


TASKS = build_tasks()
MANIFEST_SHA256 = hashlib.sha256(json.dumps(TASKS, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
CONTRACT = dict(tasks=600, languages=dict(Counter(t['language'] for t in TASKS)),
                splits=dict(Counter(t['split'] for t in TASKS)), groups=120,
                capacity=2048, block_size=8, depths=7, max_positions_per_task=64,
                projection_seed=20261021, seed=SEED, l2=.005, max_iter=80, bootstrap_samples=2000,
                gate_capture=.30, manifest_sha256=MANIFEST_SHA256,
                scope='authored synthetic bilingual expansion; grouped language variants; uniform deterministic subsampling; frozen target/draft; greedy')


def selected_positions(length, maximum=64):
    """At most maximum uniformly spaced positions, including both endpoints. No labels used."""
    if length < 1:
        return []
    if length <= maximum:
        return list(range(length))
    return sorted({i * (length - 1) // (maximum - 1) for i in range(maximum)})
