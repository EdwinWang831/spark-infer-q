"""Independent online ablation prompts, never included in E16 fitting or selection."""
TASKS = [
 dict(id='ab_en_explain',language='en',family='explain',thinking=False,prompt='Explain why an elevator dispatch system may keep someone waiting even when a nearby car is empty. Give two examples and one improvement. Use about 100 words.',output_cap=256),
 dict(id='ab_zh_explain',language='zh',family='explain',thinking=False,prompt='解释为什么超市收银队列看起来最短也不一定最快，举两个例子并给出改进办法，约160字。',output_cap=256),
 dict(id='ab_en_write',language='en',family='writing',thinking=False,prompt='Write a polite 100-word email asking a community center to move a pottery class from Wednesday to Saturday. Mention an alternative time and ask for confirmation.',output_cap=256),
 dict(id='ab_zh_write',language='zh',family='writing',thinking=False,prompt='写一封简短邮件，请社区中心将合唱排练从周四调整至周日，给出备用时间并请求确认，约150字。',output_cap=256),
 dict(id='ab_en_json',language='en',family='extract',thinking=False,prompt='Return only JSON for this event: Riverside Chess Day, June 8 2027, 10 AM to noon, room 4, free, bring a chess clock. Fields: title, date, start, end, room, price, bring.',output_cap=256),
 dict(id='ab_zh_json',language='zh',family='extract',thinking=False,prompt='只返回JSON：河畔绘画日，2027年6月9日，上午9点至11点，5号室，免费，带画笔。字段title、date、start、end、room、price、bring。',output_cap=256),
 dict(id='ab_en_code',language='en',family='code',thinking=False,prompt='Write a Python function that returns the first nonrepeating character in a string, or None. Include a docstring and three assertions. Output only code.',output_cap=384),
 dict(id='ab_zh_code',language='zh',family='code',thinking=False,prompt='写Python函数寻找列表中第一个只出现一次的整数，找不到则返回None。包含中文文档字符串和三个断言，只输出代码。',output_cap=384),
 dict(id='ab_en_math',language='en',family='math',thinking=True,prompt='A volunteer works 3 hours on Monday and 5 on Tuesday. A second volunteer works twice as many total hours. How many hours did they work together? Show a short check.',output_cap=384),
 dict(id='ab_zh_math',language='zh',family='math',thinking=True,prompt='甲周三工作4小时、周四工作7小时，乙两天合计工作时间是甲的两倍。两人总共工作多少小时？简短验算。',output_cap=384),
 dict(id='ab_mix_plan',language='mixed',family='planning',thinking=False,prompt='Plan a 90-minute neighborhood seed exchange with six volunteers and a $120 budget. 用中文写安排，保留budget、schedule、roles字段名，最后用一句英文概括。',output_cap=384),
 dict(id='ab_mix_debug',language='mixed',family='debug',thinking=False,prompt='Explain the Python error caused by removing list items while iterating over that same list. 用中文解释，给出英文变量名的修正代码和两个断言。',output_cap=384),
]
