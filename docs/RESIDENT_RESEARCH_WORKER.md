# 常驻研究 Worker

实验默认复用同一进程中的 Target、DFlash、Q4 输出头和 GPU 缓冲。更换 prompt、成本分析、预算策略或独立实验模块，不应重新加载权重。实验完成后保持 READY，**没有空闲退出计时器**。只有用户要求停止、切换不兼容的权重/后端、CUDA context 故障或必要的内存维护等情况才重启。

这是串行研究执行器，不是 DeepInfer 完整异步引擎，也不提供多请求 GPU batch 或公网推理服务。

## 生命周期

`resident_worker.py` 使用已经准备好、具有固定依赖/扩展/本地权重链接的目录。通过原 `e3_run.sh` 启动一次：

```bash
E3_ROOT=/path/to/prepared/root bash /path/to/prepared/root/e3_run.sh \
  resident_worker.py --root /path/to/prepared/root --checkpoint /path/to/target/checkpoint
```

`e12_session.py` 构造一个 session，包含 `weights`、`target`、`draft`、`projection`、`tokenizer`、`eos`、`calibration` 和 `torch`。研究模块实现 `run(session, arguments)`，复用这些对象；每个请求自行创建独立 state，不把上一请求的 KV/GDN/PLE 状态当作新请求的初始状态。

`worker-status.json` 记录 LOADING / READY / RUNNING / STOPPED、PID、加载次数及完成任务数。每个研究任务只允许在 worker 的准备目录中指定一个本地模块，并提供其 SHA256。worker 编译实际校验过的字节，避免源文件修改后误用旧 `.pyc`。

把如下 JSON 原子地放入 `jobs/pending/<unique-id>.json`：

```json
{"module":"resident_probe","sha256":"<exact source SHA256>","arguments":{"prompt":"Write two sentences explaining how a cache works."}}
```

任务依次移入 `jobs/running`，结果写入 `jobs/done`，随后回到 READY，继续保留模型。没有网络任务 API。显式创建 `jobs/STOP` 会在当前任务结束后退出。实验不得并行修改全局算子 dispatch；临时替换须在 `finally` 恢复。需要重启上下文的 GPU 故障不能靠继续提交任务修复。

## 验证

`resident_probe.py` 在已有权重上执行 prompt、DFlash 草稿、width4 Target 验证和单 token commit，与 serial logits/live state 对照。连续运行两个不同 prompt 时，须同时确认：结果通过、相同 PID、`load_count=1`、相同 Target/DFlash 对象，以及最终 READY。对象 ID 仅作同进程身份诊断，不作为模型内容哈希或性能证据。

E12 最初的一次性采集脚本已经运行后才收到常驻要求，无法安全地为正在运行的 Python 函数追加任务循环。因此先完成并保留该轮数据，再进行一次切换所必需的重载；后续研究使用常驻 worker。不能为了“保持常驻”把一个无法接任务的进程暂停后当作可复用服务。

2026-10-21 实测：[同进程连续两个 GPU probe 的复用证据](../results/e12/2026-10-21-resident-reuse.json)通过，加载计数保持1，两个 prompt 的 serial logits/live state 检查通过。worker 在本轮结束后保持 READY，模型常驻。
