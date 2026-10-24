# 双请求联合 Target：隔离参考、提交事务与原生调度

本分支承接K4运行时开销提交 `1ac99b1`，使用另一工作区既有双slot初版的快照作为计算原型，保留原工作区及 `191-k4batch-probe-v1.json` 失败，不修改它们。原初版probe先计算两个参考、再提交，参考共用同宽度buffer时首请求的conv/PLE提交可能读取第二个请求的临时结果；需用独立solo执行器并立即提交，不能据此跳过状态检查。

交付：`JointTarget` 管理两份独立状态、一次联合48层前向、E33输出筛选、按请求offset/真实接受前缀提交；完整预检keeps后才写任何状态，commit fence完成才允许retire/复用。`JointSchedulerBridge` 驱动已有DeepInfer原生Scheduler，把同一decode plan的两行一起交给GPU；初期target-only greedy、同步接口，prefill沿原8行路径。异步HTTP和动态DFlash多请求由既有K4主线继续，不在本分支重复实现。

验证首先复查首个buffer别名问题；覆盖相同/不同宽度、padding、parity、零保留/不同拒绝前缀、顺序变化、尾部容量、取消后slot复用和stale ticket。每个请求使用独立reference buffers，逐行完整logits/特征、有效state及下一步输出对照。再验证原生Scheduler两请求生成、EOS/cap、waiting/prepared取消与资源归还。以同输入同token做solo sequential与真正joint的完整decode计时，各4次交替；吞吐按总token/总墙钟，另报逐请求完成与间隔，不把此微服务边界视为公网HTTP性能。

Graph按两slot宽度1/2/4/8及各自GDN parity捕获80张图，计时中不得capture；无Graph路径只用于正确性排错。固定常驻target、E33头和原算术，不重新加载或下载。所有执行源/输入封存、共享resident队列串行作业、最终READY/同对象/内存检查。正确性、真实joint调用和速度分别验收；不以调用计数代替真实GPU工作，也不把本阶段target-only吞吐作为动态DFlash或单请求加速。
