# K3：动态预算运行时接入 Target CUDA Graph

2026-10-23，基于 K2 动态模式（不恢复固定草稿），Graph 为显式 `--execution graph` 候选，默认仍为 eager。复用 E29 的 target 算术与五层 DFlash 特征提取，不集成 E33 筛选头、不换 W8 草稿/NumPy预测器/成本曲线，不承诺减少所有 host 同步。

## 执行边界

- 单请求独占固定地址 State，窗口1～8行×GDN两种奇偶共16个 bucket；启动预热后才READY，不允许普通请求中临时capture。请求取消/完成释放lease，下一请求在同一地址reset；workspace由显式owner关闭。
- prompt继续eager，Graph只替换decode compute；stage、commit、greedy/EOS与E17预测/预算保持原样。Graph输出仍为全词表logits，保留一个完成fence；不把E33的候选logits当完整logits。
- 未预热bucket、奇偶不一致或State reallocation时明确eager fallback并记数；越界与外来state拒绝。公开实际capture/replay/fallback、准备耗时，不声称异步或多请求联合GPU batch。

## 验收顺序

1. 本地lease/reset/错误路径契约与K1/K2回归。
2. 常驻模型独立封存：1～8行、两奇偶、接受前缀1/中间/全部，比较eager的全部logits、全部DFlash特征及有效state；检查未预热width fallback。不能把不可用GPU的mock当作实测。
3. 三个提示下E17 adaptive eager/graph的输出、逐轮验证宽度、committed logits/features、最终有效state一致；首/中/末拒绝后继续、EOS、length1、waiting/prepared prompt/prepared decode取消与同workspace代次复用，无请求中capture。
4. 相同resident权重、同预算/成本、交替顺序的完整请求/TTFT/decode计时；Graph预热/辅助内存单列，不能与旧不同任务TPS横比。真实SSE/取消走Graph；收尾group/env/lease回收审计。

每轮源码与外部kernel身份封存；失败保留，修复后新版本重跑，不覆盖现有实验源、不重载或停止常驻模型。只在同预算轨迹下讨论Graph收益；推广默认需要更广负载，当前仅完成运行时接通。
