# E32：量化投影有效行的紧凑共享内存 staging

冻结 E28 W8 草稿、E17 NumPy 线性预算、E29 W8 费用与完整 target Graph；不重载模型、不修改旧封存源或全局 dispatch。E10/E11 几何与归并扫描、E30 头布局、E31 scale 转换均不重复。

源码核对发现 MXFP8 qmmf 对 1～8 行 decode 始终采用 BM16，每 stage 为输入留 16×128=2048 bytes，并对无效行执行 cp.async 零填充。候选固定原 BM16/BN64/4 warp/4 stage、split-K 与 cluster 顺序，仅按真实行数存输入，ldmatrix 中无效 MMA 行复用第0行地址，写回仍仅限有效行。有效 MMA 行及所有权重/scale/算术顺序保持不变。共享内存从 25,088 bytes/CTA 降至 17,408～20,992 bytes；输入 staging 的无效行清零消失。**不会减少全局权重字节、有效输入字节或 MMA 次数**，是否改善并发/耗时待测。

同一次构建包含 CROWS16 原逻辑控制和 CROWS1～8 单一紧凑候选，使用 worker 的 release venv。先在现有常驻 worker 做 all-width 数值筛查，覆盖真实四类矩阵、随机和带 stride 输入、尾部列、SK1/2/4/8与独立 stream；控制/候选都必须与原 resident 扩展逐位一致，边界哨兵不变。

随后复用 E30 四条校准 root 的前两轮，通过实例局部前向取得第0、23、47层 MoE共享专家 gate/up 和 down 的真实输入，以及 GDN/attention in/out 的真实输入。捕获每份输入对应的单投影 CUDA Graph，交替测原扩展与候选：热复用每区间10次，128MiB缓存扰动每区间1次，各6次配对。clone/check/capture/扰动不计入event区间。采样前向 logits、DFlash特征和提交live state对照冻结Graph，避免采集器改轨迹。

事前门槛：全部精确检查通过；按四个root加总真实投影耗时，缓存扰动下至少节省10%，root-bootstrap速度收益95%下界为正，没有root超过2%退化。未通过即结束，不扩大warp/stage/split-K扫描。若通过，另行封存完整Graph正常decode A/B，冻结相同草稿/预算/费用，验证完整输出/提交logits/特征/live state后才报告吞吐。小样本micro不是整模性能或新任务泛化证明。
