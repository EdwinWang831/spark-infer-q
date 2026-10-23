# E33：组件门槛通过后，完整生成配对对照

只有 `PASS_E33_HEAD_GATE` 才提交。保留首轮未过门槛的记录和融合后的组件数据，不把头部节时36.24%外推为decode收益。

使用E28八条旧工程root，均不是E33四条组件校准root；仍是项目已用回归集，不是新的真实流量。原manifest、自然EOS和各自output_cap保持不变。两组都用原target权重、E28 W8草稿（Q4草稿头）、E17 NumPy线性预算和冻结E29 W8费用，不在线重校准。仅greedy。

基线为原完整target Graph，包含原BF16头。候选是实例私有Graph：原48层、五处DFlash特征、最终writeback/mixer保留，Graph输出最后hidden；随后运行同一已验证的融合筛选头，动态压缩、必要同步、原BF16子集/全头回退、argmax全部计入实际decode时间。无全局dispatch变更，没有候选时仍用原完整头。两边各预热16张backbone图；筛选按1～8行预热，所有当前任务实际子集shape在正确性预热中编译，计时内不得新增捕获/JIT shape。

每条先从相同prefill初态建立完整serial oracle，保留committed logits、DFlash特征、tokens与live state。两个模式分别跑完整正确性/预热：候选诊断另外执行原完整头，对照所有候选logits逐位、每个argmax、所有提交logits/特征及最终live state。正常计时不重算完整诊断头。全部token以及每轮预算/接受/提交轨迹必须一致；输出/精度失败立即停止。

每任务每模式4次，AB/BA交替，共64次正式decode计时。包含草稿、F1、线性预测、预算、target、所有头筛选/精算/回退、commit和循环；排除prefill/首token、初态copy、warmup/capture/JIT、oracle与落盘。每次计时后独立核对tokens和live state。报告实际emitted tokens加权TPS、按root bootstrap95%和逐重复敏感性、fallback/候选分布、原始workload是否相同。只在所有正确性通过、整体速度收益至少1%、root95%下界为正且四次重复均为正时，称为这组受控输入上确认的decode收益；仍不默认接入服务，不作随机采样/长上下文/新流量结论。
