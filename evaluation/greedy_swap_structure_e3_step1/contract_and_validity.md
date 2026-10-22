# 合同、复现与有效性

本轮新代码只在本目录。生产Optimizer、成本模型、Strict Compatibility及历史数据保持只读。`protocol.json`在求解之前创建，`protocol_lock.json`锁定协议和主程序，未按结果改为最佳改善或调整ID遍历。

1-Swap使用原Greedy最终集作为起点，移出ID、移入ID均按字典序，每次遇到第一个下降超过1e-9 modeled ms的交换立即接受，然后重新扫描。保持基数，不做单独新增或删除，不接受等值平台移动。有限集合域和严格下降保证有限终止；终点只保证当前基数的一换一局部最优，不保证全局最优。空集或已经包含全部候选时没有交换邻居。此实现不是新的生产策略。

20个条件全部验证：原六种策略的选集/目标与冻结E2及B5-C4一致；每个接受步骤确为字典序首个严格改善；完整终止邻域无严格改善；每个输出≤K且只含原候选。对两个残余局部陷阱枚举已有7候选的同基数集合，诊断到更优集合的最小替换距离；另作严格下降图的可达性检查。后者不作为新策略、不改变成绩、不计入求解计时。

两个残余失败都能从原Greedy沿另一条两步改善路径到Exact。因此不能声称所有局部搜索均失败，也不能事后用该路径替换正式首改善结果。需要区分全局联合组合价值与特定不可撤销/路径选择规则的缺陷。

旧573条仅复用并复核正收益支持、重叠、唯一最大前沿实现者、预算及共同Phi；不在149候选域启动Exact或新策略。证明仅对旧Formal模型成立，不提升到Strict Runtime认证。每条旧条件及原始源行号保留在`legacy_573_structure.csv`。

计时原始记录4960条：20条件×31重复×（7种方法+单独Swap-only）。每个条件3次预热不进入统计；计时为CPU求解函数的perf_counter_ns边界，不包含文件读取、严格输入认证、收益准备、Plan对象构造/验证或Runtime操作。原策略函数内部诊断结构构造仍计入时间。不存在GPU计时、线上P95、Deadline保证或控制净收益结论。环境保留在results.json。

复现入口（均CPU-only）：

```bash
/home/wjg/uv312/bin/python -B -m evaluation.greedy_swap_structure_e3_step1.audit
/home/wjg/uv312/bin/python -B -m evaluation.greedy_swap_structure_e3_step1.legacy
/home/wjg/uv312/bin/python -B -m evaluation.greedy_swap_structure_e3_step1.diagnostics
/home/wjg/uv312/bin/python -B -m evaluation.greedy_swap_structure_e3_step1.report
```

产物以独占创建保护，不应对现有归档直接重跑覆盖。需要复现时先在独立位置保留本目录代码与协议，再执行；不删除或覆盖本轮结果。报告中的5个Snapshot包含4个E2来源组合及一个历史发现案例，多个预算和重复计时不增加独立工作负载样本量。
