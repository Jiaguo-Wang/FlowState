# Phase B5-C1 最终报告

## 最终门禁：PARTIAL

发现有真实来源的多共享**潜在内容结构**，但尚没有新的多共享Recurrent Snapshot满足Token身份、同边界物化及需求证据。因此不能给READY，也不把现有缺证据写成共享不存在。

## 主要发现

- 本轮直接读取全部5000条OpenHands原始归档会话。完整消息前缀有1217个跨会话共享节点：1个公共System节点、1216个System＋完整用户消息节点；后者全部具有相同instance_id的重复执行来源。没有发现共享到第三条完整消息的跨会话前缀；不排除消息内部Token部分共享。
- 有1216个同任务多Session组，全部记录而非筛选胜例。两个任务各两个Session可提出Trace-derived Controlled Composition，具有两个潜在互不相交共享子树；不是原生共同祖先Fork。公共System祖先与任务子集的嵌套只是内容层，不证明Runtime恰有两个对应共享Checkpoint。
- AgentX来源文件SHA与P2一致；hash和原生subagent容器仍不能补齐Token输入，旧Formal关系不转为Strict。
- FreeInference既有464行逐会话指标中，165个至少有两个block前缀分叉，25个还含subagent。已完整列出这些线索，但没有消息/Token，不能认证同边界多共享。
- LMCache已有16对Token输入和单共享恢复证据，不证明新的多共享状态；TraceElephant五个固定样本有多agent和消息，父子输入继承尚未核验；TraceLab仅有部分事件schema；GATC没有本地可用样本。

## 对核心问题的回答

1. 是否存在自然多层共享？AgentX/FreeInference有block/容器线索；本轮没有完整的原生Token＋Pending＋Checkpoint证据链。
2. 是否存在可追溯受控组合？存在OpenHands同任务重复执行的完整消息依据，可制定有限前置核验计划。不能仅凭相同任务认定兼容；完整消息检查与后续目标Token检查必须分开。
3. 是否出现多个非单例正收益支持？现有B4四组仍只有一个；新来源只有内容支持集合，没有新T/L/物化合同，所以正收益计数UNKNOWN。
4. 是否有真实容量竞争？UNKNOWN。多个需求、逻辑K不足与物理池压力是不同命题，不能通过事后压低K认证。
5. 是否能避开特殊精确可解结构？不能保证。严格Token树天然嵌套/不交，可能适用T1精确DP；两个独立单共享子树也可能继续分解。应保留这些案例，而不是追求不可能的交叉前缀。

## 唯一下一步建议

先执行一次独立、有限的 **OpenHands同任务重复轨迹受控组合CPU输入身份核验**：按来源固定顺序核对最早两个任务各两个Session、前两次历史及第三次请求输入，固定目标配置并检查完整Token前缀及消息因果截断，失败不补位。暂不投入GPU、不构造Checkpoint或比较策略。通过后才能另行决定是否进行严格Runtime采集。

## 交付和边界

source_inventory.md、sharing_structure_audit.md、strict_reconstruction_feasibility.md、candidate_workload_plan.md分别保存来源、结构、严格重建缺口和条件性方案。所有匹配的消息前缀、来源Session、层次关系及block结构线索均保存JSON账本；audit.py是只读CPU复核入口，不导入优化器或Runtime。source_manifest.json记录读取文件摘要。

未修改旧冻结产物、Greedy、Phi或Baseline；未启动GPU、Engine或Replay；未生成新Strict矩阵；没有新增策略比较或性能结论。完成后停止，等待审查。
