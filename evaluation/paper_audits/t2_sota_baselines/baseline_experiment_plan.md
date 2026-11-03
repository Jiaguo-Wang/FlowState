# T2：最小公平对照方案（不执行）

## 1. 先明确可比较对象

|对象与范围|输入真实可得|共同候选/严格边/K|关键原机制保留程度|门禁|
|---|---|---|---|---|
|内部7策略，Strict20|是，完整冻结输入|是|全部既有内部定义；Exact有证书|**READY**，已有完整结果，无需重复运行|
|KVFlow-inspired，已冻结Synthetic A/B/C CPU|预先声明的合成ASG/Fixed Prefix，身份可复核|共同CPU候选/K；A/B物理尚PROPOSED，不能称Strict GPU|STE传播保留；无原KV叶子驱逐、预取、调度|**READY，仅策略CPU范围**|
|同一KVFlow-inspired与C/K3真实状态接线|声明Token与候选来源可追溯，原生状态已有参考|16项全记账的控制合同存在；该策略新选集未逐臂恢复验证|Recurrent保留适配|**PARTIAL，GPU执行前还需独立正确性门禁**|
|KVFlow原系统，Strict20|ASG/分段输入缺失|不能忠实映射|不可通过造图弥补|**BLOCKED**|
|完整KVFlow，Synthetic|逻辑图具备；论文时点代码、实际分段/异步路径不完整|原KV预算/拓扑与Recurrent-K不等价|Prefetch/Scheduling是主要组成|**PARTIAL**，不列为立即可执行性能方案|
|现有Marconi代理，原合成CPU|声明访问与L代理具备|相同CPU候选/K|只保留复合效用外形|**READY，限明确命名的代理复算；不能充当原Marconi**|
|忠实Marconi保留策略，新Strict|当前Token/状态有；真实访问、GDN FLOP成本及调参缺口|固定候选可比，但准入/混合字节/动态树不同|尚不能保留完整原策略|**PARTIAL**；缺字段实例本身BLOCKED|
|TOPAS|关键任务服务/动态容量输入缺失，代码UNKNOWN|固定Pending和K不能表达其联合动作|抽取单项会改变方法|**BLOCKED（当前分配基线用途）**|

READY 表示所写范围可开展公平比较，不表示该范围等同原系统，也不授权执行。严格Epoch中的 UNKNOWN 不能作为零收益悄悄剔除。GPU物理身份必须新生命周期重新绑定，历史Node/Slot不可复用。

## 2. 唯一优先补充项：KVFlow-inspired 保留策略的共同输入证据包

**限定为 CPU，复用已有Synthetic A/B/C、K=3/5六条件，不新增工作负载或策略。**该项 READY 指其完整逻辑输入和既有实现已具备；已有30行基础结果，补充工作的价值是将其整理为可审查的策略级对照，而非重新追求胜率。

- 比较：现有 KVFlow-inspired、固定 Marginal Greedy、WFC、StandaloneTopK；LRU/Marconi代理仅作附表，不作为完成SOTA复现的依据。不得因新 `plan_strict()` 自动选择Exact而把既有Greedy实验名称混用。
- 输入：`kvflow_multiworkflow_cpu_main/workloads.json` 中 `graph/declarations/candidates/calls`，`protocol.json`、Token/config身份；采用原 `policy.py` 的 ID 同分规则。当前/未来调用严格分开。
- 预算：原K=3/5，所有末端和L−64祖先都收费；无额外免费保留。A20、B12、C16个候选，不删支配项。
- 准入：CPU原数据标PROPOSED；如附录连接C/K3冻结物理证据，必须逐Token/配置/位置验证对应关系并单独标出认证等级，不能把A/B一起升级。
- 输出：每项STE/priority/同分组、选集、当前每请求E/G与共同J、未来静态J、ID差异与前沿差异、预算用量；保持既有负面结果。特别展示KVFlow在全部Ready附近同分时ID规则造成的重复祖先选择。
- 评价：共同Phi只做评分、不进入KVFlow决策；明确短T外推。任何工作流JCT、预取收益、完整KVFlow优劣均不在结论范围。
- 预期成本：低，读取已有结果和声明、补充追溯即可；不需模型、GPU或移植。若完整输入摘要/域对不上，停止而非改Token或规则。

**为什么只推荐这一项：**这是外部方法中已有完整声明输入、现存无Phi策略实现、且不必改核心代码的路线。但它只能补强“workflow-distance retention”的对照，不能单独解决“缺少忠实Hybrid SOTA”的核心缺口。内部Strict20已充分比较，再跑相同20条件不会增加外部证据。

## 3. 暂不安排的项目及升级条件

Marconi 最有论文相关性，但当前 PARTIAL，不把它列为可以立即跑的新实验。要升级，至少需确定：可追溯的真实访问时刻；Qwen GDN成本/状态字节依据；真实树父子和原驱逐资格；仅用过去轨迹的α协议；冻结FA/固定候选时省略准入和KV管理的明确适配合同。不能通过统一常数或位置代理满足这些条件。即使补齐，仍应称“Marconi retention adaptation”，除非完整原机制和服务系统对比也落实。

KVFlow C/K3 GPU只有在其自身选集通过Strict绑定、全候选预算、Native/COW/Cold后才可升级；其他策略成功不自动认证KVFlow选集。本轮不提供未经这些前置门禁的性能执行命令，也不把同集结果当作控制开销相同。

TOPAS应先保留为相关工作与问题定位比较；缺源码、任务服务画像和动态请求预算时，不计划“TOPAS分数”替代品。

## 4. 若后续获准做系统级比较的共同原则

策略级：共同候选、相同Strict边、相同Pending、K、Phi及确定性同分；全量报告J/E/G与求解耗时。系统级：必须让原方法的准入、预取、调度和真实字节预算有发挥空间，统一模型、硬件、请求流和全部控制开销，并测任务JCT/请求延迟/吞吐。后者不能由当前固定保留快照的胜率代替。

本文件是建议，不执行CPU实验、移植或GPU验证；不自动进入下一阶段。
