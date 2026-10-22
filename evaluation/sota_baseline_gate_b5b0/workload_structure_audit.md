# 工作负载结构与判别力审计

仅g001–g004四个受控已知输入回放快照；16个相关预算条件。全部数值来自冻结B4-C results.json，不重新计算Phi、不调用策略。容差沿用evaluate.py的EPS=1e-9 modeled ms。K_min独立穷举仅检查前沿集合覆盖，不运行新的Exact收益求解。

## 结构总表

|快照|候选|Pending|准入/正收益边|跨Workflow正收益边|共享候选数|有重叠Pending数|K_min|
|---|---:|---:|---:|---:|---:|---:|---:|
|g001|12|4|15/15|3|1|4|4|
|g002|16|4|19/19|3|1|4|4|
|g003|20|4|23/23|3|1|4|4|
|g004|8|4|11/11|3|1|4|4|

合计56候选、16 Pending、224配对：68准入且正收益、156拒绝、0未知；12条跨Workflow正收益边。4/4快照均有共享与非平凡重叠，12/16条件（每组K1–3）同时具有共享、重叠和紧逻辑预算。K4均饱和，但不保证Greedy找到完整前沿。这里没有GPU容量压力证明。

## 逐候选支持与替代关系

### g001

主层A，附加标签A/B/C；不是三个独立分层样本。唯一多请求候选为B_TURN_001，位置2432，支持四个Pending；其余候选均仅支持一个Pending。

|候选（省略组前缀）|正收益Pending数|支持Pending|
|---|---:|---|
|A_TURN_001|1|A_PENDING_TURN_004|
|A_TURN_002|1|A_PENDING_TURN_004|
|A_TURN_003|1|A_PENDING_TURN_004|
|B_TURN_001|4|A_PENDING_TURN_004, B_PENDING_TURN_004, C_PENDING_TURN_004, D_PENDING_TURN_004|
|B_TURN_002|1|B_PENDING_TURN_004|
|B_TURN_003|1|B_PENDING_TURN_004|
|C_TURN_001|1|C_PENDING_TURN_004|
|C_TURN_002|1|C_PENDING_TURN_004|
|C_TURN_003|1|C_PENDING_TURN_004|
|D_TURN_001|1|D_PENDING_TURN_004|
|D_TURN_002|1|D_PENDING_TURN_004|
|D_TURN_003|1|D_PENDING_TURN_004|

|Pending|正收益候选数|全候选最大前沿|唯一最大前沿候选|
|---|---:|---:|---|
|A_PENDING_TURN_004|4|4224|A_TURN_003|
|B_PENDING_TURN_004|3|5312|B_TURN_003|
|C_PENDING_TURN_004|4|4736|C_TURN_003|
|D_PENDING_TURN_004|4|4224|D_TURN_003|

冻结收益向量逐维比较确认g001有10对严格支配关系，完整方向及ID见structure_statistics.json/domination_pairs。每条专属链的最深候选支配该链较浅项，但不逐维支配覆盖其他请求的共享B1。不能把共享和同请求替代混为一谈。

### g002

主层A，附加标签A/B/C；不是三个独立分层样本。唯一多请求候选为B_TURN_001，位置2432，支持四个Pending；其余候选均仅支持一个Pending。

|候选（省略组前缀）|正收益Pending数|支持Pending|
|---|---:|---|
|A_TURN_001|1|A_PENDING_TURN_005|
|A_TURN_002|1|A_PENDING_TURN_005|
|A_TURN_003|1|A_PENDING_TURN_005|
|A_TURN_004|1|A_PENDING_TURN_005|
|B_TURN_001|4|A_PENDING_TURN_005, B_PENDING_TURN_005, C_PENDING_TURN_005, D_PENDING_TURN_005|
|B_TURN_002|1|B_PENDING_TURN_005|
|B_TURN_003|1|B_PENDING_TURN_005|
|B_TURN_004|1|B_PENDING_TURN_005|
|C_TURN_001|1|C_PENDING_TURN_005|
|C_TURN_002|1|C_PENDING_TURN_005|
|C_TURN_003|1|C_PENDING_TURN_005|
|C_TURN_004|1|C_PENDING_TURN_005|
|D_TURN_001|1|D_PENDING_TURN_005|
|D_TURN_002|1|D_PENDING_TURN_005|
|D_TURN_003|1|D_PENDING_TURN_005|
|D_TURN_004|1|D_PENDING_TURN_005|

|Pending|正收益候选数|全候选最大前沿|唯一最大前沿候选|
|---|---:|---:|---|
|A_PENDING_TURN_005|5|6080|A_TURN_004|
|B_PENDING_TURN_005|4|4608|B_TURN_004|
|C_PENDING_TURN_005|5|7168|C_TURN_004|
|D_PENDING_TURN_005|5|6080|D_TURN_004|

冻结收益向量逐维比较确认g002有21对严格支配关系，完整方向及ID见structure_statistics.json/domination_pairs。每条专属链的最深候选支配该链较浅项，但不逐维支配覆盖其他请求的共享B1。不能把共享和同请求替代混为一谈。

### g003

主层A，附加标签A/B/C；不是三个独立分层样本。唯一多请求候选为B_TURN_001，位置2432，支持四个Pending；其余候选均仅支持一个Pending。

|候选（省略组前缀）|正收益Pending数|支持Pending|
|---|---:|---|
|A_TURN_001|1|A_PENDING_TURN_006|
|A_TURN_002|1|A_PENDING_TURN_006|
|A_TURN_003|1|A_PENDING_TURN_006|
|A_TURN_004|1|A_PENDING_TURN_006|
|A_TURN_005|1|A_PENDING_TURN_006|
|B_TURN_001|4|A_PENDING_TURN_006, B_PENDING_TURN_006, C_PENDING_TURN_006, D_PENDING_TURN_006|
|B_TURN_002|1|B_PENDING_TURN_006|
|B_TURN_003|1|B_PENDING_TURN_006|
|B_TURN_004|1|B_PENDING_TURN_006|
|B_TURN_005|1|B_PENDING_TURN_006|
|C_TURN_001|1|C_PENDING_TURN_006|
|C_TURN_002|1|C_PENDING_TURN_006|
|C_TURN_003|1|C_PENDING_TURN_006|
|C_TURN_004|1|C_PENDING_TURN_006|
|C_TURN_005|1|C_PENDING_TURN_006|
|D_TURN_001|1|D_PENDING_TURN_006|
|D_TURN_002|1|D_PENDING_TURN_006|
|D_TURN_003|1|D_PENDING_TURN_006|
|D_TURN_004|1|D_PENDING_TURN_006|
|D_TURN_005|1|D_PENDING_TURN_006|

|Pending|正收益候选数|全候选最大前沿|唯一最大前沿候选|
|---|---:|---:|---|
|A_PENDING_TURN_006|6|10816|A_TURN_005|
|B_PENDING_TURN_006|5|10496|B_TURN_005|
|C_PENDING_TURN_006|6|9280|C_TURN_005|
|D_PENDING_TURN_006|6|5120|D_TURN_005|

冻结收益向量逐维比较确认g003有36对严格支配关系，完整方向及ID见structure_statistics.json/domination_pairs。每条专属链的最深候选支配该链较浅项，但不逐维支配覆盖其他请求的共享B1。不能把共享和同请求替代混为一谈。

### g004

主层A，附加标签A/B/C；不是三个独立分层样本。唯一多请求候选为B_TURN_001，位置2432，支持四个Pending；其余候选均仅支持一个Pending。

|候选（省略组前缀）|正收益Pending数|支持Pending|
|---|---:|---|
|A_TURN_001|1|A_PENDING_TURN_003|
|A_TURN_002|1|A_PENDING_TURN_003|
|B_TURN_001|4|A_PENDING_TURN_003, B_PENDING_TURN_003, C_PENDING_TURN_003, D_PENDING_TURN_003|
|B_TURN_002|1|B_PENDING_TURN_003|
|C_TURN_001|1|C_PENDING_TURN_003|
|C_TURN_002|1|C_PENDING_TURN_003|
|D_TURN_001|1|D_PENDING_TURN_003|
|D_TURN_002|1|D_PENDING_TURN_003|

|Pending|正收益候选数|全候选最大前沿|唯一最大前沿候选|
|---|---:|---:|---|
|A_PENDING_TURN_003|3|3904|A_TURN_002|
|B_PENDING_TURN_003|2|3712|B_TURN_002|
|C_PENDING_TURN_003|3|5056|C_TURN_002|
|D_PENDING_TURN_003|3|3712|D_TURN_002|

冻结收益向量逐维比较确认g004有3对严格支配关系，完整方向及ID见structure_statistics.json/domination_pairs。每条专属链的最深候选支配该链较浅项，但不逐维支配覆盖其他请求的共享B1。不能把共享和同请求替代混为一谈。

## 全部预算的共同目标复核

单位：modeled ms。完整最终ID、Exact差距见condition_policy_review.csv（80行）；原始7策略全部结果仍位于B4-C。不同选择顺序不等于不同集合。

|条件|FlowState|PCB|WFC|LDR|Exact|达到Exact的PCB/WFC|
|---|---:|---:|---:|---:|---:|---|
|g001/K1|334.563447662|497.554963838|334.563447662|497.554963838|334.563447662|WFC|
|g001/K2|224.359648669|318.540423566|224.359648669|318.540423566|224.359648669|WFC|
|g001/K3|136.436789948|159.270211783|136.436789948|159.270211783|136.436789948|WFC|
|g001/K4|68.218394974|0.000000000|68.218394974|0.000000000|0.000000000|PCB|
|g002/K1|545.999749991|636.659268355|545.999749991|636.659268355|545.999749991|WFC|
|g002/K2|363.184899992|405.364415466|363.184899992|405.364415466|363.184899992|WFC|
|g002/K3|223.086373602|174.069562577|223.086373602|174.069562577|174.069562577|PCB|
|g002/K4|82.987847211|0.000000000|82.987847211|0.000000000|0.000000000|PCB|
|g003/K1|959.964932377|959.964932377|1015.024675511|959.964932377|959.964932377|PCB|
|g003/K2|552.288450383|552.288450383|685.857711236|552.288450383|552.288450383|PCB|
|g003/K3|193.885055650|193.885055650|369.721389140|193.885055650|193.885055650|PCB|
|g003/K4|0.000000000|0.000000000|102.763475543|0.000000000|0.000000000|PCB|
|g004/K1|253.455257084|426.219729646|253.455257084|426.219729646|253.455257084|WFC|
|g004/K2|153.168924167|279.241538815|153.168924167|279.241538815|153.168924167|WFC|
|g004/K3|97.217634680|139.620769407|97.217634680|139.620769407|97.217634680|WFC|
|g004/K4|48.608817340|0.000000000|48.608817340|0.000000000|0.000000000|PCB|

FlowState 12/16最优；PCB 8/16；WFC 8/16；两者最优条件互补，联合覆盖16/16。这个事后事实不是可部署的Best-of策略，不知道最优解时不能按结果选择基线。所有7个冻结策略之间，同目标不同最终集合的配对数为0（忽略顺序）；Exact代表集合与策略顺序不同不构成胜负。

全部Greedy非最优条件：
- g001/K4：落后Exact 68.218394974 modeled ms；集合['RQ3_G001_B_TURN_001', 'RQ3_G001_B_TURN_003', 'RQ3_G001_C_TURN_003', 'RQ3_G001_A_TURN_003']。
- g002/K3：落后Exact 49.016811025 modeled ms；集合['RQ3_G002_B_TURN_001', 'RQ3_G002_C_TURN_004', 'RQ3_G002_A_TURN_004']。
- g002/K4：落后Exact 82.987847211 modeled ms；集合['RQ3_G002_B_TURN_001', 'RQ3_G002_C_TURN_004', 'RQ3_G002_A_TURN_004', 'RQ3_G002_D_TURN_004']。
- g004/K4：落后Exact 48.608817340 modeled ms；集合['RQ3_G004_B_TURN_001', 'RQ3_G004_C_TURN_002', 'RQ3_G004_A_TURN_002', 'RQ3_G004_B_TURN_002']。

## 为什么简单策略能达到Exact

**冻结结构已证实，以下化简为条件性数学推论。** 每组只有一个覆盖四请求的共享B1，其他候选按请求可分，最深专属项在该请求收益上支配浅层项。任意可行解中，同一专属链的多个项可由最深项替代而不降低F。因此最优解只需考虑“无B1的至多K个最深专属项”或“有B1及至多K−1个最深专属项”两类。固定是否含B1后，剩余收益按请求可分，选最大单请求剩余收益即可。这里是解释结构，不实现或运行新求解器。

四个请求各自最大前沿均由不同唯一深层项达到，所以任何达到全候选前沿的集合都必须含这四项：K_min≥4；四项集合构成见证给出K_min≤4。本批全候选前沿恰为T，J(C)=0。这并非任意工作负载都零残差。

PCB按每请求最佳项推荐并排序，故能够体现无共享项的解族；WFC在本批总会优先覆盖四请求的B1，再处理专属项。WFC的Token推进排序不一般等价于Phi边际排序：本批后三步在g001/002/004与Greedy一致是冻结观察，不是普适定理。两解族覆盖所有可行最优形态，并不证明PCB/WFC各自一定得到该族最优。

g001/K1–3、g002/K1–2、g004/K1–3由共享优先族获胜，WFC达到Exact；g003/K1–4、g002/K3–4、g001/K4、g004/K4由PCB达到Exact。g002/K3是紧预算下Greedy也会失败的直接例子：早选B1且无交换，最终不如A4/C4/D4，差49.016811025。K4时三组Greedy保留B1挤占第四个深层项。

g003/K1：WFC选择B1覆盖4请求、推进9728 Token；FlowState选择A5只覆盖1请求但推进10816 Token，冻结收益分别365.672296502与420.732039636。故覆盖数量优先本身即可解释分歧，不是位置项独立消融证据。该组PCB与Greedy一致并达到Exact。

## 判别力与主张边界

FlowState对PCB：8胜4平4负；对WFC：4胜12平0负；对StandaloneTopK：5胜11平。所有严格差条件保留在condition_review.json；FlowState没有任何条件同时严格优于PCB和WFC。四个组有可解释的局部判别力，不能说完全无差别；但不支持“全局集合分配普遍优于强简单策略”或独特必要性。

当前结构缺少多个有不同非单例支持域、相互嵌套的共享候选，以及在紧预算下这些共享层与专属深层真正竞争的多层实例；也缺少能隔离Phi位置项与简单Token排序差异的预注册条件。这是尚未覆盖的检验维度，不保证FlowState获胜。合法Token树通常是嵌套/不交覆盖，不应为复杂性构造不合法交叉矩阵。

没有新增性能实验、策略组合或结构搜索；没有将4快照的16预算视为16独立工作流。已有目录中的audit_workload_structure.py及workload_structure_metrics.json保持原样，本轮直接消费冻结收益并独立交叉核对主要计数。
