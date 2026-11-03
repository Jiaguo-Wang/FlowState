# T2：真实输入与适配资格

标记：AVAILABLE=已有记录可读；DECLARED=预先声明的合成输入；MISSING=当前归档缺少；UNKNOWN=没有足够证据。可读 Token 不自动认证原始 Agent 调度语义。

## 1. 数据族分开判定

|数据族|Token/严格边/物理状态|访问及 FLOP 输入|ASG 与 Fixed Prefix|可开展的比较|
|---|---|---|---|---|
|旧 OpenHands/AgentX 573 Formal|旧逻辑矩阵；不能升级 Strict；AgentX 存在已知 Formal-only 错边|存在受控代理；并非原论文全部输入|MISSING 决策时 ASG/固定分段声明|旧模型化内部对照及明确命名的代理适配；不是当前 Strict SOTA 证据|
|旧 OpenHands 24 GPU、216 生命周期|同快照输入/选集可追溯；缺当前完整 Strict/逐请求 Cold/COW 认证|LRU 时序与 Marconi 增量 Token 代理已有；原生 recency/真实 FLOPs/调参缺口|MISSING|历史适配比较 QUALIFIED，不能自动成为新 SOTA 控制闭环|
|E2 w1–w4+B5-C4，即 Strict20|5×7候选、4 Pending、完整三态边；既有 UNKNOWN=0|所审计输入中 LRU/Marconi 完整评分与访问证据 MISSING/UNVERIFIED|ASG、Agent 调用身份及固定末端语义 MISSING|Greedy、Exact、1-Swap、WFC、PCB、LDR、Standalone 可用；KVFlow 不可补造|
|B4-B1/B4-C g001–g004|较早严格矩阵及回放事件；不是上述 Strict20|可能包含 creation/access 记录；须按本族合同使用，不能借给 w1–w4|原生 ASG MISSING|复用其已经认证的比较范围；不跨族补齐新矩阵元数据|
|Synthetic Sequential Stage1|CPU 10末端；真实 Runtime 另有10祖先，原10项预算模型不完备|DECLARED 合成访问，非原生 LastAccess；Token 代理|DECLARED 完整10Agent图、固定 Token|CPU 中性对照；原10候选表不能直接作完整物理公平比较|
|Synthetic A/B/C，K=3/5|CPU48个 PROPOSED；6条件、30策略行|DECLARED 合成访问序；L 代理 FLOPs、逻辑字节1|DECLARED graph/declarations，固定/动态 Token及配置|KVFlow-inspired CPU READY；LRU 合成事件对照 READY；Marconi 只代理启发式|
|C/K3 GPU Pilot、正式60臂|16个状态实际物化、64个候选—请求配对；相同 FA/物理控制；正式比较 FS/ST/WFC|完整原生 Marconi 访问历史、GDN FLOP/字节及调参未认证|CPU声明可追溯；未执行 KVFlow GPU策略臂|内部三策略 GPU 已完成；不等于 Marconi/KVFlow 已完成 GPU 对照|

来源：[真实接入审计](../../kvflow_real_data_connection/final_report.md)、[Strict20 输入结果](../../kvflow_inspired_retention_cpu/final_report.md)、[旧24审计](../../legacy_runtime_evidence_p0_audit/final_report.md)、[合成CPU原始表](../../kvflow_multiworkflow_cpu_main/cpu_comparison.csv)、[合成后续纠偏](../../kvflow_multiworkflow_cpu_review/review_report.md)、[C GPU协议](../../c_formal_perf_gpu/protocol.json)。本次直接读取 E3 CSV 140 行及合成 CSV 30 行，没有运行选择器。

## 2. 方法所需字段与不可代替项

|方法|核心输入|已有记录能提供什么|缺口与禁止替代|
|---|---|---|---|
|Marconi 原策略|当前树父子关系、可驱逐资格、被命中节点实际时刻、模型层 FLOPs、SSM/KV字节、历史 bootstrap|Strict Epoch 可提供当前前缀位置、身份及必要物理观测；旧版有访问顺序/模型代理|Rank 不是间隔倒数；L 或 ΔL 不是经验证的 Qwen GDN FLOPs；逻辑1不是完整 Hybrid 字节；单次 Census 不能恢复历史调参输入|
|KVFlow-inspired 当前策略|Invocation、Ready、predecessors、AND/OR、当时可见性、固定/完整 Token、配置摘要|Synthetic `workloads.json` 的 graph/declarations 全部声明|Strict20 没有对应声明；Workflow ID、LCP或整个 Prompt 不能替代|
|KVFlow 原系统额外|逐 Agent 命中分段历史或显式固定末端、Client身份、CPU副本、异步加载状态、调度进度|作者代码说明接口；合成数据满足声明路线的逻辑输入|OpenHands 工具调用并非 Agent ASG；现候选保留实验没有原系统预取/状态调度路径|
|TOPAS|DAG任务/阶段、服务分布、已消耗服务、Ready/Running、到达时间、动态KV预算、搬运/抢占代价|部分图在合成中 DECLARED|服务画像、条件进度、运行请求集、带宽/动态容量合同 MISSING；不能用 FlowState Phi 直接填全部成本|
|内部策略|Candidate/Pending完整域、L/T、Strict边、冻结Phi、K|Strict20 AVAILABLE|旧Formal不能当Strict；任何 UNKNOWN 必须使完整比较失败或按预先声明的不完整范围报告|

StrictAllocationEpoch 是分配证据合同，并不自行记录 ASG、原生 LRU 时钟或 FLOP 模型。[strict_epoch.py](../../../flowstate/strict_epoch.py)。一次物理观测无法“补出”过去从未记录的事件。

## 3. 原始轨迹到底缺什么

[已有直接字段核验](../../kvflow_real_data_connection/final_report.md)覆盖旧24+新5快照的116个 OpenHands Session：984条截断消息、376次工具调用；另检查393条 AgentX trace。OpenHands 有 System 文本和工具交互，不代表没有 Prompt；缺的是固定分段与调用图在当时的声明。AgentX 有子 Agent 容器和事后时间，但 block hash 不可逆，合成回放 Token 不是原始固定提示。

正式 KVFlow 允许从持续缓存命中启发式识别固定前缀；当前归档仍缺 Agent 对应的持续命中、固定段更新/失效记录，不能据此挽救上述数据接入。只因所有受控 Pending 已准备而把 STE 全设0，仍解决不了候选固定/动态分类及未来缓存 Agent 的输入缺口。

## 4. 预算与可见性

Synthetic 图表示预先声明的程序依赖，未来调用身份/顺序可已知；未来真实输出仍不可作为当前输入。CPU中预热和访问事件是 DECLARED，GPU C 后续才证明16个状态驻留；不反向改写原 CPU 产物的 PROPOSED 标签。

4032祖先与4096末端是独立 Recurrent 占用，C的16项必须全部记账；不能让任何基线免费保留祖先。等大小 Recurrent 的 K 可统一，不能声称等价于 Marconi/KVFlow 的原始总 KV 字节预算或 TOPAS 的动态 batching 容量。所有程序必须在同一严格域评价；对当前 Pending 与未来静态调用的目标分别报告，不将后一项称作真实在线 JCT。
