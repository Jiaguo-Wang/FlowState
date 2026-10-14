# 新 cohort 的缺失证据与未来采集要求

**状态：条件性方案，尚未注册；未创建新 cohort。** 当前文件不足以直接构建满足严格门禁的新 cohort。下列是后续单独设计任务必须冻结的要求，不是本轮采集授权。

## 1. 来源、版本和范围

不得覆盖 OpenHands 168 / AgentX 23 的正式总体。AgentX 原始文件摘要为 29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276；它可用于任务/时序形状参考，不能补出原生 fork 或驻留。若使用该来源再执行，必须明确为新的受控执行，不声称恢复了原始时刻的状态。

真实执行框架版本、任务来源与许可、容器/模型/tokenizer/runtime commit、记录格式版本、全量入组时间窗口 [start,end)、capture 频率和停止样本数均必须在第一条数据生成前写入注册文件并签名/哈希。当前这些来源绑定参数为 UNKNOWN/待定，不填写虚构时间或 cohort 数量。没有确认自然出现所需结构前，不承诺样本规模或效应大小。

## 2. 联合事件记录

必须记录稳定 root_workflow_id、branch_id、request_id、parent_request_id、原生 fork_event_id、继承前缀长度与 token/model/tokenizer identity、state generation。Fork 必须由框架真实事件和父执行证据确认；LCP 只作一致性校验，不能替代父子继承。不同任务的相同 prompt 不合并为同一 recurrent state。

请求状态机至少包括 ready、queued、dispatch、running、complete/cancel；每个事件有单调序号和同钟 timestamp。记录实际 barrier 起止和在该边界阻止状态变化的机制。pending 定义为边界之前已 ready、尚未 dispatch、未 complete/cancel 的真实 continuation；至少两个，不能用未来请求或已执行请求的上下文凑数。不能仅以 conversation start/end 间隔判定。

原生 epoch 捕获规则必须在观察策略结果前固定，例如每个真实调度分配 barrier 记录一次，再按预先固定的 root-workflow/epoch 序号抽样。不得延迟或重排执行以制造有利需求。所有捕获和排除都保留日志。

## 3. 可执行语义和 eligibility

在同一边界观测 A_p、attention-side R_p、T_p=min(A_p,R_p)，不把历史 token 长度当 R。每个候选记录创建与有效时点、L、checkpoint ID 到 rank-local state handle/slot generation 的映射、所有必需 recurrent/conv 组件、resident/valid/protection 状态。状态 observation 必须非干扰；epoch 内不得发生未受控 request execution、native eviction 或 slot reuse。

候选必须已物化且合法可管理；不同 logical ID 不凭 slot 整数去重。共享推荐的同一 canonical checkpoint 只计一个逻辑单位。每条兼容边同时验证原生 ancestry、模型/缓存兼容和 L≤T；超过 T 的真实 checkpoint 可以留在候选总体但不对该 p 产生可用边，不为了缩小总体删除不利候选。冻结 Phi 定义域为 T≤131072；旧 context census 的 131200 阈值不能自动作为模型门禁。

缺少父事件、pending 状态、状态 identity、rank 证据、同 epoch 一致性或输入越域者，写入排除日志：source ID、epoch、gate、reason、字段缺失、证据哈希。UNKNOWN 不能填零或转 PASS。不能按任何策略胜负排除。

## 4. 预算协议与容量证书

A 协议：如保留 25%/50%/75% candidate-based K，必须全部预先规定，并如实报告无收益/重复候选、有效 pending、K_min_frontier 和宽松程度。不能变更旧 573 cases。该协议利于与旧数据描述性对照，但不保证产生有效容量竞争。

B 协议：新 cohort 可使用实际系统容量预算。先按部署资源合同固定 recurrent 管理池字节 B_r、工作/active state 预留 W_r、transient/COW/headroom H_r，以及 checkpoint 完整组件每 rank 字节 b_r；逻辑上限为 min_r floor((B_r-W_r-H_r)/b_r)。任何额外限额须有明确系统理由。attention/KV 预算保持独立，不挪用容量证明。数值需根据硬件/服务约束在看 workload 的 K_min 或策略结果之前冻结，不能设 K=K_min−1 来强行制造困难，也不能由胜负反向选 K。等大小或 TP 假设不满足时，应重新做独立设计，不临时改正式算法。

本轮没有给新 cohort 选一个具体 K，因为缺少已核准的物理资源合同。K 和 capture 规则冻结后，按事先规定的 shared/unshared、binding/nonbinding 结构层报告全体及子层。结构性富集只能预注册并完整报告原始分母，不能只保留正收益例。

每个 epoch 计算 E_p(C)。令 Q 为 E_p(C)>0 的 p；每个 c 的覆盖集合是其能精确达到这些最深前沿的 p。对该覆盖问题精确求 K_min_frontier，并保存所选 ID 证书或其他可验证最优证明。空 Q 时 K_min=0，K/K_min 不适用；计算不可完成则 UNKNOWN。K<K_min 仅证明有效容量限制，不证明 FlowState 胜出。明确记录共享祖先与至少两个分支更深专属项是否同时存在、收益覆盖是否重叠。

## 5. 后续拟比较的方法与指标

仅在数据独立冻结、门禁通过且另获执行授权后，使用不变 FlowState、PCB、LDR、WFC、StandaloneTopK、LRU、LFU、Marconi Adaptation；不把 adaptation 称作完整外部系统。Exact OPT 仅对预先固定的可穷举阈值内实例，完整公布分母和未求解实例，不因难以穷举而删总体。

共同评分：modeled C(S)、逐 p E/G、selected sets、容量证书、配对胜平负、greedy–Exact gap。Full Coverage 明确定义为所有当前 pending 的整数 G_p=0；达到全候选前沿另外报告，不能混为 Full Coverage。成本 tie 使用冻结 1e-9 modeled ms，不能把非零整数 gap 以成本容差抹掉。

模型成本与 CPU allocation 结果分开。实际 TTFT、control/reconciliation 和 control-inclusive E2E 需后续独立 runtime protocol，本阶段不测，也不把旧结果拼接成新结果。

## 6. 统计与停止规则

多个 budgets 是同 snapshot 的重复条件；同 root workflow 的多个 epoch 有相关性。主配对与重采样以独立 root workflow 为 cluster，预先规定 epoch 权重/聚合，另报每 snapshot 结果。不能把所有请求、epoch、预算摊成独立 n。允许持平、负例和模型选择器非最优结果，完整公布。

未找到自然产生且可审计的紧约束共享结构时，保留可行性失败结论，不增加合成 fork、不放宽 pending/eligibility、不按结果调预算。本文件到未来采集设计为止；本轮未采集、未创建 cohort、未跑策略、未运行 GPU。
