# Phase B4-A：真实负载采样协议冻结

## A. Final Gate

**PARTIAL**。

现有设施可支持“真实OpenHands消息来源、固定分组与已知输入的受控回放”小规模严格准入采集；不能据此支持“自然在线Agent Ready或自然Allocation Boundary”的采样主张。当前源码和数据没有提供足以认证原生根Workflow、跨Session因果关联、真实Ready时间及全局事件次序的证据。

协议已写入 `sampling_protocol.json`，本轮只审阅源码、冻结清单和Parquet页脚，没有正式采样、读取新Session消息、生成Token、运行GPU或比较策略。未修改B1/B1-C/B2/B3或生产代码。

三项实质限制必须先于B4-B明确：

1. 自然在线边界不可用；只能实施明确命名的受控回放范围。
2. 四Session组成一个边界，30个源Session上限只容纳7个完整组，不能承诺12个边界；保留原分组，不把组当作一个根Session。
3. 既有组顺序是固定hash顺序，而非原始时间顺序。如果“原始顺序”要求原生全局时序，当前数据不满足；本协议仅冻结已有来源清单顺序，不能冒称自然事件序。

## B. 合法Pending来源与时间边界

### 已确认路径

- `evaluation/rq3_openhands_population.py:26–45` 固定源、seed、4Session组与round2/3/4/5；`:100–139` hash排序及round轮转；`:142–176` 连续分组。既有group不是自然Agent Fork或Join。
- `evaluation/rq3_openhands_neutral_collector.py:236–303` 的 materialize_group_requests 在遇到第r+1个assistant消息时，用该消息之前的history及Chat Template生成Pending输入，随后停止；不把该assistant输出加入输入，不读取r+2消息。
- 这段history可以包含原轨迹中已记录的工具输出及此前assistant文本。它们在受控回放准备时已知，却不证明是当前目标模型生成后自然触发的工具结果。历史请求通常只生成1个Token，后续历史输入仍来自记录，不能称闭环Agent重执行。
- 同文件`:333–386` replay_group_to_barrier 按turn再A/B/C/D串行执行到指定round；不是并发Agent的原生等待队列。
- `evaluation/strict_compatibility_b1/run_pilot.py:122–205` 回放完成后记录barrier、census并检查候选Token前缀、node和驻留；`:205–235` 保存Pending输入及T=min(anchor,FA)。main的group参数在`:457–479`，但原main还会启动reference及evicted流程，不适合直接批量运行。
- `evaluation/strict_compatibility_b1c/runner.py:28–72` 观察原始match返回及req.origin_input_ids摘要；`:75–109` 在静止调度器读取完整树、conv/temporal和引用、验证观测非干扰。B1-C成功证据证明该工具链在g000可用，不证明其他组已经通过。

### 输入何时“当前已知”

受控边界前显式构建并冻结每个Pending的完整Token缓冲，记录prepare时间及摘要；此后将其视为“客户端准备好、尚未dispatch”的已知输入。该定义只适用于这个受控回放，不把离线文件中可见的未来消息当作自然在线已知。

候选Token身份须从实际已提交历史输入和物化节点对应起来；Pending实际Runtime接收身份须由原始请求路径记录。先冻结整个边界证据，再按固定顺序提交Pending核对接收摘要。后续提交只能确认输入与冻结缓冲一致，不将其生成的新状态、E或Handle反填为之前边界证据。

自然Agent何时真正Ready、是否同时Ready、工具结果是否已在原生执行中产生、Root之间是否相关：**UNKNOWN**。现有source_id、session_id、messages序列不能替代原生Ready/Fork事件。

### CPU侧检查范围

Parquet `/home/wjg/data/agentic_coding_trajectories/sessions.parquet` 存在，325563194字节、15000行、8个row groups；字段为session_id、source_dataset、source_id、agent_framework、recorded_model、messages_json、n_turns、max_isl、total_tokens、ground_truth_meta_json。本轮仅读页脚，不判断嵌套messages是否还含未知时间字段；顶层没有原生Ready/parent字段，未发现现有Runner使用此类原生事件。

冻结population_manifest共有662组，验证全部组内4个唯一Session、ordered_session_ids唯一且分组切片一致。它来源于旧dataset-level资格筛选（含总turn数等），所以是历史筛选总体的受控抽样框，而不是无条件代表所有OpenHands流量的新随机样本。文件来源、参数和哈希见cpu_preflight.json。模型目录及uv312存在；没有启动Docker或查询当前GPU，当前驱动、容器权限和空闲状态仍待B4-B运行前核对。

## C. 三类结构可采集性

| 类别 | 固定结构判据 | 已有能力/限制 | 本轮结论 |
|---|---|---|---|
| A 重复链与外部竞争 | 同一Pending上祖先/后代候选均合法，另有候选改善链外Pending；至少2 Pending、3候选 | B1具备多个同支历史Checkpoint；新组是否物化完整须Runtime确认 | 工具支持，发生频率UNKNOWN |
| B 共享浅层与专属深层 | 祖先准入至少2 Pending，存在更深专属分支候选，K_min>1 | g000已有共享证据；其他组不能凭文本共前缀认证 | 可测，不能保证新样本存在 |
| C 不同T/深度竞争 | 两个互斥正推进支持项，对应T不同，且L或合法基础E不同 | T需要同边界FA观测；深度来自物化节点 | 可测，结构差异不保证目标差异 |

判定只使用Token准入、L/T/E、覆盖和预算结构，不读取Phi收益、策略选择或胜负。若无合法基础前沿证据，不能虚构e0；候选外有效状态影响E时实例标不完整并报告。

固定主层优先级A>B>C；全部附加标签保留。主层满4后不把样本转派B/C以补额。A较宽，可能遮蔽大量同时满足B/C的边界；这是该优先级的已知设计限制，不能看结果后换优先级。未来报告必须同时列主层计数与非互斥结构标签计数；某层为0也停止。

## D. 冻结采样协议

### 计数、顺序与总体

- 上限保持30个根Session、100个自然Allocation Boundaries、每层4、总12、每根最多入选1个边界。检查失败及无结构也消耗检查额度。
- 当前仅可识别源Session ID；将每个源Session作为保守的检查配额单位，但**不能由此证明原生Root唯一性**。若未来确认两个Session同根，必须去重按原事件顺序处理，不能重分组补位；若原生root身份是必须认证条件，则当前自然轨道停止。
- 受控轨道按既有manifest的group_ordinal升序，原A/B/C/D成员与allocation_round不变，不按旧168成功列表、历史策略结果或新准入成败过滤。组内turn升序，再A/B/C/D。没有全局自然时序的主张。
- 30个Session最多取前7个完整组g000–g006（28个Session）。剩余2个不拼成新组、不与后续Session组合。不得将30个组误称30个Session。
- g000为B1/B2已知发现案例，描述性单列，不计“新确认性样本”；仍占配额，不跳过后补。新边界上限因此为6，整体描述性上限7，均小于12。
- 每组只检查原指定round的一个边界，不为了找结构在同根其他round反复尝试。所有结果，包括失败、无标签、主层已满和UNKNOWN，保留审计账本。
- 受控回放另记replay_barrier_count，不把它当natural_boundary_count。当前原生自然边界认证数量为0。若必须严格实施“100自然边界”轨道，则该轨道尚不可执行，不能用受控计数替换。

### 同边界准入与预算

记录完整候选×Pending笛卡尔配对。EPOCH_ADMISSIBLE需满足Token前缀、L≤T、同配置、完整状态/映射、同边界及安全引用条件；明确违例REJECTED；缺项UNKNOWN。CPU重建消息不能直接授予准入。

存在会影响矩阵结论的关键UNKNOWN时，保存实例但不纳入完整严格矩阵比较；不把UNKNOWN改为0或删边后假装完整。若具备准入但缺逐边实际resume认证，保留两种证据等级，不将所有边升级。

对每个正式完整实例固定K=1…min(4,候选数)；单独计算K_min_frontier。后者为达到全候选最大E的最少逻辑项数，不保证G=0，不等于物理容量。本阶段不执行任何K sweep或新比较。

### 未来B4-C统一指标

1. 检查Session/组/边界数，合格、失败、UNKNOWN、无标签、层满额排除数；源Session数与原生Root认证数分列。
2. 每快照和合计EPOCH_ADMISSIBLE / REJECTED / UNKNOWN数量；每个判断原因与证据来源。
3. 跨Workflow共享比例至少列两个分母：跨Workflow准入边/全部准入边，以及跨Workflow准入边/全部跨Workflow配对；同时列计数，不把跨Workflow当真实Fork。
4. K、K_min、K<K_min比例，逐策略J、每Pending E/G和选中集合；同收益异集合及顺序差异单列。
5. FlowState对WFC、StandaloneTopK、PCB/LDR的胜平负；容差1e−9 modeled ms；Exact OPT成本差距和求解状态。
6. 所有主层实例及附加标签、负例与持平，不仅代表性正例。多预算是同快照重复条件；不得把边、请求或预算当独立统计样本。

预设候选最多4r≤20时，CPU Exact最多2^20子集，可用确定性穷举或等价精确算法；遇额外物理候选导致规模更大，应记录精确求解未完成而非按结果删候选或冒称Exact。策略比较仅由后续B4-C任务授权。

## E. 最小Runtime Patch Plan（只设计）

不修改Optimizer、Controller、SGLang或旧采集脚本，新增独立采集入口并复用现有函数：

1. 参数化group与新输出路径、预注册计数器、失败账本；拒绝覆盖旧文件。
2. 将B1的输入准备、history replay、barrier证据提取为capture-only调用，不调用其reference/evicted多臂验证流程。每组新Engine，不保留前一组缓存。
3. 复用B1-C原始match只读观察器，保留历史请求实际Token摘要、Checkpoint来源、完整Token路径、conv/temporal映射和引用；不能用旧node/slot常量。
4. Pending准备时冻结完整输入及配置身份，记录prepare/ready/dispatch边界。此处ready明确命名为controlled_input_ready，不伪装agent_native_ready。屏障冻结后按A/B/C/D提交每个Pending、各1个确定性Token，仅核对实际接收身份；不比较策略、不据此宣称每条边实际命中。
5. 保存creation_order、last_access_order、原始match输入和既定Marconi元数据，避免B2的时序缺口；无需为本次采样启动Marconi选择。
6. 采集完整树以排除候选外状态改变前沿；静止期间census前后非干扰校验，保存池占用、headroom和保护状态。若alias/组件/前沿来源无法说明，标UNKNOWN并停止该实例，不削弱门禁。
7. 结构分类及quota决策在准入矩阵冻结后进行，不提前计算策略胜负；只有元数据自检和既定门禁可以终止。

原生Agent Ready/parent/root事件不能靠这些日志字段新增“实现”。若论文必须要求自然在线Pending，需要单独提供原生事件源；这不是本轮的小补丁能够认证的能力。

## F. 资源与执行开销

- 复用B1-C已验证的SGLang0.5.17镜像、Qwen3.5-9B、TP1、单H100、28 recurrent slots及原引擎参数；当前路径存在，动态可运行性待执行前检查。
- 受控新采集最多6个独立Engine lifecycles（g001–g006），不重跑g000；不是12，更不是216。每组1个采集生命周期足以获得新边界证据。
- 对应原round序列3/4/5/2/3/4，最多84次历史请求，加冻结后24次Pending身份核对，共108次请求。失败提前停止时少于此数。这是代码路径推导的上限，非实测执行记录。
- 默认不运行冷参考或隔离反事实；若另行要求每组输出参考，再加每组1个冷引擎且请求间清缓存，最多12个生命周期，但这是额外授权范围，不能本轮自动加入。
- r≤5最多20个历史检查点，28槽可能提供临时余量，但实际pool还包含工作/Tracking/COW，不能从28−20直接宣称安全可运行。启动门禁、实际占用和native eviction记录决定通过与否；不得失败后增池或删候选求通过。
- 墙钟/GPU小时UNKNOWN。本轮没有测量，不能拿g000单例启动或延迟直接外推其他输入长度。禁止以采集耗时估计策略性能。

## G. 停止条件与B4-B建议

达到任一检查上限、入选上限或完整分组界限即停，不延长搜寻满足层额或正例。身份、因果输入、状态完整性、保护冲突、native eviction污染、OOM或清理失败停止该实例；共性必要接口不可用时整批停止，不重试到成功、不换样本。

**唯一建议：B4-B先完成独立capture-only入口和CPU合同测试，并明确接受“受控已知输入回放、非自然在线Agent”的适用范围，再按锁定清单执行最多6个新边界采集。** 执行前还须完整数据摘要锁定、环境只读核验及门禁通过。本报告不构成自动运行B4-B的指令。

若目标必须是自然在线多Workflow Ready与原生Root去重，则当前设施不满足，不能将受控轨道包装为完成该目标，应先补原生事件证据，而非直接扩展GPU回放。

## 产物与验证

- sampling_protocol.json：机器可读协议、范围、优先级、上限、指标和补丁计划。
- cpu_preflight.json：清单一致性、Parquet页脚、源代码哈希与路径存在性；没有检查新会话内容。
- freeze_protocol.py：只读元数据核验及协议生成入口。
- evidence_manifest.json：本轮产物摘要。旧冻结来源的哈希在cpu_preflight.json中；数据当前仅元数据身份，正式B4-B前必须补全文件SHA-256，不能将mtime当不可变内容证明。
