# Retention Capacity Contract：现有能力与缺失约束

这是静态合同说明，不是已实施的新配额，不改任何旧K。判定：**数学上自洽，当前Runtime只实现其中受限的单次reconciliation部分；长期资源合同需要接口扩展。**

## 1. 三类资源与物理集合

P是普通recurrent池可分配槽数，不含padding槽0。按物理槽唯一身份（engine、rank、pool、slot、generation）记账，不能将logical ID直接相加。

- W：活动请求working、tracking及COW目标的物理槽集合。COW目标若就是working，只计一次。
- R_used：其他不可驱逐/传输/会话保护/未纳管的实际槽集合。
- H_ret：拥有跨epoch保留权的可管理checkpoint槽集合。
- H_lease：空闲期间临时借用、必须在下一次admission前归还的checkpoint槽集合。
- Free：尚未分配槽；r_reserved是其中被服务合同预留的数量，不能又当已占用R_used相加。

一般安全式：`|W ∪ R_used ∪ H_ret ∪ H_lease| + |Free| = P`。只有所有权类别已证明互斥时，才可写为各集合大小之和。只读COW source仍属于原checkpoint，不因为新增引用再计一个槽；引用使它暂不可回收。物理共享ID须canonical去重，alias冲突未核验时拒绝驱逐。

配额条件：`|H_ret| <= K_ret`，`|Free| >= r_reserved`（即将派发阶段的要求）。若把未占用预留也纳入R，则用容量承诺式 `|W|+|R_used|+r_reserved+|H_ret| <= P`，不可误写成实际占用恒等式。

## 2. 候选与不可驱逐状态

候选必须具有完整conv+temporal驻留、可验证token前缀/extra_key、与目标共享模型/版本语义、实际物理L、稳定handle与状态代际、完整FA路径；可执行边另须对**实际pending token历史**成立且L≤T。不允许仅凭人工lineage宣布可恢复。

有active、tracking、未完成COW读引用、device/host/session引用、transfer、保护标记或未知所有权的状态不能进入可自由回收集合。adapter已经检查node/prefix/驻留和lock/session/host条件，但完整跨步骤slot-generation与全池owner账本仍缺失。

## 3. 合法的临时候选超过K_ret

若部署预先确定K_ret=P−执行阶段有依据的预留−其他占用，空闲时可以允许H_lease暂借未来执行空间；这要求真实可撤销lease、全入口admission gate和完整物化/所有权事件。候选总数可以大于K_ret，但不能超过物理池容量，且不能把租借槽在H与W双重计账。

下一次派发前必须：冻结admission → 等待在途操作完成 → 同边界读取P/C/FA/recurrent/owner → 验证实际可执行边 → Allocate → 回收未选中、无引用状态 → 检查全局H≤K_ret和Free满足预留 → 才Resume。失败时不派发；当前逐项驱逐没有跨操作rollback，不能声称事务原子性。

如果K就是全部物理驻留的硬上限，C已经驻留意味着K_min≤|C|≤K；不能同时宣称K<K_min。可撤销借用解决的是跨阶段驻留权分配，不是凭空制造GPU显存。

## 4. 现有代码实际实现了什么

`flowstate/optimizer.py:65–85`仅检查等大小并将budget_bytes整除为逻辑项数。`flowstate/controller.py:55–116`对传入候选做当轮选择及未选中状态驱逐，不记录跨轮配额；它不枚举候选集外状态。因此只能约束输入候选中所保留的项，不能自动保证全局persistent使用≤K。

`tests/runtime/rq6_batched_control_transport.py:194–238`检查选择合法、view digest、前后状态和FA保存；`:275`在fully_idle时服务RPC。`tests/runtime/control_path_optimized_transport.py:16–29`的epoch token防止句柄重复消费，但不是physical slot generation，更不是跨RPC持有调度事务锁。

`flowstate/adapters/sglang.py:332–414`重新验证node、exact前缀、FA路径、recurrent状态及引用；`:416–472`仅detach/free MAMBA并验证释放范围。归还slot后可由池复用，不代表预分配tensor的HBM被释放。底层slot分配/释放独占性与generation的当前SGLang源码副本缺失，保留P3历史证据等级。

尚未实现或尚未完整证明：全入口长期K_ret quota、reservation/lease兑现、slot generation/owner引用全集、全局persistent账目、Agent Ready与runtime barrier联接、整个Observe→Resume区间的外部admission封闭。不能把现有K改名成已实现的K_ret。

## 5. 分环境证据，不跨实验拼接

| 环境 | 物理池 | 实际观察 | W/Tracking/COW与H | 配额/压力结论 |
|---|---:|---|---|---|
| AgentX B2 | 192 | 23快照末次census空闲43–185；g000为82空闲、110 evictable、0 protected，候选仅109 | 不能把109候选等同全部H；瞬态历史有census但无完整角色owner事件 | 旧百分比K非池上限，无长期quota；没有该边界物理耗尽 |
| F1 | 471 | 主扫描最终空闲468/466/462/454/438/406；2子请求指纹修正运行空闲466 | 源槽2→工作3/5，tracking4/6；完成后缓存保留5槽。运行中不可将source与读引用重复加总 | K字段为children数，无retention K；未观察到容量墙 |
| WP3A | 16 | 每arm对称驱逐target或decoy一槽；raw CSV有available前后与+1 | 单consumer受控干预；不是完整quota分配 | 无独立retention quota；不能用人为干预证明自然紧缺 |
| OpenHands | 28 | 历史校准20 persistent、reserve4+spare4；离散census最低free8；串行并发1 | 这是校准配置/观测，不能将reserve视为永远已占用的槽 | 逻辑K取候选比例，单次可实现；无长期池配额合同 |

B2来源：`/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/per_snapshot/*/snapshot.json`，本轮汇总`pool_observations.json`。F1来源：`/home/wjg/code/newfind/artifacts/exp_f1_hybrid_fork_20260814T160725Z/raw/`。WP3A来源：`motivation/artifacts/wp3a_direct_eviction_20260819/raw_runs.csv`。OpenHands来源：`evaluation/runtime_artifacts/openhands_4workflow_occupancy_calibration_20260831_050831_071744/summary.json`及P3物理核算报告；非新测量。

K_min仅是候选最大前沿的逻辑覆盖指标。只有每项确为独占完整状态、合同已绑定物理资源并可兑现时，K_ret<K_min才有长期保留取舍意义；仍不意味着当前HBM已耗尽，不保证任何算法更优。
