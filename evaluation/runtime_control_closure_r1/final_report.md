# Phase R1：严格兼容性到 Runtime 控制闭环审计

## 最终门禁：BLOCKED

**门禁对象是“直接进入 g001–g004 严格矩阵的正式 Runtime Policy A/B”，不是否定已有单节点驱逐或 B1-C 恢复实验。** 当前具有真实可用的状态观测、Recurrent-only 驱逐和请求恢复原语，但没有证据表明生产入口已经把 B4-C 严格准入矩阵、当前物理映射、逻辑预算后置条件与实际恢复验证串成同一个闭环。

关键阻断是明确的数据流矛盾：生产 `GlobalOptimizer` 使用旧 Workflow-ID/Lineage 兼容性；B4-C 通过覆写成本接口绕过它，却只执行 CPU 选择，未调用 Controller 或 Runtime。直接把严格 Snapshot 转成生产核心对象会丢失严格矩阵，并拒绝已经准入的跨 Workflow 边。另有严格选集尚未协调、映射/别名缺乏统一前置合同、策略间等价初态及输出参考未验证等前置缺口。

本轮仅阅读源码与冻结记录，新增本报告；未执行求解、GPU、Engine、Docker、回放、测试或修复。所有路径相对于 `/home/wjg/code/FlowState`，容器内历史源路径单独标识。VERIFIED、PARTIAL、BLOCKED、UNKNOWN 均按具体链路范围解释。

## 1. 生产控制调用图

### 1.1 核心库路径：有选择与驱逐，没有原生严格矩阵入口

```text
调用方提供 PendingContinuation / CheckpointCandidate / handles / budget_bytes
  → StateController.reconcile
      → validate_unique_checkpoint_ids + _validate_handles
      → GlobalOptimizer.select
          → _recovery_cost
              → recovery_gap → executable_frontier
                  → state_catalog.is_compatible
                      → Workflow ID 相等、Lineage 前缀、L≤T
      → 对 candidates 中 resident 且不在 selected_ids 的项逐个驱逐
          → RuntimeAdapter.evict_mamba_only
              → SGLangAdapter.evict_mamba_only
                  → 精确树定位、身份/引用/传输检查
                  → 原生 MAMBA 组件 detach/free
                  → 局部后置检查
      → 返回 allocation
```

依据：`flowstate/controller.py:55–112`；`optimizer.py:34–145`；`executable_state.py:11–35`；`state_catalog.py:60–73`；`adapters/sglang.py:99–121,291–533`。

Controller 不生成 Pending、不捕获 Token、不建立严格边、不拥有 scheduler admission 屏障。`SGLangAdapter.inspect_checkpoint:89–97` 明确抛出 NotImplementedError，因为句柄不能自行提供 Workflow/Lineage。现有 Candidate/ Pending 核心对象也不包含严格边矩阵、Token 身份、配置摘要或 Barrier 证书（`state_catalog.py:11–27`，`workflow.py:8–28`）。这不是一个自动接入在线 Agent Ready 的完整服务入口。

### 1.2 严格证据采集路径：没有 Allocation / Reconciliation

```text
B4-B0 capture.main
  → 准备固定组与当前回放 Pending 输入
  → neutral_collector.replay_group_to_barrier
  → 静止状态、全树 census、候选精确身份、FA 查询
  → contracts.build_edges / edge
  → freeze_snapshot
  → 单独提交 Pending 核对实际输入身份
  → 清理
```

依据：`evaluation/strict_compatibility_b4b0/capture.py:40–106`、`contracts.py:36–81`；`evaluation/rq3_openhands_neutral_collector.py:333–590`。历史按轮次及 A/B/C/D 顺序串行执行，每个历史请求要求新增唯一 Recurrent 节点（约 430–470 行），在 Barrier 重新定位并核对状态（约 510–550 行）。Pending 使用 allocation_round+1 的已准备输入，FA 来自只读 `inspect_fa_frontier`；T=min(anchor_pos, resident_fa_frontier)，见 `workflow.py:24–28`。

这属于受控已知输入回放，不是自然 Agent 原生 Ready。`capture.py:88–104` 先冻结 Snapshot，再做输入核对，后续新状态不得回填。该程序没有按 K=1–4 驱逐未选中状态。

### 1.3 B4-C 路径：严格矩阵接入选择逻辑，但没有物理执行

`evaluation/strict_compatibility_b4c/evaluate.py:26–29` 的 `MatrixOptimizer` 继承生产 Greedy，**仅覆写 `_recovery_cost`** 为 `common.score(epoch, selected_ids)`。`load:31–47` 从 EPOCH_ADMISSIBLE 边构建 `common.Candidate.compatible`；`evaluate:49–88` 进行 CPU 选择、评分和穷举，没有 Controller/Adapter 调用。

`common.score` 位于 `evaluation/p1_strong_baseline_validation/common.py:59–71`，用显式 compatible 集合算 E/G/Phi；没有再次调用旧 Workflow 谓词。`MatrixOptimizer.select((), core, ...)` 的 continuations 是空元组，覆盖后的成本函数依赖独立 epoch；不能把该调用模式直接视为生产 Controller 已接入严格状态视图。

### 1.4 既有 Runtime 控制路径：物理机制存在，但输入不是 B4 严格矩阵

- 旧 RQ4：`evaluation/rq4_unified_runtime_harness.py:590–655` 使用 `StateController(FrozenSelectedSetOptimizer(selected_ids), RecordingRuntimeAdapter(...))`。它验证指定集合可落实，不证明本轮在线严格矩阵计算已经接入。
- 优化后 E2E：`evaluation/control_path_optimized_e2e.py:43–70,173–215` 构造当前观测候选与 Pending，调用普通 `GlobalOptimizer`，再调用批量客户端。该选择仍走旧 Workflow 谓词。
- 批量客户端：`evaluation/control_path_optimized_client.py:46–73` → `tests/runtime/control_path_optimized_transport.py:64–95` → `tests/runtime/rq6_batched_control_transport.py:194–243` → `SGLangAdapter`。
- B1-C：`evaluation/strict_compatibility_b1c/runner.py:218–288` 手工确定隔离计划，经同一 Adapter 驱逐原语恢复单个目标请求；没有调用生产 GlobalOptimizer/StateController 计算该隔离集合。

## 2. Compatibility 集成审计

|检查|输入与实际操作|证据及结论|未闭合项|
|---|---|---|---|
|当前 Pending 与 T|回放已准备输入、FA 无副作用查询，T=min 输入 anchor 与 FA|VERIFIED：B4-B0 capture 与 neutral collector；B4-B1 Snapshot|不是自然在线 Agent Ready|
|候选生成|历史请求物化新增节点，精确 Token 前缀、node/slot 回查；全树 resident 与候选节点相等|VERIFIED：capture.py:64–73；B4-B1 g001–g004|只在当前采集屏障成立，不能沿用历史 Engine 的 node ID|
|严格准入|Token、L≤T、配置、边界、组件、引用及 alias 检查|VERIFIED：contracts.py:50–81；T2 structure_audit.json|不是逐边实际恢复验证|
|UNKNOWN|false→REJECTED；缺字段→UNKNOWN；B4-C 要求无 UNKNOWN|VERIFIED：contracts.py:73–74、B4-C load:36–40|生产核心只有布尔驻留，无三态证书消费接口|
|生产选择关系|is_compatible 要求同 Workflow 与 Lineage|BLOCKED：严格跨 Workflow 关系不能按现有默认路径参与收益|接入已验证的不可变 epoch 关系，而不是伪造 Workflow ID|
|B4-C 适配|显式矩阵 common.score 绕过旧谓词|VERIFIED，仅 CPU|没有调用物理协调、请求恢复或注册生产入口|
|stale / 配置变化|Adapter 单目标可拒绝部分变化；批量 view 可拒绝观测后变化|PARTIAL|严格边证书、配置、选集与实时 view 尚无一体化校验|

**确定矛盾示例：** g001–g004 各共享 B1 对其他三个 Workflow 有准入边，合计 12 条；B4-C/T2 的 224 配对中有 68 准入。生产 `state_catalog.py:67` 必定排除这些 Workflow-ID 不同的配对。这证明默认生产评分与严格评分语义不同；没有重新计算默认策略的选集，也不推断必然在每个 K 改变最终结果。

## 3. 逻辑到物理映射审计

### 3.1 可确认的映射与安全检查

`RuntimeCheckpointHandle`（`flowstate/adapters/sglang.py:18–47`）包含 checkpoint_id、完整 Token 元组、extra_key、可选 expected_node_id 和 expected_prefix_digest；**没有预期 slot、slot generation 或 allocation epoch 字段**。

`_find_exact_node:291–330` 遍历精确 radix 分段，拒绝分段内部的近似终点，不调用可能修改替换状态的 match lookup。`_validate_target:332–414` 检查 node_by_id 仍指同一对象、可选 node ID、前缀长度/摘要、FA 路径、Recurrent 驻留、device/host/session refs、无 host copy、LRU 成员及运行范围。可防止已发现的 stale node、缺组件、锁定状态被静默驱逐。

这些检查不是通用的槽位所有权代际证明；同 node 上发生状态替换、slot 重用后再恢复相同外观的 ABA 不能仅靠该句柄确认。生产 Controller 的 `_validate_handles:115–137` 只验证字典键/逻辑 ID 一致，没有检验不同 ID 是否指向同 node 或同 slot。缺句柄检查只覆盖待驱逐项（89–96 行），选中项缺物理句柄也不在核心 Controller 中直接拒绝。

### 3.2 批量路径更强，但不能叫原子事务

`rq6_batched_control_transport.py:54–92` 的 view 包含 scope、全树、accounting、各句柄路径；`194–225` 在任何驱逐前重新读 view，必须与 expected_view_digest 相同，并验证所有候选 node/前缀/驻留。优化传输 `EpochHandles.take:23–29` 拒绝重复消费、错 token、错候选顺序/集合和视图；协调 finally 清除 epoch 上下文。它们能在变更前拒绝已观测的 stale view。

安全时点由测试传输安装的 scheduler `on_idle` 和 `_validate_runtime_scope` 支持（批量 transport:256–282），不是 Adapter 自己停止整个 scheduler。Observe 与 Reconcile 是不同请求，view 相等检查不是全时段锁或跨 rank 事务。选集成功后到 request dispatch 的期间仍需调用方禁止并发 admission；当前生产核心不提供该屏障。

### 3.3 别名与失效处理范围

B4 的 `contracts.build_edges:76–81` 检查候选 slot 重复，`capture.py:72–73` 检查全树候选覆盖；T2 在四个快照未发现相同前缀/位置重复项。这足以支持这些采集边界的独立候选假设，**不证明任意未来 Controller 输入无别名**。

通用 Controller 可以接收两个不同逻辑 ID 指向同 node 的句柄；若一个选中、另一个未选中，ID 级排除不能证明释放对象不属于选中集合。批量 `_proof` 可能在驱逐后检测 selected 不驻留，但检测发生在破坏之后。这里是源码合同缺口，不是报告 B4 已发生该冲突；四个冻结快照没有观察到该反例。

因此映射结论为 **PARTIAL**：已有精确定位、stale 拒绝与后验验证；缺少把严格 epoch 证书绑定到当前生命周期映射并在变更前验证 keep/free 物理集合互斥的统一入口。没有自动重新决策或回滚。最小处理是拒绝并终止受影响生命周期，不可以沿用旧 S 继续恢复。

## 4. 逻辑预算落实审计

### 4.1 没有容量压力也可以主动驱逐，但需要实际调用

`StateController.reconcile:77–110` 明确遍历当前候选中所有驻留且未选中的项，并调用 Recurrent-only eviction；不检查 pool 是否已满。因此在无别名、可安全驱逐、视图覆盖完整、调用成功的条件下，可以主动落实逻辑保留预算。归还 slot 是预分配池的可复用空间增加，不等于释放相应 HBM allocation。

批量路径 `rq6_batched_control_transport.py:194–243` 接受 selected_ids，驱逐其在 handles 中的补集。`_proof:94–176` 验证 selected 驻留、evicted 缺失、node/路径身份不变、FA 树/分配器不变、变化节点恰为目标、free slots 增量及节点减少数符合预期。

但该批量 RPC **不接收 K/budget_bytes**，仅能实现传来的集合，无法独立证明 |S|≤K。它也不是持久 quota 管理器：没有为 selected 设置新的 pin、保护引用或保证今后 native eviction 不触及它。这里“保留”是当前协调不驱逐及后验驻留，不是长期保护。

### 4.2 B4 的四类集合不能合并

|组|B4-C 逻辑 S|采集 Barrier 的物理 resident|当时 protected / 由策略 evicted|按 B4-C S 恢复的实际 E|
|---|---|---:|---|---|
|g001|每策略/K 单独保存于 B4-C CSV/JSON|12 候选，16 free slots|protected=0；未执行该策略驱逐|UNKNOWN：未运行该比较|
|g002|同上|16 候选，12 free slots|protected=0；未执行该策略驱逐|UNKNOWN|
|g003|同上|20 候选，8 free slots|protected=0；未执行该策略驱逐|UNKNOWN|
|g004|同上|8 候选，20 free slots|protected=0；未执行该策略驱逐|UNKNOWN|

来源：各组 `snapshot.json/barrier_start/accounting, nodes, candidates`；逻辑集合来源 B4-C。每组 28-slot pool，不能把 K=1–4 称为物理池耗尽。B4-B1 在全部候选仍驻留时执行了 Pending 输入身份核对；其 E 不是执行 B4-C 选集后的 E，不能用这些请求认证该策略结果。B4-C 的“未选中”是离线算法状态，不是历史 Runtime 的 eviction 事件。

### 4.3 未管理状态、别名与请求间变化

B4-B1 捕获时全树 resident 节点等于候选节点，是 **VERIFIED 的边界事实**；恢复期间新请求可以物化新状态，后续请求面对的集合可能不再只是最初 S。`capture.py:102–104` 只要求旧节点/slot 未丢失替换，允许新增节点。生产 Controller 仅驱逐传入候选，不会主动枚举候选之外的可恢复状态。因此“一次协调后所有四请求 E 恰由固定 S 决定”仍需逐请求证据，不能从初始 census 推出。

不同逻辑候选若共享同一物理对象，|S| 与释放数量不再一一对应；批量 proof 的节点/slot 计数采用独立目标假设。严格四快照当前没有该别名，但通用映射入口必须在变更前拒绝不符合的输入或显式引用保护。不能通过双重计数声称执行了物理 quota。

## 5. 组件隔离、失败、Backup/Restore 与 COW

|能力|路径/操作/后置条件|判定与边界|
|---|---|---|
|Recurrent-only 驱逐|Adapter `_evict_mamba_component_only:416–469` 调用原生 `_evict_component_and_detach_lru`，仅 MAMBA/DEVICE，再 `_free_values`|VERIFIED：已有原语与 B1-C 受控反事实；并非新发明的 SGLang 原语|
|FA 隔离|Adapter `_validate_postconditions:492–531` 检查 FA 路径对象、树结构和分配器；批量 proof 比较全树与路径摘要|VERIFIED：限定已验证模式；不能仅凭局部对象相等推出任意 GPU 张量内容完全一致|
|active/transfer 保护|`_validate_target:391–414` 引用与 host copy；`_validate_runtime_scope:533–557` 拒绝 ongoing insert/transfer/eviction、HiCache、session radix、int8 pool|PARTIAL：依赖调用方安全时点；无通用跨候选所有权保护证明|
|选中状态保护|ID 不在 eviction 补集中；批量 selected_resident 后验检查|PARTIAL：没有新增 pin/lease；不是直到未来请求结束的持久保护|
|失败不静默|Controller `ReconcileExecutionError:23–40,104–109` 报告已完成项；客户端 call:28–44 检查 ok 并异常关闭；batch proof 失败抛异常|VERIFIED：不会把检测出的失败当成功；但非回滚事务|
|部分变更后失败|Adapter:433–448 finally 仍调用 free；后续范围或 sanity 检查可抛错；batch 顺序驱逐无 rollback|PARTIAL：需 fail-stop、保留失败账本，不可继续使用原视图。服务端异常回复未必含完整 completed 列表|
|Backup/Restore|FlowState Adapter 没有 save/restore 公共 API；反而拒绝 host 副本与 ongoing backup|BLOCKED：不能假定可从一个策略状态原地无损还原到另一策略初态|
|原生 COW|B1-C install_observation 包装原 match/copy 一次；源 slot 到独立工作 slot 的原生复制|VERIFIED：单案例请求匹配与 8 Token 输出；不泛化到全部边|

B1-C 报告保存的容器源码身份与路径：`/sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py:130–202`（状态验证与 COW 源）；`managers/schedule_batch.py:1310–1343,2640–2660`；`model_executor/model_runner.py:1506–1523`；`mem_cache/memory_pool.py:939–975`（conv 与 temporal 复制）。本轮未启动容器读取这些路径，以上为冻结报告及 source_identity 的历史证据，不宣称对另一安装版本重新验证。B1-C 的实际调用记录与输出比源码名称更强，但仍局限于该生命周期。

## 6. Runtime 正确性证据等级

|证据等级|现有支持|B4 严格 Policy 闭环的状态|
|---|---|---|
|1 EPOCH_ADMISSIBLE|B4-B1 四组 68 条、完整 224 配对；T2 双向前缀核对|VERIFIED，不自动升级|
|2 Actual Runtime State Matched|B1-C retained 请求直接命中 node20，源 slot6|PARTIAL：该证据不是 g001–g004 按策略协调后请求|
|3 RESUME_VALIDATED|B1-C B1→A 单边实际 COW 与驱逐反事实|PARTIAL：B4-B1 准入边没有逐边恢复认证|
|4 Policy-selected Set Physically Reconciled|旧 RQ4/RQ6/E2E 在旧选集合同下有协调证明|BLOCKED：未执行 B4-C 严格选集；不能迁移旧 PASS|
|5 Output Correctness Validated|B1-C retained/removed 与独立 cold reference 的 8 Token 完全一致|PARTIAL：缺 g001–g004 严格策略对应输出参考与核对|
|6 Request-level Performance Measured|旧 RQ4 TTFT、旧优化 E2E 请求时间|UNKNOWN：B4 严格策略没有该测量，CPU modeled ms 不能替代|
|7 Control-inclusive Net Benefit Measured|旧 probe-inclusive E2E 有测量，但对两基线 CI 均跨 0|UNKNOWN：严格选集未测；不支持稳定净收益|

B1-C 精确事实（`evaluation/strict_compatibility_b1c_output/b1c_g000_20261014_r2/final_report.md`，各 arm `evidence.json`）：目标 `RQ3_G000_B_TURN_001` → `RQ3_G000_A_PENDING_TURN_003`；retained H/E/G=4352/2432/1920，removed=4352/0/4352；FA 不变；实际 match node20，COW 6→19；两条件输出与 cold reference 8/8 一致。隔离先移除 A2/A1，移除臂再移除 B1。这是明确手工隔离，不是任一 B4-C 策略计算结果。

旧优化 E2E 来源 `evaluation/control_path_optimization_output/control_20260929/optimized_formal/analysis/final_report.md`：216/216 生命周期及 864/864 请求通过原门禁；FlowState epoch 2425.641 ms；对 LRU 改善均值526.612 ms、CI[-1057.440,2214.943]，对 Marconi 改善−549.649 ms、CI[-1784.913,699.396]。这是不同旧样本/兼容路径的 probe-inclusive 结果，不是本轮严格矩阵证据，也不是稳定正收益。

`evaluation/serving_path_e2e_output/serving_20260929/final_report.md` 为 BLOCKED_BY_CORRECTNESS_GATE，0/216 正式生命周期；缺完整被动变更事件，无法用末态代替逐请求状态验证。这个失败结果必须保留，不能把探针开销从旧 epoch 中扣掉后宣称 serving 净收益。

## 7. 严格 Runtime A/B 的可执行性

|问题|现有能力与输入|门禁|
|---|---|---|
|等价初态|独立 fresh Engine＋固定历史回放已有 runner；B1-C 用 semantic_nodes 比较两臂|PARTIAL：没有 g001–g004 多策略复现初态证据；不能复用同一已变更 Engine。重新定位 node/slot，比较 Token/位置/FA/组件语义而非要求跨生命周期数值 ID 相同|
|相同 C/P/准入/K|B4 冻结 Token、配置、矩阵与 B4-C 预算齐全|PARTIAL：缺把当前采集等价状态绑定这些输入的严格 Runtime 选择入口|
|选中/未选中实际驻留|批量 view、proof、逐节点 inspect 已有|PARTIAL：需变更前 keep/free 物理互斥、完整 census、基数不超过 K 的断言和严格选集一致性|
|恢复前 FA 与 Recurrent 前沿|无副作用 FA 查询、全树 Token/slot census；实际 match hook 可记录恢复|PARTIAL：必须在各请求前后区分初始模型 T/E 与新状态；不得用额外有副作用 lookup 改变缓存|
|Cold 输出参考|B1-C 独立 reference 方法与确定性 8 Token 门禁|PARTIAL：仅 g000/A 有该证据；本次四组全部 Pending 参考尚未闭合|
|分段计时|旧 E2E 同单调时钟记录 observation/allocation/reconciliation/submit/first/completion|PARTIAL：可复用方法但未接到严格路径；harness 检查必须保留或有等价证据，不能隐去|
|公平策略 A/B|各局部原语可用|BLOCKED：上述严格数据流、身份及等价初态门禁未闭合，不能直接进入正式比较|

理论上可在新生命周期按相同输入建立相同逻辑初态，但源码“可以调用”不等于已验证成立。本轮不执行该验证，也不预先给出 Runtime 选集成功或性能改善结论。

## 8. 最小必要修复与验证范围（仅建议，未执行）

按严重性排序，以下为解除阻断所必需，不涉及改变目标函数、Greedy 或预算。

1. **P0 / 严格 Compatibility 接线。** 位置：`flowstate/controller.py:55–73`、`flowstate/optimizer.py:137–145`、`flowstate/executable_state.py:17–20`；参考 B4-C `MatrixOptimizer`。建立明确消费不可变严格 epoch 关系的运行入口，携带 Pending Token/config/Barrier、候选清单、K 及证书摘要，拒绝 UNKNOWN。避免回退旧 Workflow 谓词；不伪造 Workflow Identity。必须证明 Runtime 调用使用该入口，而不是仅 CPU 子类存在。
2. **P0 / 变更前的物理集合合同。** 位置：`StateController._validate_handles`、`RuntimeCheckpointHandle`、`rq6_batched_control_transport._validate_identity_and_residency/_batch_reconciliation`。选中项也必须有有效映射；核对当前组件及全树 census，显式断言 keep/free 节点与 slot 互斥。对现有四快照可以坚持一对一并拒绝 alias，未必需要开发通用共享引用计数。若无法提供代际，明确只支持连续静止、view 未变、禁止并发 admission 的受控 epoch，而非宣称通用 ABA 安全。
3. **P0 / 严格选集与预算后置条件。** 在批量入口绑定 K 和选集，核对 |S|≤K、当前候选域完整、managed resident 恰为 S、未选中均不可恢复且 FA 不变；不能只检查 selected 子集合法。任何受保护未选中项无法释放必须 fail-stop，不通过删候选或增预算继续。保留现有完整验证，不以物理池有空闲为由跳过驱逐。
4. **P0 / 初态等价及实际恢复门禁。** 位置：独立严格 Runner 接线，复用 B4-B0 `capture` 与 B1-C `semantic_nodes/isolation_plan/install_observation` 的能力，不修改冻结产物。独立生命周期按原输入复现、重新定位并比较等价初态；在任何性能比较前验证严格选集、实际 match/COW、输出参考及请求间状态变更。当前没有这些四组策略臂证据，所以正式 A/B 尚不可直接开展。
5. **P1 / 失败账本与计时口径。** 批量协调异常应保留已完成驱逐和失败视图；上层终止并清理，不在部分变更后自动重试。复用旧 E2E 分段时钟但明确同步验证开销；不能把旧 serving 门禁缺口当作已解决。无需为本轮目标引入通用 restore/rollback，可优先采用失败即停和独立生命周期。

**最终回答：** 现有实现已具备“看见状态、按指定集合移除循环组件、让单个合法跨 Workflow 请求恢复”的局部能力；尚缺“严格关系实际进入生产选择、选集与当前物理身份/预算共同绑定、变更前保护检查、四组策略臂等价初态、协调后逐请求恢复及输出验证”这条完整证据链。故唯一门禁为 **BLOCKED（不可直接进入正式严格 Runtime Policy Comparison）**。无需据此推断正式 CPU 目标结果错误，也不能用已有旧 Runtime PASS 消除该阻断。

本轮完成并停止；不自动修复、不启动 Runtime A/B。
