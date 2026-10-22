# 最小在线状态视图合同（Minimal State View Contract）

## 0. 语义红线（先于一切设计）

FlowState **观察 FA-KV 与 Recurrent 两类状态，但只控制 Recurrent State**。本合同给出的最小视图不得为压缩成本而删除对 FA 状态的观察（逐 Pending 的驻留 FA 前沿仍是规划目标 T 的来源）；也不得把 FA 状态纳入任何控制/驱逐动作（控制动作仍然只有 Recurrent-only eviction 一类）。现有严格兼容、安全检查与状态新鲜度要求一律保留，允许改变的只是其**实现载体**（内容对拍 → 身份/代际/定点重验）。

## 1. 字段清单与维护责任划分

维护责任三类：**[R] runtime 直接维护**（在状态变更点增量更新，决策时零计算读取）；**[D] 决策边界读取**（决策时只读、定点、少量计算，不读张量内容）；**[A] 审计数据**（不出现在每轮决策路径，仅审计模式生成）。

| # | 字段 | 内容 | 责任 | 现有来源（代码位置） |
|---|---|---|---|---|
| 1 | Checkpoint 逻辑身份 | checkpoint_id ↔ (token_ids 摘要, L)；插入/分裂时登记一次，digest 只算一次 | [R] | 父链重建+`token_digest`（b1c/runner.py:85-98）；`StrictCandidate.token_digest`（strict_epoch.py:30-37） |
| 2 | Position 与 Recurrent 组件可用性 | resident 标志（slots 非空）、conv/temporal 完整性、节点在 Mamba LRU | [R]（evict/COW/load-back 变更点更新） | `PhysicalNode.conv/temporal_complete`（strict_control.py:25-36；r2c0/bridge.py:83） |
| 3 | Node-Slot 及组件映射 | node_id → slots、组件切片身份（存储字节区间摘要） | [R] 登记映射；[D] 仅对**选中候选**在 apply 时计算切片身份 | `slice_identity`（r2c0/bridge.py:27-35）；`state_identity.target_mamba_slots`（b5c3/bridge.py:53） |
| 4 | FA Frontier 与 Planning Target | 每 Pending：resident_fa_frontier；T=min(len(token), frontier) | [D]（radix 游走，无内容哈希，每 Pending 一次） | `inspect_resident_fa_frontier`（barrier_fa_frontier_control.py:26-93）；T 推导（b5c3/bridge.py:55-57） |
| 5 | Residency 引用与别名所有权 | 每节点 lock_ref/host_lock_ref/session_ref、protected、host_copy；槽/节点/组件唯一性 | [R]（SGLang 已维护计数器，读取即可）+ [D] 记账闭合校验 | `validate_view`（strict_control.py:112-138）；`_validate_target` 引用断言（sglang.py:399-412） |
| 6 | Epoch 身份与新鲜度认证 | epoch_id、configuration_digest、`mutation_generation`（单调计数）、观察时间戳 | [R] generation（任何树/槽/引用/FA 驻留变更 bump）；[D] 组装与比对 | 现窗口身份（b5c3/bridge.py:31 `window=f'b5c3-{pid}-{monotonic_ns}'`）；generation 为新增载体，替代"前后双观察相等"证明 |
| 7 | Strict Compatibility 原始事实 | 14 条条件（strict_epoch.py:183 `_CONDITIONS`）逐边三态 | [D] 由字段 1/2/4/5 + Pending 输入**推导**（exact_token_prefix=L_le_T+token 身份；residency 类=字段 2/5；physical_identity=apply 时定点核验） | `build_edges`（strict_compatibility_b4b0/contracts.py）+ `StrictAllocationEpoch.validate`（strict_epoch.py:131-144） |
| 8 | Apply-time 物理重验 | 选中候选的 node_id、prefix_digest、驻留、引用清零、路径 FA 完整、LRU 成员 | [D]（每选中项一次，复用既有 Adapter 重验） | `_find_exact_node`+`_validate_target`（sglang.py:291-414）；handle `expected_node_id/expected_prefix_digest`（sglang.py:18-52） |

**[A] 审计数据（不进入每轮决策路径）**：全树 FA 内容哈希、逐槽 Recurrent 内容哈希、FA K/V 内容对拍、语义快照前后相等证明、census 巨型 JSON、Journal 逐阶段写盘、独立 Oracle、逐候选历史来源检索（origins）。审计模式（可切换）复用现有重型路径原样生成。

## 2. 视图读取接口（决策时序）

1. Scheduler 静止窗口内（scope 检查一次，窗口令牌复用）；
2. 读登记表快照（字段 1/2/3/5）+ generation + epoch 身份（字段 6）——纯内存 O(N)，无张量访问；
3. 逐 Pending FA 前沿游走（字段 4）；
4. 推导严格边（字段 7），构造与现格式兼容的 strict_epoch 输入；
5. 窗口结束；CPU 规划（窗口外，D2 已证可行）；
6. apply 时：generation 比对 + 逐选中项定点重验（字段 8）+ bind/preflight（小视图上执行，语义同 strict_control.py:141-163）。

## 3. 并发修改、Eviction、COW、Radix 变化的处理

- **串行性来源**：登记更新与观察读取都发生在 Scheduler 线程（变更点即原语执行处；观察即 `on_idle` 排空的控制调用）——单线程内无并发写；跨线程 TCP 仅投递命令（targeted_probe.py:531-566），维持现状。
- **变更点清单（generation 必须 bump 的全部入口）**：基数树插入/分段/合并（match_prefix 内）、`evict_component`（含 cascade 与 `_update_evictable_leaf_sets`）、`copy_from`（COW 工作槽）、`_free_values`（槽释放）、load_back/write_through/prefetch/backup（scope 已禁用，仍列为 bump 点以防御配置漂移）、FA 组件驱逐（影响字段 4 的前沿）。挂接位置与现有 instrumentation 同点（b1c/runner.py:28-73；b5c3/bridge.py:93-105；d4a/bridge.py:77-105），不改 SGLang 磁盘源码的 MonkeyPatch 方式沿用。
- **Eviction**：目标节点 slots 清空、free_slots 变化、LRU 摘除——登记表同步更新；若发生在计划与 apply 之间，generation 变化使旧计划拒绑（fail-closed）。
- **COW**：COW 目的槽属于活动请求（引用非零），不进入候选域；候选槽引用必须为零（字段 5 断言，沿用 `references==0` 合同）。COW 完成后目的槽内容变化不 bump 候选身份（候选只认源槽身份），但活动请求占用的槽记入非候选占用。
- **Radix 变化（分裂/合并）**：节点 id 与 token 重分段即逻辑身份变化——分裂时对新节点登记新身份、旧 digest 失效；任何"登记表节点在树中已不存在/前缀不符"在定点重验时发现即 fail-closed。
- **防漏更新**：未登记的变更路径会导致 generation 不 bump 而登记表漂移。两道防线：apply 时对选中项的定点物理重验（必经路径，漂移即失败）；周期性（非逐决策）全量 census 对账（审计模式复用），漂移即告警并冻结控制。

## 4. 观察-应用防错误绑定

沿用并强化现有三重身份（`PhysicalBinding`：epoch_id+epoch_digest+plan_digest+view_digest，strict_control.py:64-75）：

1. 计划绑定生成时的 (epoch_id, plan_digest, view_digest, generation) 四元组；
2. apply 时重读视图，`bind` 重算结果必须与绑定**逐字段相等**（strict_control.py:157 现语义保留），且 generation 未变；
3. 每个驱逐动作经 `RuntimeCheckpointHandle(expected_node_id, expected_prefix_digest)` 定点重验（strict_control.py:214-215；sglang.py:359-382）。
任何一环失败：不执行动作、生命周期隔离（`_r2c0_failed` 停机语义保持，r2c0/bridge.py:155-161），不复位重试。

## 5. 与现有重型路径的关系

重型路径不被删除，降级为**审计模式**：同一负载上两路径必须给出相同的候选集、严格边与 T（对拍门禁，见 implementation_and_validation_plan.md R3-B）；决策模式只走最小视图。任何"轻量视图与重型 census 不一致"即 FAIL 并阻止决策路径上线。此关系保证轻量化不改变决策语义，只改变证据生成时机。
