# 控制关键路径逐类分析（七类）

统一栏目：**代码位置 / 数据来源 / 是否需 GPU 同步 / 是否 Scheduler 关键路径 / 是否生产必需安全检查 / 是否有独立耗时证据 / 可否移出控制关键路径 / 移出后如何保证 Epoch 新鲜度**。

术语约定：本文"Scheduler 关键路径"指 Scheduler 线程在静止控制窗口内串行执行、阻塞后续调度的区间（`ProbeState.drain_one` 在 `on_idle` 中执行，targeted_probe.py:498-522；D2 已证明 CPU 规划可移出该窗口）。耗时证据均引自 `observation_cost_breakdown.md` §1 所列原始产物，均为单次实测、非 P95。

---

## 一、Scheduler 安全一致性必需检查

- **代码位置**：`runtime_control_closure_r2c0/bridge.py:41-58`（scope：owner 线程、无 HiCache/HiSparse/分离式传输/额外 ReplaySSM、ongoing_* 传输标志、驱逐进行中标志）；`motivation/artifacts/wp3b_gate_20260820/targeted_probe.py:277-311`（`_validate_runtime_scope`：fully_idle、无 ongoing insert、tp/pp/dp=1、overlap 关闭）；`flowstate/strict_control.py:112-138`（`validate_view`：idle/admission_closed/transfers_pending、节点/槽/组件唯一性、槽位记账闭合）；`flowstate/adapters/sglang.py:123-289`（`validate_runtime_capabilities`+`_validate_target`+`_validate_runtime_scope`+`_validate_postconditions`）。
- **数据来源**：Scheduler/cache 内存属性与计数器（LRU 成员、`_node_arena`、allocator 计数、lock_ref/host_lock_ref/session_ref）。
- **GPU 同步**：否，纯 CPU 属性读取。
- **Scheduler 关键路径**：是（必须与决策同窗口，语义即"连续静止控制区间"）。
- **生产必需安全检查**：**是**。删除即失去"观察时点无并发变更"前提，严格 Epoch 与绑定合同失效；B5-D3 final_report 亦将其列为"可能进入真实生产关键路径"。
- **独立耗时证据**：无单独计量（分散于 heavy_observation_gpu_sync / apply_time_freshness_preflight / strict_reconciliation_with_postchecks / adapter_eviction_i 等段内）。Adapter 逐项驱逐含此类检查整体实测 1.0–1.4 ms/次（D3 CSV），可作为"此类检查+单节点驱逐"的量级上界参照。
- **可否移出**：**否**（语义不可移出）。可做的只是消除重复：同一窗口内 `scope` 被反复执行（`observe_control` 入口、每次 `observe` 内 `state_snapshot`、`reconcile_strict` 的 held/observe 包装各 1 次；见 observation_cost_breakdown.md §2 计数），可在窗口内做一次性检查+窗口令牌复用。
- **移出后新鲜度**：不适用（不移出）。窗口令牌（`_r2c0_owner` 线程身份）+ owner 校验即现有新鲜度机制的一部分，保持不变。

## 二、Runtime 原生元数据读取

- **代码位置**：全树 census 循环 `evaluation/strict_compatibility_b1c/runner.py:85-98`（父链 Token 重建、`token_digest`、mamba_slots、引用计数）；`barrier_fa_frontier_control.py:152-208`（recency/reference 行、LRU 真实顺序 `lru_node_order`、evictable 叶集合）；`targeted_probe.py:237-256`（`_accounting_snapshot`：allocator 计数、free_slots）；`barrier_fa_frontier_control.py:26-93`（逐 Pending FA 前沿游走）。
- **数据来源**：基数树 arena、节点 component_data、LRU 链表、池分配器计数；`tensor_ids` 对槽位索引做小型 D2H。
- **GPU 同步**：不需要设备同步（槽位索引 `.tolist()` 的小型搬运除外，随步骤一确认的静止点已安全）。
- **Scheduler 关键路径**：当前在（作为观察的一部分）；**本质上是决策输入**（候选集、L、T、引用状态），轻量化后仍在窗口内读取，但可由增量维护的视图替代全量 census（见 minimal_state_view_contract.md）。
- **生产必需安全检查**：决策必需（非"安全检查"类）：无此数据无法构造候选/Pending/严格边。
- **独立耗时证据**：无单独计量（混入观察 RPC 1102.325/1203.035 ms 内）。
- **可否移出**：**部分可移出**。逐节点父链 Token 重建+全量 `token_digest` 可改为变更点一次计算并登记（插入/分裂时）；决策时只读登记结果。FA 前沿游走保留为决策时定点读取（每个 Pending 一次，无内容哈希）。
- **移出后新鲜度**：登记视图带 `mutation_generation`（任何插入/分裂/驱逐/COW bump）；决策时视图快照 = (登记表, generation)；apply 时 generation 不变 + 定点重验通过才执行（详见合同文档 §6）。

## 三、GPU 同步与设备内容访问

- **代码位置**：显式 `torch.cuda.synchronize()`（`multishare_runtime_gate_b5c3/bridge.py:39`）；全树 FA 内容哈希 `_tensor_sha256`（targeted_probe.py:46-53, 121）；逐槽 Recurrent 内容哈希 `payload`（b5c3/bridge.py:15-23）；FA K/V 内容审计 `fa_payload`（D3/D4a control.py:38-53，`index_select`+`.cpu()`）；COW 源/目的槽内容哈希（`exact_greedy_paired_gpu_fix_b5d4a/bridge.py:62-75`）。
- **数据来源**：GPU 张量本体（FA value、conv/temporal 槽切片、FA 池 K/V 行）。
- **GPU 同步**：**是**（本类即全部设备同步与 D2H 大流量来源）。
- **Scheduler 关键路径**：当前在（观察/审计都在窗口内执行）。
- **生产必需安全检查**：**分层**——(a) 一次设备同步以确认静止点：必需（且属一次性，量级未单独计量）；(b) 内容级哈希（全树 FA、逐槽 Recurrent、FA K/V 审计）：**实验审计**，验证"只读观察未改变内容/驱逐只改目标"，不是决策输入（决策只用元数据+驻留标志，见 `build_edges` 输入与 `StrictAllocationEpoch.validate` 条件）；D3 将 fa_content_* 两段明确归类"重型实验审计"（153.654/288.041 ms）。
- **独立耗时证据**：整段有（heavy_observation_gpu_sync 1203.035、fa_content_audit_before 153.654、fa_content_postcheck 288.041，均 D3）；子步无。
- **可否移出**：**(b) 全部可移出**——移出后决策输入不减少；(a) 保留一次。移出方式：轻量视图不读张量内容；"内容未变"保证改由"窗口内无并发变更 + 逐项驱逐的精确释放范围检查（Adapter `_evict_mamba_component_only` 的 tracker/free 断言 + `sanity_check`，sglang.py:416-469）"承担，该路径已实测 1.0–1.4 ms/项。
- **移出后新鲜度**：结构/账号新鲜度由 generation+apply 时定点重验保证；内容正确性由逐项驱逐的后置条件（结构、槽释放、FA 不变）与请求恢复正确性门禁（B5-D3 已验证）承担，而非每轮全量内容对拍。重型内容审计保留为可切换的审计模式（见 candidates 文档）。

## 四、全树遍历及组件哈希

- **代码位置**：`_global_maps`（targeted_probe.py:106-140，structure/full/mamba 三表+三个 SHA256）；`state_snapshot` 内两次调用（b1c/runner.py:82-83, 102-103）；乘数结构见 observation_cost_breakdown.md §2（一次 final 观察=4 次全树哈希；D3 reconcile 全程 ≈13 次 observe≈26 次全树哈希）。
- **数据来源**：全部树节点的 FA value 张量（内容）、结构行、mamba 槽行。
- **GPU 同步**：是（FA value 内容哈希隐式 D2H）。
- **Scheduler 关键路径**：当前在。
- **生产必需安全检查**：结构表（node/parent/children/key 长度）与 mamba 槽表是必要决策输入（候选定位、记账闭合）；**FA 内容哈希（full_rows 的 `_tensor_sha256`）属审计**；"前后两次全树快照相等"的非干涉证明属审计（其语义已被"静止窗口+无并发变更检查"覆盖）。
- **独立耗时证据**：无单独计量（嵌于观察 RPC 与 preflight/reconciliation 各段）。
- **可否移出**：**FA 内容哈希与重复遍历可移出**；结构事实改由增量登记维护（插入/分裂/驱逐时更新），决策时 O(N) 读取登记表并做结构摘要（纯 CPU、无张量访问）。当前候选构建还存在 O(N²) 前缀包含比较（b5c3/bridge.py:53 `path_full_all_present` 对每候选全表扫描），登记视图可将其变为沿父链 O(depth) 判定。
- **移出后新鲜度**：结构变更（分裂/合并/驱逐）即 bump generation；apply 时对**选中候选**定点重验 node_id+prefix_digest+驻留+引用（`_validate_target` 既有能力，sglang.py:332-414），不再全树对拍。

## 五、CPU 序列化、传输与解码

- **代码位置**：Scheduler 侧 `json.dumps(response, sort_keys=True)` + 逐行 TCP（targeted_probe.py:543-562）；客户端 `json.loads`（targeted_probe.py:574-586）；客户端侧 `encode()` 对整份物理观察与 strict_epoch 再序列化+SHA256（D3/D4a control.py:59, 64, 67；shadow.py:11, 18）。
- **数据来源**：观察结果巨型字典（含全部 census、逐槽哈希、全量 token；D3 落盘 1.1–1.6 MB/份为量级参照）。
- **GPU 同步**：否。
- **Scheduler 关键路径**：序列化在 Scheduler 进程内（阻塞窗口）；传输/解码在客户端。D2 的 1102.325 ms 为全链路单值，内部占比未计量。
- **生产必需安全检查**：传输本身不是安全检查；体积是"把审计证据随观察一并返回"的后果。
- **独立耗时证据**：无单独计量。
- **可否移出**：**可**。轻量决策视图只需 KB 级字段（见合同文档 §字段表），审计证据改由审计模式单独拉取或落盘，不随每次决策传输。客户端侧对整份观察的二次序列化+哈希也随视图缩小而消失（仅对 strict_epoch 小对象做摘要）。
- **移出后新鲜度**：视图携带 epoch_id+generation+configuration_digest；缺失或失配即拒绝。传输层失败语义不变（现有 `ProbeState` 超时/nonce 机制）。

## 六、原始证据写盘与独立实验 Oracle

- **代码位置**：D4a `Journal` 逐阶段 record（`exact_greedy_paired_gpu_fix_b5d4a/control.py:59, 75, 85, 102, 124, 141`，D4b protocol 已声明逐阶段 fsync 属实验审计成本）；census/快照写盘（D2 runner.py `write(...)` 系列）；独立 Decimal Oracle（shadow.py:26-39；D3 control.py:73-85）；逐候选来源检索（origins：逐候选×逐历史文件读取比对，D3 control.py:61-66）。
- **数据来源**：全部为 CPU/磁盘。
- **GPU 同步**：否。
- **Scheduler 关键路径**：Journal 写盘在窗口内（D3/D4a）；Oracle 与 origins 在窗口内 CPU 段（D2 路径的 oracle 在客户端窗口外）。D4b protocol 明确 D4a 逐阶段 fsync 是审计成本。
- **生产必需安全检查**：**否**，全部为实验审计（D3 category 列自证：independent_oracle 0.257 属重型实验审计；epoch 认证段 347.979 中混有的 Journal 写盘亦是）。
- **独立耗时证据**：oracle 0.257/0.404 ms 有独立值；Journal/fsync、origins 检索无单独计量（混入 strict_epoch_authentication 347.979 ms）。
- **可否移出**：**全部可移出**（审计开关化：需要出报告时启用，逐决策路径不执行）。
- **移出后新鲜度**：无影响（不参与新鲜度语义）。

## 七、Plan 验证及重复身份认证

- **代码位置**：`validate_plan`（`flowstate/strict_control.py:94-109`，内含 `epoch.validate()` 全量重验+内容摘要重算+前沿/成本重算）；`StrictAllocationEpoch.validate`（`flowstate/strict_epoch.py:98-146`）；`from_frozen`（strict_epoch.py:148-173）；`bind`/`preflight` 的重复全量验证（strict_control.py:141-163：preflight 内部再 bind，即再 validate_plan+validate_view）；`_find_exact_node`/`_validate_target` 的定点重验（sglang.py:291-414）。
- **数据来源**：CPU 对象（不可变 Epoch/Plan/View）。
- **GPU 同步**：否。
- **Scheduler 关键路径**：窗口内执行（apply 侧）；D2 中 plan_construction_validation 13.553/13.747 ms 在窗口外 CPU 段。
- **生产必需安全检查**：**是**（身份/预算/补集/前沿一致性是严格合同本体；apply 时定点物理重验不可删除）。但**重复次数可精简**：同一不可变 Epoch 在一次控制流程中被 `validate()` 全量重验多次（from_frozen 1 次、每次 validate_plan 各 1 次、preflight 再触发）；不可变对象+未变更 generation 下结果必然一致。
- **独立耗时证据**：plan_construction_validation 13.553/13.747 ms（D2）；exact_solve_plan_validation 8.875 ms（D3）。内部各次 validate 无独立计量。
- **可否移出**：验证本体不可移出；**重复验证可合并**（窗口内缓存"Epoch 已验+digest"结果，generation/身份不变则复用；apply 时保留一次定点物理重验+一次最终 postcheck）。
- **移出后新鲜度**：不减弱——复用前提恰是"不可变 Epoch 身份一致 + 视图 generation 未变"；任一变化即放弃复用、全量重验、失败即隔离（沿用现有 fail-closed 语义）。

---

## 汇总表

| 类 | GPU 同步 | Scheduler 关键路径 | 生产必需 | 独立耗时证据 | 可移出/可精简 | 新鲜度替代机制 |
|---|---|---|---|---|---|---|
| 一 安全一致性检查 | 否 | 是 | 是 | 无（段内混合；Adapter 整体 1.0–1.4 ms/项） | 仅消除窗口内重复 | 不适用（保持） |
| 二 原生元数据读取 | 小型 D2H | 是（决策输入） | 决策必需 | 无 | 增量登记替代全量 census | generation+定点重验 |
| 三 GPU 同步与内容访问 | **是** | 是 | 同步一次必需；内容哈希非必需 | 段值有，子步无 | 内容哈希全移出、同步留一次 | 窗口静止+逐项驱逐后置条件 |
| 四 全树遍历及组件哈希 | 是(FA 内容) | 是 | 结构事实必需；内容哈希非必需 | 无 | 内容哈希移出；遍历增量化 | 结构变更 bump generation |
| 五 序列化传输解码 | 否 | Scheduler 侧序列化在 | 否 | 无 | 视图缩至 KB 级；审计另行拉取 | epoch_id+digest 随视图传递 |
| 六 写盘与 Oracle | 否 | 部分（Journal） | 否 | oracle 有；fsync 无 | 全部审计开关化 | 无影响 |
| 七 Plan 验证与重复认证 | 否 | 是 | 是 | 有（13.5/13.7/8.9 ms 段值） | 合并重复验证 | 不可变身份+generation 复用条件 |
