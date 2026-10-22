# Phase R2-C0：真实 Runtime 桥接与只读 Preflight

## 最终门禁：READY

仅适用于本次 g001、Qwen3.5-9B、TP=1、28 槽、无异步缓存功能的受控已知输入回放。完成一个独立 Engine 生命周期及原有 12 条历史请求；4 条 Pending 仅使用冻结的已知 Token 输入检查身份和前沿，没有提交执行。没有调用实际驱逐、Backup/Restore、缓存清空、StrictReconciler 或任何策略性能 A/B。

**五个问题的答案：**

1. **当前 Scheduler 提供真实连续隔离：是，但限单次 Scheduler 同步控制调用。** 不是跨多个 RPC 的租约，也不是停止前端网络接收。
2. **PhysicalView 完整、准确、无副作用：通过本轮 R2-B 合同。** 读取全部 14 个非根节点，其中 12 个 Recurrent 候选、2 个仅 FA 节点；根节点纳入全树摘要。逐槽独立组件切片和三次观测一致性通过。
3. **g001/K2 真实 Binding/Preflight：通过。** B1/B3 绑定当前 node 20/29、slot 6/22；其余 10 个为逻辑补集。没有释放这些状态，最终物理驻留仍为 12 项。
4. **仍有限定：** Slot Generation 未暴露；失败停驻分支有源码依据与 CPU 负例，未故意在 GPU 注入故障；未做逐字节 Recurrent 数值或全部 FA payload 哈希；未证明通用并发服务器或跨 RPC 隔离。上述能力不在本轮 READY 的范围内。
5. **可以进入 R2-C1 正确性验证准备：是。** 必须在新生命周期的同一同步调用内重新观察、绑定、Preflight，再执行后续协调；本次已关闭的窗口和绑定不得复用。本文不授权也未自动执行 R2-C1。

## 1. 环境、输入与执行身份

- 模型 `/home/wjg/models/qwen3.5-9b`；GPU0 NVIDIA H100 PCIe；TP=1。
- 镜像 `lmsysorg/sglang:v0.5.17-cu129-runtime`，ID `sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926`。
- SGLang 0.5.17、Transformers 5.12.1；关键原生源码与 B1-C 保存的 SHA-256 全部一致。
- 参数与冻结 g001 配置逐项一致，且在当前 Scheduler 内再次核对实际 server_args；含 extra_buffer、track_interval=256、max_states_per_path=-1、disable_overlap_schedule=True、disable_cuda_graph=True、mem_fraction_static=0.4、28 槽。
- `source_identity.json` 核查 244 项原来源文件。Optimizer 与 Controller 明确采用 R2-A/R2-B 已授权的当前摘要，其余沿用原锁，不篡改历史锁。当前新桥接及核心由 `implementation_lock.json` 另锁。
- 当前 Scheduler 源码哈希 `1d4a7683e32f0862cf5b03be33e5ec23f9d112e59546d391e28fa222fcd1619b`；原生源码只读归档在 `runtime_sources.json`，以原始绝对路径为键。
- `prepared_inputs.json` 直接保存冻结的 12 个历史请求与 4 个 Pending Token 数组，没有文本重建。`history_identity.json` 保存历史请求实际 match 所见输入摘要与数量，12/12 一致。
- Pending 未在本轮重新提交；其实际 Runtime 接收身份沿用 B4-B1 已冻结证据。当前已知输入不是自然在线 Agent Ready，不引入未来请求输出。
- 冻结 Snapshot SHA、当前物理观测、当前控制窗口、严格 epoch/plan digest 分别记录，未将旧 SHA 当成 GPU 状态证明。

## 2. Scheduler 连续隔离证据

原生 `/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py`：

- `event_loop_normal:1682–1713`：在 1689–1690 接收/处理请求，1695 选择 batch；没有 batch 时 1708 同步调用 `on_idle`。在调用返回前不会再次进入 recv/process/run_batch。
- `process_input_requests:1838–1866`：请求分派、会话回收与外部语料加载位于同一循环的此阶段；不在只读控制回调中执行。
- `is_fully_idle:3967–4020`：检查 running/waiting、chunked、last_batch、grammar、各分离式/异步队列。
- `on_idle:3925–3965`：原生 idle housekeeping 在本次回调返回后才执行。

实际调用链：

`event_loop_normal → patched_on_idle → ProbeState.drain_one → r2c0_readonly → scope → CUDA synchronize → observe → 当前语义/边核对 → bind → observe → preflight → observe → 一致性核对 → 返回`。

入口复用 `tests/runtime/wp3b_end_to_end_transport.py:172–176`；队列由 `motivation/artifacts/wp3b_gate_20260820/targeted_probe.py:498–524` 在 Scheduler 线程同步消费。控制服务器线程只排队和等待结果，不读写缓存。全部核心操作在同一线程 125863452914432 内完成，起止事件见 `preflight_results.json.events`。

`bridge.py:41` 的 scope 检查实际 idle/队列、无进行中 insert/eviction/transfer，并禁止 HiCache、session cache、HiSparse、unified memory、external corpus、disaggregation、额外 ReplaySSM 与 overlap。`bridge.py:107` 同步完成在途 CUDA 工作，之后只读访问状态。当前 running=0、waiting=0、chunked=False。

因此 `admission_closed=True` 表达“单次同步调用期间 Scheduler 不执行新的 Admission”，不是设置一个布尔量来关闭调度。TCP/IPC 前端仍可能排队；本轮独占 Engine 只有当前调用方，未注入额外请求测试通用服务入口。单线程同步执行及禁用异步写者共同构成隔离条件，批处理本身不是原子性证明。

异常路径：`bridge.py:141–146` 捕获失败、记录账本并置生命周期失败；队列先返回错误结果，随后 `guarded_idle:155–159` 停驻，不进入下一轮请求接收，直到父进程清理。没有吞异常后继续服务或自动重试。实际 GPU 正例没有触发该分支；不得把 CPU 负例称为真实 GPU 故障注入验证。

本轮成功返回后窗口结束。**不提供跨 RPC 的持续保护，也不允许在未来 RPC 中复用此绑定。**

## 3. PhysicalView 与组件身份

实现：`bridge.py:60 observe`，复用 `strict_compatibility_b1c/runner.py:75 state_snapshot` 与 `barrier_fa_frontier_control.py:152 semantic_cache_snapshot`。直接遍历当前 `_node_arena` 和父链 Token Key，不调用 cache.match_prefix。历史回放本身的原生 match 仅调用一次并被动记录。

|字段|实际来源与验证|
|---|---|
|完整非根节点|当前 arena 中 14 个节点；根 node 9 纳入全树摘要|
|Token Prefix/位置|当前 Radix Key 父链拼接的完整 Token 数组，逐项与冻结候选精确相等|
|Node/Slot|当前节点 ID 与 MAMBA.value；不以历史编号查找|
|Conv/Temporal|当前 MambaPool.mamba_cache 的 Conv 与 Temporal 张量|
|组件切片|`slice_identity:27` 读取 storage 地址、storage_offset、shape、stride、dtype、device，生成每层每槽的实际字节区间；24 个独立组件身份，区间无重叠|
|槽位记账|容量28；驻留12；空闲16；两集合互斥且完整覆盖1..28；sentinel0不计入可分配容量|
|保护/引用|FULL/MAMBA lock_ref、host_lock_ref、session_ref 全部读取；候选全为0，无host副本；无进行中驱逐/传输|
|FA与非目标摘要|全树 FA 索引/结构摘要、FA allocator available、FA保护与可驱逐计数、引用、recency、LRU、叶集合；只读期间一致|
|额外可恢复状态|候选域外 resident Recurrent 节点为0；node19/22仅有FA，无Recurrent slots|

当前 Conv 布局为 `[24,29,8192,3]` bfloat16；Temporal 为 `[24,29,32,128,128]` float32。组件身份不是整池指针；同一 Tensor 的不同 Slot 在每一层对应不相交区间。观察不写入张量，未逐字节复制/哈希所有模型状态；本轮“不变”指上述完整合同视图、索引和安全元数据一致，加上只读调用路径，而非新输出正确性认证。

`FA allocated_count` 原生接口不可用，仍记为 null；R2-B 所需的 Recurrent resident/free accounting 已完整核对。未捏造 Slot Generation、通用跨进程 ownership version 或 TP 多rank证据。

## 4. 当前物化状态与冻结语义对照

|Checkpoint|当前Node|当前Slot|逻辑计划角色|
|---|---:|---:|---|
|RQ3_G001_A_TURN_001|18|3|逻辑驱逐补集（未执行）|
|RQ3_G001_A_TURN_002|24|12|逻辑驱逐补集（未执行）|
|RQ3_G001_A_TURN_003|28|20|逻辑驱逐补集（未执行）|
|RQ3_G001_B_TURN_001|20|6|保留计划|
|RQ3_G001_B_TURN_002|25|14|逻辑驱逐补集（未执行）|
|RQ3_G001_B_TURN_003|29|22|保留计划|
|RQ3_G001_C_TURN_001|21|8|逻辑驱逐补集（未执行）|
|RQ3_G001_C_TURN_002|26|16|逻辑驱逐补集（未执行）|
|RQ3_G001_C_TURN_003|30|24|逻辑驱逐补集（未执行）|
|RQ3_G001_D_TURN_001|23|10|逻辑驱逐补集（未执行）|
|RQ3_G001_D_TURN_002|27|18|逻辑驱逐补集（未执行）|
|RQ3_G001_D_TURN_003|31|26|逻辑驱逐补集（未执行）|

这些编号恰好与历史记录相同，但绑定使用当前完整 Token 查找与当前组件读取；编号相同不是认证依据。

|Pending|Planning Target|当前FA Frontier|
|---|---:|---:|
|RQ3_G001_A_PENDING_TURN_004|4224|4224|
|RQ3_G001_B_PENDING_TURN_004|5312|5312|
|RQ3_G001_C_PENDING_TURN_004|4736|4736|
|RQ3_G001_D_PENDING_TURN_004|4224|4224|

12 个候选的位置与 Token、4 个 Pending 的 T/FA，以及所有 48 对的准入状态和逐项条件与冻结合同一致：EPOCH_ADMISSIBLE=15、REJECTED=33、UNKNOWN=0，跨 Workflow 准入边=3。没有按 Workflow-ID 重新过滤，也未将准入提升为 RESUME_VALIDATED。

## 5. 真实 Binding 与 Preflight

使用 `StrictAllocationEpoch.from_snapshot_bytes → GlobalOptimizer.plan_strict` 产生 g001/K2 计划。只调用生产 `flowstate/strict_control.py` 的 `bind` 和 `preflight`，没有调用会驱逐的 `StateController.reconcile_strict`。

- B1/B3 均有当前精确物理映射，选集2项、补集10项。
- 全部 Node/Slot/组件切片唯一；Keep/Evict 无交集。
- 不存在候选外 Recurrent 状态；所有候选具备必要组件，无引用/host保护冲突。
- 初始、Preflight、结束三次 PhysicalView 完全一致；完整 semantic snapshot、组件切片身份也一致。
- FA索引、LRU、访问计数、引用和free-slot记账均未被观察修改。
- **物理池仍为12个驻留候选，而非K=2。** 本轮没有实施保留配额，不声称逻辑预算已经造成物理压力，也没有给Selected加通用持久pin。

## 6. CPU 负面门禁

`cpu_tests_final.log`：59 passed、23 subtests passed。包括 R2-A/R2-B 回归及本轮 5 项测试；7个视图变体逐项拒绝。覆盖 Admission未关闭、控制身份失效、view变化、selected缺映射/候选域不完整、组件身份缺失、Keep/Evict slot别名、槽位记账缺失、配置与Epoch/Plan摘要错误、真实字节区间重叠。模拟操作列表保持为空。

初次测试因独立目录相对导入失败，保留在 `cpu_tests.log`；修正测试导入后上述最终结果通过。没有为通过测试修改生产合同或冻结输入。CPU测试不是59次GPU实验。

## 7. 清理、异常和完整性

`engine_cleanup.json` 确认 shutdown 返回；`process_result.json` 返回码0。`gpu_cleanup.json` 确认本轮容器不存在、GPU0恢复14 MiB、计算进程列表为空。未操作其他容器或GPU进程。

启动日志包含未使用的音视频依赖加载告警；关闭时有 CUDA IPC producer 终止告警。完整保存于 `runtime.log`，未省略。没有 OOM、retraction 或准入失败，实际清理门禁通过。`--rm` 已移除容器，因此后续定向清理提示“无此容器”，以最终容器查询和GPU结果核实，而不只凭清理命令返回码。

执行前后 244 项来源摘要一致；B4-B1/B4-C Manifest 与本轮实施锁复核通过。没有覆盖历史锁或修改生产源码；新增文件仅在本目录。见 `integrity_summary.json` 与 `evidence_manifest.json`。

## 8. R2-C1 的边界与剩余验证

已补齐真实 Scheduler 单次连续窗口、只读全树 PhysicalView、组件切片身份、当前语义等价及生产 bind/preflight 证据。可以准备一次限定g001的真实 Recurrent-only Eviction正确性验证。

仍须在R2-C1另行验证：实际释放恰为逻辑补集、Selected与FA/非目标状态不变、真实恢复前沿、输出对照、异常停止与清理。未来协调必须直接放入同一Scheduler同步控制区间，不能拆成回到主循环后的多个RPC。本轮没有提供生产系统通用事务、回滚、跨rank原子性或持续Generation保护，也不支持Runtime策略性能和净收益主张。

**本轮到此停止，没有执行R2-C1。**
