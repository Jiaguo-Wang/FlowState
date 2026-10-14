# P3-G1 物理 Retention Budget 合同审查

本轮只读审查与 CPU 整数容量推导，未启动 GPU、Engine、Agent、Pilot、replay 或策略比较。日期：2026-10-09。

源码约定：`S/` 为 `/tmp/audit_sglang/sglang-v0517-tag/python/sglang/`，使用 P3-G0 与停止的正式实验容器逐字节核实的版本；本轮重新核验其 16 个文件摘要。其余相对路径以 `/home/wjg/code/FlowState/` 为根。新扩展仅为设计，没有实施；UNKNOWN 不当作零。

**RESOURCE_CONTRACT_NEEDS_EXTENSION**

现有普通 recurrent pool 同时存放 checkpoint、working 与 tracking/COW 状态，没有独立的 checkpoint retention quota。可以复用 recurrent-only eviction 实现“固定物理池 + 可撤销空闲容量借用 + 跨 epoch 保留配额”的合同，但还缺全入口 admission、所有权/代际账本及 reservation 兑现接口。旧逻辑 K 不应被改称已由物理池容量强制约束。

## 核查结论与证据

| 问题 | 结论 | 证据 |
|---|---|---|
| 总容量如何配置 | max_mamba_cache_size=N 个可用槽，另有 slot0；非 K | S/srt/mem_cache/kv_cache_configurator.py:1874,1976；memory_pool.py:1191；allocator/mamba.py:93 |
| 完整 checkpoint 大小 | Qwen TP=1 普通模式旧记录为49.125MiB，含全部 conv+temporal | `/home/wjg/code/newfind/artifacts/exp2-b-PASS-20260812T130323Z/strict_L2048.json:2023`–`:2055`；CPU推导见 physical_memory_accounting.md |
| Working/COW/临时空间 | working每请求1槽，no-overlap tracking再1槽；COW destination通常就是working，donation会形成峰值 | memory_pool.py:1170,1294,1400,1438；MambaComponent:173,509 |
| 是否独立 checkpoint quota | 当前没有；所有角色共享 free list，int8独立池不在正式支持范围 | memory_pool.py:1191–1241；adapter.py对应 `flowstate/adapters/sglang.py:533` |
| Native eviction 是否先改变 C | 会：请求槽检查、COW不足、slot申请失败和path cap都可能触发 | allocation.py:279–283；MambaComponent:193,248,473 |
| 旧 K 的含义 | 选择器预算，非池上限，也无长期 quota | rq4_unified_runtime_harness.py:137–143；optimizer.py:74；controller.py:55 |
| 旧 headroom 证据 | N=28、20历史checkpoint、working reserve4+spare4；原校准并发1 | openhands_4workflow_occupancy_calibration.py:52–72,554；冻结 summary.json:8,1087 |
| 新合同能否不用重写 allocator | 有限独占/有界阶段可由控制/admission层扩展；现有接口尚不足以直接兑现 | 现有 idle control、mapping与evict可复用；缺口见 retention_budget_contract.md:第6节 |

源码短名的完整路径在 physical_memory_accounting.md 与 source_integrity.json。已阅读 P3-G0 final_report、missing_primitives、minimal_pilot_protocol；其结果和文件未修改。

## 资源合同的要点

长期保留上限 `K=N−R_exec−O`；K×b 是既有物理池中允许跨 epoch 保留的完整状态 payload，而 R_exec 是即将兑现的运行/瞬态服务预留，O 为其他已计费不可管理占用。预留必须事前绑定并能在 admission 前检查兑现，不是根据 K_min 倒算。

在空闲阶段允许 checkpoint 暂借未使用的执行 reservation，所有候选仍需真实驻留且总占用≤N。到下次执行前收回 lease，recurrent-only reconciliation 后全局 retained≤K、free≥R_exec；失败不派发。释放归还 slot，不释放底层预分配 tensor HBM。若要求任何时刻 checkpoint节点数都≤K，那就不能声称看到超过K个驻留候选。

K<K_min 的合法性要求真实候选先能在N内物化，长期配额有独立服务/部署资源依据，lease可撤销且没有active/COW/generation冲突；不由策略胜负决定。它只证明不能保留全部最深前沿，不保证任何算法获胜。

## 五个最终回答

1. **K 的真实物理含义：**同一有限 recurrent pool 中跨 epoch 的完整 checkpoint 驻留权上限，K个唯一槽、K×b payload；不是池总槽数、不是FA预算，也不是驱逐后GPU总显存必降的字节数。当前正式实验尚未实现这项长期资源合同。
2. **Candidate Count>K：可以有条件合法。**候选使用事前授权的、可撤销的空闲执行容量；全部有效且物理总量≤N，reconciliation后归还借用。若K就是硬物理总上限则不可能。
3. **Recurrent-only reconciliation：现有原语可有条件安全执行。**限独占idle、完整身份/驻留、无active/session/transfer引用、selected组件不与释放目标别名、FA保存和释放数验证；新owner/generation/全池quota门禁未补齐前不能声明合同已安全兑现。没有跨操作rollback，部分失败必须停止，不放行请求。
4. **最小配置：**建议复用 Qwen3.5-9B、TP=1、N=28、extra_buffer/no-overlap，设计Q=2和4Q=8槽保守执行预留；在峰值资格验证通过、O=0时派生K=20。状态payload含padding为1424.625MiB，实际GPU总量/新工作流峰值UNKNOWN。它适合事件验证，**不会在两pending下产生K<K_min**。没有独立资源依据，不推荐人为缩成K=1。
5. **下一步能否开始 Native Event Instrumentation：可以进入有限接口开发，但不能直接启动 Pilot。**优先补齐事件/代际账本、admission gate、联合capture、quota/lease兑现门禁和有界峰值检查；先冻结任务与资源参数并完成CPU正确性验证。真实紧预算共享竞争仍待独立资源合同，不是instrumentation完成就自动成立。

## 判定边界

不选 READY：现有 K 与N未联接，没有强制reservation/lease接口；只记录 free 或事后报错不足以隔离资源。不选 BLOCKED：物理槽借用和已有recurrent-only释放机制不存在原则矛盾，可在明确范围内补齐。对于无上界持续流式/异步/多租户工作负载，当前最小idle-only设计不提供保证。

四项主要交付：physical_memory_accounting.md、retention_budget_contract.md、minimal_pilot_resource_config.md、本报告；另有CPU容量计算与来源完整性摘要。没有运行策略、GPU、Engine、Pilot或新cohort；没有修改Core Allocator、Eviction、Optimizer、旧报告/数据或论文。本轮到设计审查为止。
