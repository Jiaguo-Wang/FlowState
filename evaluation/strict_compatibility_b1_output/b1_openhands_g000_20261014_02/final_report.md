# Strict Compatibility B1 最小 Runtime Pilot 报告

**最终门禁：PARTIAL**

本次确认：在既有 OpenHands 冻结 group 上，可以在同一受控 SGLang Runtime lifecycle 中采集真实请求 Token IDs、Checkpoint Token 前缀、Runtime 节点与 Recurrent slot、FA-KV/Recurrent 驻留、引用计数及请求恢复前沿。一个同 Workflow 正例及其 recurrent-only eviction 反事实通过单令牌输出正确性检查。严格兼容矩阵可在该边界生成；但请求命中记录仍不返回被使用的 Node ID，槽位代际也不可见，因此尚不能把所有 epoch-admissible 边提升为逐边的 Runtime 恢复认证。

## 执行范围与环境

- 复用冻结输入：`evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017/population_manifest.json` 中 `group_ordinal=0`、`allocation_round=2`；四个既有 Session 为 A/B/C/D。没有重新选择或生成总体。
- Runtime：本地镜像 `lmsysorg/sglang:v0.5.17-cu129-runtime`，SGLang 0.5.17；Qwen3.5-9B；单张 H100；TP=1。使用既有 Common Barrier engine 配置：`context_length=131200`、`mem_fraction_static=0.4`、`max_mamba_cache_size=28`、`mamba_radix_cache_strategy=extra_buffer`、`mamba_track_interval=256`、关闭 CUDA Graph 与 overlap scheduling。
- Tokenizer 和 Chat Template 使用同一模型目录；Token IDs 由历史 collector 的 `apply_chat_template` 路径产生并原样提交。完整 Tokenizer 文件及模型 `config.json` 已记录 SHA-256。模型权重文件未逐片计算 SHA-256；权重内容指纹为 UNKNOWN，不能宣称完成权重级内容认证。
- 主机侧 Python 3.12.13（`/home/wjg/uv312`）用于语法检查；GPU Runtime 按用户要求使用原有 SGLang 容器环境。
- 数据集文件为只读挂载；代码仓库以只读方式挂载，只有新输出目录可写。历史 frozen artifact 未被写入。

## 当前信息边界与状态边界

冻结回放在每个 Session 的 pending turn 3 输入标记处停止构造。四条边界记录均显示：未读取 pending assistant 输出、未消费 r+2 消息、未构造 r+2 请求、未读取未来计时或 Checkpoint。Pending 的完整输入 Token IDs 在历史请求 replay 前即由当前已知消息构造，并与实际传给 `engine.generate(input_ids=...)` 的值一致。

八个历史请求（四个 Session 各执行 turn 1 和 turn 2）按冻结 collector 顺序串行执行，之后进入无活动请求的 quiescent barrier。Barrier 后逐项查询 Checkpoint 与 pending FA frontier；这是同一静止 epoch 的顺序观测，不声称为跨组件原子事务。census 报告 28 个 Mamba slots、8 个 resident recurrent slots、20 个可用 slots、8 个 evictable slots、0 个 protected slots；未观察到 native recurrent capacity eviction 或 FA cascade eviction。

## 采集统计

| 项目 | 结果 |
|---|---:|
| 冻结 Snapshot | `b1-openhands-g000-round2`（group 0） |
| 真实物化候选 Checkpoints | 8 |
| Barrier 上已知 Pending Continuations | 4 |
| Candidate–Pending 配对 | 32 |
| `EPOCH_ADMISSIBLE` | 11 |
| `REJECTED` | 21 |
| `UNKNOWN` | 0 |
| 其中跨 Workflow 的 epoch-admissible 边 | 3 |
| FlowState/Baseline 策略比较 | 未执行 |

`EPOCH_ADMISSIBLE` 表示该边满足本 Pilot 明确的 Token 前缀、`L≤T`、同模型/Tokenizer/Runtime 配置、同一 quiescent barrier、完整 FA 与 recurrent 驻留、节点/前缀摘要一致及可见引用计数清零条件；它不自动表示请求已实际使用该特定物理节点。21 条拒绝边由 `L>T` 或完整 Token 前缀不匹配解释。

最具代表性的跨 Workflow 候选为 `RQ3_G000_B_TURN_001`：Runtime node 29、Recurrent slot 6、`L=2432`。它与 A/B/C/D 四个 pending 的输入均有相同的 2432-token 前缀且各自 `T≥2432`，因此产生 3 条跨 Workflow 的 `EPOCH_ADMISSIBLE` 边。当前正例选择了同 Workflow 的更深 `RQ3_G000_A_TURN_002`；跨 Workflow 的三条边没有逐边隔离恢复，故只能称为边界准入证据，不能称为跨 Workflow Runtime 恢复认证。

## Runtime 恢复验证

正例选取 `RQ3_G000_A_TURN_002`：`L=4352`、node 32、Recurrent slot 12；目标为 `RQ3_G000_A_PENDING_TURN_003`，`T=4352`。checkpoint 前缀与 pending 输入逐 Token 相同，运行时检查显示 FA 与 Recurrent 均驻留，四项可见引用计数均为 0。

| 条件 | 请求 | H | E | G=H−E | 输出 Token 与全量重算参考 |
|---|---|---:|---:|---:|---|
| 正例：候选 Recurrent state 驻留 | A pending turn 3 | 4352 | 4352 | 0 | `248058 = 248058`，PASS |
| 反事实：只驱逐该 Checkpoint 的 Recurrent state | 同一 A pending turn 3 | 4352 | 3904 | 448 | `248058 = 248058`，PASS |

驱逐返回的安全证明为：目标 node 相同、仅目标 Mamba 组件变化、FA identity 与 FA residency 保持、树结构不变、allocator 检查通过、scheduler idle 且没有活动或排队请求。驱逐后 node 32 的 FA 仍驻留、Mamba 不再驻留；恢复前沿由 4352 降为 3904，说明请求仍可从更浅的 recurrent state 恢复。该结果不等于完整重算，也没有发生错误输出。

请求级指标只暴露 `physical_fa_hit`、`executable_prefix` 和 `replay_gap`，未暴露实际命中的 Node ID。因此，正例与驱逐差分提供了对 node 32 Recurrent 状态贡献的因果支持，但“请求—物理 Node ID”直接映射仍为 UNKNOWN。Runtime 也没有暴露 slot generation。

21 条 `REJECTED` 边中有精确 Token mismatch 的负边；本 Pilot 没有单独构造并执行一个“只剩不兼容候选”的反例请求。因此，负边的 Token 级拒绝已验证，负例恢复路径的逐节点归因尚未验证。

## A–G 结论

**A. Pilot Gate：PARTIAL。** 真实采集与一个恢复/驱逐闭环通过；跨 Workflow 逐边 Runtime 认证、负例隔离恢复、slot generation 与模型权重内容指纹尚缺。

**B. 已补齐字段与代码位置。** 新采集器为 `evaluation/strict_compatibility_b1/run_pilot.py`。它复用 `evaluation/rq3_openhands_neutral_collector.py` 的 `materialize_group_requests`、`replay_group_to_barrier` 与 `sglang_runtime_factory`；复用 `motivation/artifacts/wp3b_gate_20260820/targeted_probe.py` 的 checkpoint inspect/census/eviction 证据，以及 `tests/runtime/wp3b_end_to_end_transport.py` 的请求匹配指标。产物包含实际 Pending Token IDs、Checkpoint Token IDs、Token digest、配置摘要、节点 ID、slot、resident components、可见引用计数、census、barrier 时间戳、H/E/G、输出 Token 与安全 eviction proof。

**C. Snapshot、Candidate、Pending 与 Edge 数量。** 1 个冻结 Snapshot；8 个候选；4 个真实 barrier pending；32 条配对边。

**D. Compatibility 统计。** 11 `EPOCH_ADMISSIBLE`、21 `REJECTED`、0 `UNKNOWN`。3 条准入边跨 Workflow，但尚未逐边验证真实请求使用对应的跨 Workflow状态。

**E. 已完成 Runtime 验证。** 一条同 Workflow 正例达到候选位置并与全量重算输出一致；对该候选执行安全 recurrent-only eviction 后，FA 保留、E 下降 448 tokens，输出仍一致。由于 match instrumentation 不返回 Node ID，证据等级为“候选级状态差分支持恢复贡献”，而不是直接 Node-to-request handle 认证。

**F. 仍缺能力。** 请求 match 记录增加实际命中 Node ID 或等价 recurrent handle；提供 slot generation/ownership generation；逐边隔离验证至少一条跨 Workflow 边及一条不兼容边；为模型权重建立内容指纹；对候选状态规模与逻辑预算建立正式容量合同。当前 28-slot pool 在 8 个候选后仍有 20 个空闲 slot，未形成真实物理容量竞争。

**G. 是否扩展到更多 OpenHands Snapshots。** 暂不建议扩展到其余 167 个 Snapshot。先补 request-to-node 关联，并用当前已观察到的 B1→A 跨 Workflow 边完成一次隔离恢复验证；本 Pilot 没有策略比较，也不能说明不同保留策略会产生不同结果。

## 执行记录与限制

最终有效结果位于本目录 `pilot_evidence.json`。此前两次试运行分别在字段映射阶段失败或因审计器要求了探针未提供的引用字段而被保守门禁阻断；它们没有作为结果计数。最终运行使用修正后的探针字段白名单，未放宽任何兼容条件。GPU 运行结束后 `nvidia-smi` 显示两张 H100 各 14 MiB，占用与运行前一致；Pilot 容器已退出，GPU cleanup PASS。

最终状态是 `PARTIAL`，不是完整 Runtime Compatibility Closure，也没有覆盖完整 OpenHands population。
