# Runtime 状态能力

本轮为只读源码审查，未运行 Agent、Engine、GPU、trajectory replay 或策略比较。日期：2026-10-09。

路径约定：`S/` = `/tmp/audit_sglang/sglang-v0517-tag/python/sglang/`；`B/` = `/tmp/audit_sglang/sglang-v0517-tag/`；其余相对路径以 `/home/wjg/code/FlowState/` 为根。源码引用均为本轮读取版本。16 个关键 SGLang/runtime/工作流文件已与停止的正式控制实验容器 `flowstate-control-profile` 逐字节对照，见 `source_verification.json`。早期 `/tmp/sglang0517_src` 的 Mamba 文件摘要不同，本报告不沿用其行号或新增行为。

## 实际 cache 路径

`S/srt/mem_cache/registry.py:114` 将 hybrid SSM 送到 `_create_unified_radix_cache(:163)`；`:172`–`:176` 安装 FULL 和 MAMBA。实际复用对象是 UnifiedRadixCache + MambaComponent。`S/srt/mem_cache/hi_mamba_radix_cache.py:97` 的 HiMambaRadixCache 是另一条实现；不能凭类名把它认作正式控制路径。当前 adapter 明确拒绝 HiCache、session radix 和 int8 checkpoint pool（`flowstate/adapters/sglang.py:533`），Pilot 保持该边界。

## 只读能力与缺口

| 项目 | 现有接口与代码 | 结论和必要限制 |
|---|---|---|
| Request / Session Identity | `S/srt/managers/io_struct.py:164` rid、`:167` session_id；Engine.generate `S/srt/entrypoints/engine.py:384`；`tests/runtime/wp3b_end_to_end_transport.py:44` 从 params.req.rid 记录匹配结果 | rid 可关联；session_id 不是 fork ancestry。空闲时请求对象可能已释放，需要实时事件留档 |
| FA-KV frontier R | `evaluation/barrier_fa_frontier_control.py:26` traverse_device_fa_frontier，`:96` inspect_resident_fa_frontier | 遍历子节点与 FULL value，不调用有 recency/COW 副作用的 match_prefix；遵守 page alignment 和 effective_lookup_limit，默认生成请求留最后 token（`:12`） |
| Checkpoint identity / L / residency | `motivation/artifacts/wp3b_gate_20260820/targeted_probe.py:174` _path_snapshot，`:209`–`:233` 返回 prefix、node、沿路径位置、slot、引用和驻留 | 可以证明当前物理映射，不能凭 checkpoint_id 命名证明原始请求或创建时点。需要插入提交事件与 runtime token identity |
| 映射 | `flowstate/adapters/sglang.py:19` RuntimeCheckpointHandle；`:291` 查 exact node；`:332` 校验 node/prefix/value；probe 的 target_mamba_slots | 完整路径可读取；需按 engine UUID/rank/pool/slot 标识组件，canonical checkpoint 去重；逻辑 ID 唯一不等于 slot 唯一 |
| Generation / ownership | `S/srt/mem_cache/memory_pool.py:283` 的 req_generation，`:315` 请求槽分配递增；`S/srt/mem_cache/allocator/mamba.py:81`/`:88` 仅切 free list/归还 | req_generation 是 request pool 代际，不是 recurrent slot generation。RuntimeCheckpointHandle 未存 recurrent generation。缺 alloc/free/donate/insert/COW 的关联事件，跨 epoch ABA 身份无法单凭 slot 整数排除 |
| TP 一致性 | `tests/runtime/rq5b_kimi_phase1_transport.py:19` runtime_scope、`:130` 每 rank 端口；`evaluation/rq5b_kimi_phase2.py:74` rank_facts、`:87` 双 rank submit | 现有两个 rank 的逻辑事实比较和 rank-local slots 可复用；并发 RPC + 事后比较不是跨 rank 原子屏障。首个 Pilot 限 Qwen TP=1，Kimi TP=2 仍需全 rank barrier ACK |
| 容量/headroom | probe `_accounting_snapshot(:237)` 返回 free、schedulable、evictable/protected；allocator size `mamba.py:38`；MambaPool conv/temporal shape/bytes | 可读取池占用与物理组成；并未自动产生业务 K、working reserve、临时预留资源合同。空闲 free 数也不是可直接分配的长期 checkpoint budget |

只读非干扰证据可复用 `tests/runtime/control_path_optimized_transport.py:33` semantic_state、`:39` frontier_batch 的逐查询前后语义比较。该入口在 `:43` **硬编码四个 pending**，不能直接用于至少两个分支的 Pilot；应新增独立可变长度捕获接口，保留原入口与门禁，不改旧实验。现有 `_batch_view`（`tests/runtime/rq6_batched_control_transport.py:54`）不包含 pending R；必须把路径、R、accounting 合并到同一受控观察边界，不能把两次 RPC 的 nonce 当一致性证明。

## 共享祖先与分支状态的真实执行路径

1. `MambaComponent.finalize_match_result_in_cache`（`S/srt/mem_cache/unified_cache/components/mamba_component.py:173`）取 best_match_node 的 source slot，为请求 alloc 独立 destination slot（`:187`），设置 `req.mamba_cow_src_index(:200)`。源缓存状态不直接成为可写工作状态。
2. `S/srt/managers/schedule_batch.py:2646` 收集 COW 源/目的；`S/srt/model_executor/model_runner.py:1505`–`:1520` 真正执行状态复制。`MambaPool.copy_from`（`memory_pool.py:939`）复制全部 conv 和 temporal（`:967`–`:977`）。只看到 match 字段不能声称复制已完成，需记录完成的 GPU 工作 fence。
3. `MambaComponent.prepare_for_caching_req(:509)` 选择真实 tracking 长度；finished 时转交 slot index，unfinished 时 donate buffer 或复制工作状态；`HybridReqToTokenPool.donate_mamba_ping_pong_slot`（`memory_pool.py:1438`）把已捐槽替换成新槽。`commit_insert_component_data(:204)` 将值挂到 exact node；`cleanup_after_caching_req(:588)` 保留已插入槽并释放其他工作/跟踪槽。
4. 不同 token 路径有各自新叶且池容量允许时，祖先和分支专属 checkpoint 可以同时存在。**源码支持这种机制，不等于本次真实工作流已经形成这种状态。** 两个分支若输出相同、token 边界不匹配或未触发 tracking，可能只有一个 checkpoint；必须如实失败，不能改 lineage/捏造 fork 或强行分出逻辑 ID。
5. `redistribute_on_node_split(:296)` 将新父 recurrent value 设为 None，并不从深状态插值出浅祖先。真实 fork 文本位置必须有已物化且仍驻留的祖先 L，不能靠 tree split 生成 recurrent checkpoint。

自动破坏候选的路径：COW alloc 不足 `:187`–`:198`、`_alloc_mamba_slot(:473)` 会触发 native eviction；`_evict_excess_path_states(:248)` 在 cap 启用时删部分浅状态，虽保护 tail/fork/locked/leaf，但不保证将来才形成 fork 的祖先提前受到保护。配置、内存和事件必须全量记录。RQ5-B `evaluation/rq5b_kimi_phase1.py:29`–`:31` 显式 extra_buffer、max_states_per_path=-1，仅可作已有路径参考；新 Pilot 参数必须事前冻结，不能为修复失败事后调大池或改 cap。

## Allocation Boundary 与保留集合

`tests/runtime/rq6_batched_control_transport.py:275` 只在 is_fully_idle 时 drain_one；probe `:498` 同步处理一个控制消息，但不是长期锁。SGLang `scheduler.py:3967` 的 is_fully_idle 包括 running/last/chunked/overlap/PP 与 `:3984` waiting_queue 为空。收到未来请求的 socket、外部 client 或两次控制 RPC 之间的执行不由这些条件永久阻止。

最小可验证范围：独占一个 Engine，所有真实 Ready 请求在 Agent admission gate 外等待；已发送请求全部完成且无在途发送；关闭 overlap、HiCache/session/disaggregation 等未覆盖异步变化；scheduler 安全点内一次 capture 读取 C/R/accounting，并确认 GPU 任务完成及无插入/传输；完成后才解除 admission。TP>1 要所有 rank 对同一 barrier ACK，任一失配整体拒绝，不能用 batch 或时间相近替代。

`tests/runtime/rq6_batched_control_transport.py:194` 的 reconciliation 校验 digest、集合、身份、驻留，逐项调用 adapter.evict_mamba_only，再统一验证 selected residency、FA/结构与释放数（`:94` 的 _proof）。adapter 在 `:399` 拒绝设备/主机/session 引用，`:416` 只分离 recurrent 不 cascade。共享祖先与独立分支节点原则上可用同一 substrate；需 canonical 去重、物理别名检查、全 rank 保留集检查。它不是 fork 操作，不自动 pin selected 状态防止之后 native eviction，也不能驱逐 active/transient/session-protected 状态。Pilot 不执行任何策略选择或实际集合驱逐，仅审查接口可满足的前提。

StateCatalog (`flowstate/state_catalog.py:12,60`) 和 PendingContinuation (`flowstate/workflow.py:9,25`) 接收上游 lineage、驻留布尔和 A/R，计算 compatibility、T=min(A,R)；它们不采集事件、不证明 ancestry、不持有物理容量控制。不能把数据类实例当作运行时证书。
