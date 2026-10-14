# 物理容量核算

本轮只读审查与 CPU 整数容量推导，未启动 GPU、Engine、Agent、Pilot、replay 或策略比较。日期：2026-10-09。

源码约定：`S/` 为 `/tmp/audit_sglang/sglang-v0517-tag/python/sglang/`，使用 P3-G0 与停止的正式实验容器逐字节核实的版本；本轮重新核验其 16 个文件摘要。其余相对路径以 `/home/wjg/code/FlowState/` 为根。新扩展仅为设计，没有实施；UNKNOWN 不当作零。

## 容量配置与真实路径

`S/srt/mem_cache/registry.py:114` 将 hybrid SSM 交给 UnifiedRadixCache，`:163`–`:189` 安装 FULL/MAMBA。HiMambaRadixCache 在 `S/srt/mem_cache/hi_mamba_radix_cache.py:97` 存在，但不是本合同的正式运行路径；不借助 HiCache/offload 隐藏额外候选。

`S/srt/server_args.py:2461` 的 max_mamba_cache_size 是总 recurrent 槽数 N，不是 retained checkpoint quota。`S/srt/mem_cache/kv_cache_configurator.py:1874` 优先使用显式 N，并按 attention DP shard 划分；本轮 TP=PP=DP=1 不发生该拆分。未显式设置时，`:1923` 按可用预算和 mamba_full_memory_ratio 分配；`:1953` 为普通模式求可用槽数，`:1976` 扣除含 padding 的状态池容量。此自动估算不是选择器 K。

`HybridReqToTokenPool._init_mamba_pool`（`S/srt/mem_cache/memory_pool.py:1191`）以同一 mamba_size 初始化 MambaPool 和 MambaSlotAllocator（`:1220`）。普通 tensor 分配在 `:540`、`:564`，形状包含 size+1。`allocator/mamba.py:93` 中 slot 0 只用于 padding，free list 为 1..N；不能把 padding 当 headroom。

## 一个 slot 的大小：有旧实测，不需要新 GPU

`/home/wjg/code/newfind/artifacts/exp2-b-PASS-20260812T130323Z/strict_L2048.json:2023` 的完整状态 fingerprint 记录 temporal=50,331,648 bytes、conv[0]=1,179,648 bytes、合计 51,511,296 bytes（`:2034`）；`:2057` 起有逐层 dtype/shape。模型配置 `/home/wjg/models/qwen3.5-9b/config.json` 与 `motivation/artifacts/checkpoint_size_20260819/findings.md:5`–`:22` 支持同一 Qwen3.5-9B、TP=1 普通池模式。

CPU 复算：24 个 recurrent layers，每层 temporal=(32,128,128)、FP32；conv=(8192,3)、BF16。因此

`b = 24 × (32×128×128×4 + 8192×3×2) = 51,511,296 bytes = 49.125 MiB`。

完整 checkpoint、working slot 和 tracking slot 的 payload 都是 b。不是 b 乘以角色数量就得到单 checkpoint 大小。findings.md 引用的 server_probe.log 当前本路径不存在，本轮没有使用该缺失日志；采用上面确实读到的原始 fingerprint 作为旧实测依据。新 Pilot 实际 tensor dtype、layout 与额外内存仍须未来启动时核验，不把旧数值当成任意配置都恒定。

静态普通池的 conv+temporal payload 为 `(N+1)b`。它不包含权重、FA-KV、CUDA workspace/graph、请求映射、allocator 索引、padding 对齐与 PyTorch 分配器开销，也不是整卡 reserved/allocated HBM。recurrent-only free 将槽归还预分配池，不释放底层 torch 大 tensor；`nvidia-smi` 显存下降不是正确性门禁。

## 角色、共享与峰值

| 项目 | 物理核算 | 直接证据 |
|---|---|---|
| Persistent checkpoint | 普通模式每个唯一 checkpoint 一个完整 slot，b；同一个 canonical checkpoint 被多个 pending 使用只计一次 | MambaComponent.commit_insert_component_data，`S/srt/mem_cache/unified_cache/components/mamba_component.py:204` |
| Working | 每个活动请求一个可写 slot，b；chunk 续接已分配的槽可复用 | memory_pool.py `:1294`–`:1325` |
| Tracking/transient | extra_buffer 且 no-overlap 时每请求一槽；overlap 为两槽，本合同禁用后者 | memory_pool.py `:1170`、`:1400`–`:1423` |
| COW destination | 通常就是该请求 working slot；不能再给同一 destination 重复加 b。多个分支共享 source，但各自 destination 独立 | MambaComponent.finalize_match_result_in_cache `:173`–`:201`；model_runner.py `:1505`–`:1520` |
| Tracking donation/replacement | donation 将旧 tracking 槽转为缓存，并分配新 tracking 槽；长期增加的 checkpoint 和新 working/transient 的并存计入峰值 | MambaComponent.prepare_for_caching_req `:509`–`:586`；memory_pool.py donate_mamba_ping_pong_slot `:1438` |
| Native reserve | 不是单独 tensor 或已隔离的物理区，当前主要是 admission/容量估算中的空闲槽需求 | allocation.py `:252`–`:284`，kv_cache_configurator.py `:1659`、`:1734` |

普通池没有 persistent-only 分区。工作、tracking、捐赠后的 checkpoint 都从同一 free list 取得/归还槽。int8 checkpoint pool 可独立，但正式 adapter 不支持，本轮不改模式来获得容量优势。ReplaySSM、speculative、Unified Memory、offload 的额外缓冲不纳入本合同，须关闭。

精确账目使用物理集合，而不是相加可能重叠的角色计数：`N = |Retained ∪ Borrowed ∪ Working ∪ Tracking ∪ OtherAllocated| + Free`；要求这些所有权集合有效且互斥，合法 donation 记录一次所有权转交。COW source 属于 Retained/Borrowed，不重复计数。不同逻辑 checkpoint 指到同槽时先 canonical 去重/验证，不能按逻辑 ID 数宣称物理容量。

## 两种 headroom 因子不能混淆

`kv_cache_configurator.py:118`–`:123`、`:1659`：默认 base=3，no-overlap extra_buffer 加1，规划因子为4。`:1737` 使用 `floor(N/4)` 限制最大并发请求。本因子要求 `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=false`（S/srt/environ.py:982 默认如此）；开启后不能继续按同一推导解释。它是可容纳请求数量的保守规划，不是“每个请求固定占用四个 checkpoint”，也不是运行时强制留空四槽的独立 quota。

`S/srt/mem_cache/common.py:27` 的 prefix-cache factor=3；`allocation.py:279` 在 alloc_req_slots 时要求 free≥3×本次请求数，否则 `:283` 触发 native eviction。COW 可能已经提前分配 working destination，所以在 match 前只有3Q空闲不能保证之后检查通过。必须分别记录 match 前、alloc_req_slots 前和插入/捐赠时的实际 free、请求数与已有 working 槽，不只检查 epoch 末的一次 census。

## 旧正式配置到底证明什么

- `evaluation/control_path_optimization_output/control_20260929/optimized_formal/protocol.json:17`–`:31`：N=28、extra_buffer、no-overlap、track_interval=256、path cap=-1、TP=1；四请求串行。没有 retention quota 配置。
- `evaluation/openhands_4workflow_occupancy_calibration.py:52`–`:72`：28槽、预期20 persistent、working reserve=4、额外 spare=4；`:554` 的 gate 检查无 native eviction 且最小 free≥4。
- 冻结 `evaluation/runtime_artifacts/openhands_4workflow_occupancy_calibration_20260831_050831_071744/summary.json:8` 为 concurrency=1；`:1087` 起记录槽规划，`:1093` PASS。该 summary 的 :246 记录 census 的 minimum_available_mamba_slots=8、:251 无 native capacity eviction、:255 headroom sufficient。这是已有离散 census/门禁证据，不是新并发工作流的连续峰值测量，也不证明两个请求并发时同样 peak。
- `evaluation/rq4_unified_runtime_harness.py:137`–`:143`：K=max(1,floor(0.25×candidate_count))；`:949` 的8/12/16/20候选分别K=2/3/4/5。`flowstate/optimizer.py:74` 仅做 budget_bytes//checkpoint_size；`flowstate/controller.py:55`–`:116` 仅当轮选择和驱逐，没有长期配额、lease 或物理 reservation。

因此旧 K 是施加到状态保留集合的逻辑约束，有真实驱逐效果，但**不是已实现的物理池容量绑定预算**。不得用旧 PASS 证明新合同已执行。
