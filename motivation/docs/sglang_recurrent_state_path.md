# SGLang Hybrid/Mamba Recurrent State 完整代码路径调查

> 调查日期：2026-08-18。目的：为 FlowState Motivation Experiment 做代码路径准备。本轮只调查，不修改 SGLang 核心代码。
>
> 所有行号基于实验冻结镜像源码（提取快照 `/tmp/sglang0517_src/sglang`），行号指 `python/sglang/srt/` 下的文件。

---

## 1. 当前 SGLang 版本

| 项 | 值 |
|---|---|
| 来源 | Docker 镜像 `lmsysorg/sglang:v0.5.17-cu129-runtime`（image id `1dfce9033670`，容器 `specsa-sglang-v0517-dev` 内 `/sgl-work-space/sglang`） |
| `pip show sglang` | 0.5.17 |
| git commit | `29481685462732237d80d86076d6563e1f658102` |
| git describe | `v0.5.17`（branch release/v0.5.17，2026-08-07） |
| 宿主实验仓库 | `newfind` @ `exp2/ad-freeze`，HEAD `540774bdb678ce3197c50231bd1b27eca1810dc5` |

**⚠️ 前提性关键发现（影响全文阅读方式）：此版本存在两套语义相同、dispatch 不同的 mamba radix 实现。**

| | LIVE（默认实际运行） | LEGACY（用户清单中的符号所在） |
|---|---|---|
| 类 | `UnifiedRadixCache` + `MambaComponent` | `MambaRadixCache` / `HiMambaRadixCache` |
| 文件 | `mem_cache/unified_radix_cache.py`、`mem_cache/unified_cache/unified_tree_core.py`、`mem_cache/unified_cache/components/mamba_component.py` | `mem_cache/mamba_radix_cache.py`、`mem_cache/hi_mamba_radix_cache.py` |
| 选择 | `registry.py:114-115`：`ctx.is_hybrid_ssm` **无条件** → `_create_unified_radix_cache()`（`registry.py:163-193`）；`--enable-hierarchical-cache` 时同路 + `init_hicache`（`registry.py:188-189`） | **v0.5.17 默认链不可达**：全树无 `HiMambaRadixCache(` 实例化点；`mamba_radix_cache` 模块只被 `hi_mamba_radix_cache.py:38` import |
| 判定 `is_hybrid_ssm` | `kv_cache_builder.py:159-165`（GDN / Mamba2 / KimiLinear / hybrid_lightning / 模型注册 `uses_mamba_radix_cache` 任一命中） | `server_args.py:6007` 注释声称 `--enable-hierarchical-cache` 启用 HiMambaRadixCache，**与 registry 实际行为矛盾（stale comment）** |

两套实现的 match / insert / evict / restore / replay **语义逐条对齐**（下文每节先给 LIVE 版行号，括注 LEGACY 版对应行号）。`schedule_batch.py` / `model_runner.py` 侧的 track、COW、branching 消费逻辑为两套共享（只依赖 Req 字段与 `BasePrefixCache` 协议）。

- 验证方法（同时是 NEEDS_RUNTIME_VALIDATION #1）：启动 hybrid-SSM 模型 + `--enable-hierarchical-cache`，看日志 `Tree cache initialized: ... impl=...`（`registry.py:233-242` 打印 impl 类名），预期 `UnifiedRadixCache`。**✅ 已验证 2026-08-19**（无 hicache 配置下）：`Tree cache initialized: source=default impl=UnifiedRadixCache hybrid_swa=False hybrid_ssm=True hierarchical=False`（artifact `runtime_validation_gap_replay_20260819/server_full.log`）。

---

## 2. Recurrent State 数据结构（路径 1）

### 2.1 GPU pool：`MambaPool`

- `class MambaPool`：`memory_pool.py:329`；内部状态容器 `MambaPool.State`（frozen dataclass，`memory_pool.py:334-379`）：
  - `conv: List[torch.Tensor]` — 每 tensor shape `(num_mamba_layers, size+1) + conv_shape`；标准 Mamba2/GDN conv_shape = `(conv_dim/tp, conv_kernel-1)`（`memory_pool.py:540-547`）
  - `temporal: torch.Tensor` — shape `(num_mamba_layers, size+1) + temporal_state_shape`，GDN/Mamba2 为 `(H/tp, head_dim, state_size)`（`memory_pool.py:564-568`）
  - `replayssm_d/k/g/...`：GDN ReplaySSM ring buffer，仅 `--enable-linear-replayssm*` 时分配（`memory_pool.py:575-668`）
  - `SpeculativeState` 扩展 `intermediate_ssm` / `intermediate_conv_window`（`memory_pool.py:381-387, 670-782`）
- **布局：`[num_mamba_layers, slot]` 二维前导轴，每 request 占 1 个 slot 的全部 mamba 层状态**。per-layer 切片：`State.at_layer_idx`（`memory_pool.py:359-372`）→ `MambaPool.mamba2_layer_cache`（880-881）→ `HybridReqToTokenPool.mamba2_layer_cache(layer_id)`（`memory_pool.py:1366-1367`，经 `mamba_map` 全局 layer_id→pool index，1355-1364）。
- slot 分配器 `MambaSlotAllocator`：`allocator/mamba.py:30-97`，request 级 free-list，**slot 0 保留为 padded dummy 写目标**（`free_slots = arange(1, size+1)`，`allocator/mamba.py:93-97`），故 tensor dim1 = `size+1`。
- req→slot 映射：`HybridReqToTokenPool.req_index_to_mamba_index_mapping`（`memory_pool.py:1242-1244`，写于 `alloc()` 1335-1336，读 `get_mamba_indices` 1344-1345）；请求侧持有 `req.mamba_pool_idx`（`schedule_batch.py:900`）。
- ping-pong track buffer（`--enable-mamba-extra-buffer`）：`req_index_to_mamba_ping_pong_track_buffer_mapping (req_pool_size, 2)`（`memory_pool.py:1246-1252`；`_alloc_ping_pong_buffer` 1400-1424，`donate_mamba_ping_pong_slot` 1438-1461）。
- 可选 int8 checkpoint 池：`HybridReqToTokenPool.mamba_ckpt_pool`（`memory_pool.py:1233-1238`）→ `MambaCheckpointPool`（`mamba_checkpoint_pool.py:186`，内嵌自己的 slot allocator :225）。注意：`--enable-int8-mamba-checkpoint` 与 `--enable-hierarchical-cache` 互斥（`server_args.py:6004-6023` 显式 reject）。

### 2.2 与 FA-KV pool 的组合

- `HybridLinearKVPool`（`memory_pool.py:3555`）= `full_kv_pool`（`MHATokenToKVPool` 等，3600-3635）+ `mamba_pool`（3591）。**`mamba_pool` 与 `HybridReqToTokenPool.mamba_pool` 是同一实例**（`kv_cache_configurator.py:1383-1392`）。
- **slot 空间完全独立**：FA-KV 是 per-token/per-page slot（`token_to_kv_pool_allocator`，索引写 `req_to_token[req_idx, :seq_len]`）；mamba 是 per-request slot（`MambaSlotAllocator` 注释明确 "do NOT inherit the KV base class"）。两者只在 radix 树节点上关联（`TreeNode.value` 与 `TreeNode.mamba_value` 并存）。
- `HiMambaRadixCache.__init__` 强制要求 `HybridLinearKVPool + HybridReqToTokenPool`（`hi_mamba_radix_cache.py:113-121`）。

### 2.3 Prefill/Decode 产生与更新

**Prefill（extend）**：
- GDN（qwen3_next 路径）`gdn_backend.forward_extend`：conv 由 `causal_conv1d_fn(..., conv_states=..., cache_indices=...)` **in-place** 更新（`layers/attention/linear/gdn_backend.py:563-573`）；ssm 由 `kernel_dispatcher.extend(..., ssm_states=...)` in-place 返回 `last_recurrent_state, h`（668-684）。
- Mamba2 `MambaMixer2.forward`：`mamba_chunk_scan_combined(..., return_varlen_states=True)` → `ssm_state[state_indices_tensor_p] = varlen_state`（`layers/attention/mamba/mamba.py:601-632`，scatter 在 632）。
- 中途 track（本调查核心之一，详见 §3.2）：conv `conv_states[conv_states_mask_indices] = x_to_track`（`gdn_backend.py:555-561`、`mamba.py:566-571`）；ssm `_track_mamba_state_extend`（`hybrid_linear_attn_backend.py:801-822`）：aligned 从 `ssm_states[track_ssm_final_src]` 拷、unaligned 从中间缓冲 `h[track_ssm_h_src]` 拷（src/dst 计算 `_init_track_ssm_indices`，`hybrid_linear_attn_backend.py:313-360`）。

**Decode**：
- GDN `forward_decode`：`causal_conv1d_update` + `kernel_dispatcher.decode(..., ssm_states, cache_indices=...)` in-place（`gdn_backend.py:379-468`）；Mamba2：`selective_state_update(..., state_batch_indices=...)`（`mamba.py:739-752`）。
- track 触发条件：`seq_lens_cpu % mamba_track_interval == 0`（`mamba2_metadata.py:43-47` 注释、`hybrid_linear_attn_backend.py:392-406` `_replayssm_track_flush_mask` 同条件对齐）。
- CUDA graph：状态本体（pool tensor）被 kernel in-place 写，**无需静态化**；静态化的只有 slot 索引 buffer（`state_indices_list`、`mamba_track_indices_buf`，`hybrid_linear_attn_backend.py:408-463`；replay 时 `_replay_metadata` 544-725 刷新）。

**Host 侧**：
- `MambaPoolHost`（`memory_pool_host.py:66`）：`temporal_buffer` + `conv_buffer`（162-228）；D2H `backup_from_device_all_layer`（475-520）、H2D `load_to_device_per_layer`（428-473）；组装进 HiCache 经 `build_hybrid_mamba_stack`（`hybrid_cache/hybrid_pool_assembler.py:548-627`，`PoolName.MAMBA` entry 的 `device_alloc_fn/free_fn` 直接接 `mamba_allocator`）。
- 非 HiCache 的整状态 CPU 拷贝：`MambaPool.get_cpu_copy / load_cpu_copy`（`memory_pool.py:988-1037`）——exp-2 harness 的 offload/restore RPC 走的就是这对 API 的血缘。

### 2.4 TreeNode 上的四态标记（语义核心）

`mamba_radix_cache.py:119-133`（unified 版为 `UnifiedTreeNode.component_data[ComponentType.MAMBA]`，`unified_cache/components/tree_component.py:48`）：

| 属性 | 定义 | 含义 |
|---|---|---|
| `evicted` | `value is None` | FA-KV device 状态已释放 |
| `backuped` | `host_value is not None` | FA-KV 有 host 备份 |
| `mamba_evicted` | `mamba_value is None` | mamba device 状态已释放 |
| `mamba_backuped` | `mamba_host_value is not None` | mamba 有 host 备份 |

**四者正交** → FA-KV 与 mamba 可以独立 eviction/backup（§4）。

---

## 3. Checkpoint 创建路径（路径 2）

checkpoint 有两层含义，分开追踪：

### 3.1 树内 checkpoint（device slot 挂到 TreeNode）——请求边界产生

调用链（两栈共享的 scheduler 侧）：

```
decode 完成/prefill 完成
→ release_kv_cache (mem_cache/common.py:132-167, cache_finished 调用点 :149)
   / maybe_cache_unfinished_req (common.py:98-102)
   调用点: batch_result_processor.py:108/275/278/354/357/1058,
           scheduler.py:2837 (stash_chunked_request, chunked=True),
           scheduler.py:4350/4370/4386 (abort, is_insert=False),
           schedule_batch.py:1855 (retract)
→ MambaRadixCache.cache_finished_req (mamba_radix_cache.py:541-664)
   [unified 对应: UnifiedRadixCache.cache_finished_req → MambaComponent.prepare_for_caching_req (mamba_component.py:509+)]
→ mamba_value 来源（mamba_radix_cache.py:597-625）:
   a) enable_mamba_extra_buffer: ping-pong keep slot（state@mamba_last_track_seqlen）, :598-618
   b) int8: _commit_int8_checkpoint, :613-616/620-623
   c) 默认: req.mamba_pool_idx.unsqueeze(-1).clone()（请求 active slot 克隆入树）, :625
→ insert(InsertParams(key=token_ids[:cache_len], value=kv_indices, mamba_value))
   cache_len = req.mamba_last_track_seqlen（extra_buffer 时, :560/574-580 KV 尾部截到 track 边界）
   [unified: unified_radix_cache.py:624; 尾节点 commit: mamba_component.py:204-236 commit_insert_component_data]
→ _insert_helper 走树/建节点；命中已存在节点且其 mamba_value 为 None 时补挂（hi_mamba_radix_cache.py:917-923）
```

`cache_unfinished_req`（chunked prefill 中途，`mamba_radix_cache.py:666-785`）：同样以 `mamba_last_track_seqlen` 截断，donate ping-pong slot（723-727）或 `mamba_pool.copy_from` 到新 slot（729-736）。

### 3.2 中途 checkpoint（track 机制）——forward 途中按 interval 快照

**这是 checkpoint position 的真正来源**。`--enable-mamba-extra-buffer` 开启：

- **Decode 侧**：每步 decode 后 `_mamba_check_track_boundary`（`batch_result_processor.py:1170-1195`）：`req.kv_committed_len % mamba_track_interval == 0` → 该步 forward 已把 state snapshot 进 ping-pong 非 active slot（kernel：`track_mamba_states_if_needed`，`kernels/ops/mamba/mamba_state_scatter_triton.py:85-125`，调用点 `hybrid_linear_attn_backend.py:757-799`），随后换 `mamba_next_track_idx`、记 `req.mamba_last_track_seqlen = track_seqlen`（`batch_result_processor.py:1100-1111`）。lazy 模式第二 slot 按需分配：`mamba_lazy_prealloc_at_boundary`（`schedule_batch.py:2856-2885`）。
- **`mamba_track_interval`**：`server_args.py:2518-2520`，**默认 256**（约束：≥ speculative draft tokens 且 % page_size == 0，5607-5609）。
- **Extend 侧**：`prepare_for_extend` 里逐 req 构建 track entry（`schedule_batch.py:2419-2423` → `_mamba_radix_cache_v2_req_prepare_for_extend`，2557-2638）：`mask = extend_range.length >= mamba_cache_chunk_size`（2572），目标 seqlen = chunk 对齐位置（2587-2591），**每次 extend forward 最多 track 一个位置**。
- **Branching 强制 track（replay 自愈，详见 §6）**：`schedule_batch.py:2616-2631`。
- 快照落地为 `track_ssm_final_dst/track_ssm_h_dst` 指向的 ping-pong slot；aligned 取 `last_recurrent_state`、unaligned 取中间 `h`（`_init_track_ssm_indices`，`hybrid_linear_attn_backend.py:313-360`）。

### 3.3 Host backup（D2H 物化）

LEGACY 栈（用户符号）：`HiMambaRadixCache.write_backup`（`hi_mamba_radix_cache.py:213-252`）：

```
触发:
  a) write_through 策略: _inc_hit_count 中 hit_count >= write_through_threshold
     (hi_mamba_radix_cache.py:376-383; 调用自 _insert_helper :905/916;
      write_through_threshold = 1 if write_through else 2, :175-177)
  b) write_back 策略: eviction 前强制 backup (_evict_device_leaf, :702-706)
→ mamba_backup_transfers(node)  :2069-2079  构造 PoolTransfer(PoolName.MAMBA,
    host_indices=node.mamba_host_value, device_indices=node.mamba_value)
→ cache_controller.write(device_indices=node.value, extra_pools=extra_pools)
  (hybrid_cache/hybrid_cache_controller.py:364; 底层 MambaPoolHost.backup_from_device_all_layer,
   memory_pool_host.py:475-520)
→ 成功后 mamba_backup_commit(node, transfers)  :2081-2090
   （把 controller 分配的 host slot 写进 node.mamba_host_value，挂入 mamba_host_lru_list）
→ node.backuped/host_value 就位; ongoing_write_through[node.id] 挂起, inc_lock_ref 保护
→ 完成回收: writing_check (sync CUDA event; ack_write_queue)  :385-430
   （scheduler 事件循环驱动: scheduler.py:3104-3105 check_hicache_events）
```

**copy 的内容**：`conv` 全部层 + `temporal` 全部层（`MambaPoolHost.backup_from_device_all_layer` 按 layer 循环 D2H）＋ FA-KV 页（KV 与 mamba 一起 backup，`write` 的主 payload 是 `node.value`）。**position 记录**：checkpoint 挂在 radix TreeNode 上，位置 = node 的 key 长度（即该前缀的 token 数），无显式 position 字段——树结构即 position。

LIVE 栈对应物：`MambaComponent.build_hicache_transfers`（`mamba_component.py:678`）/ `commit_hicache_transfer`（:758），由 UnifiedRadixCache 的 BackupKV action 驱动（`unified_tree_core.py:954-955` `_inc_hit_count_and_check → _build_backup_kv_action`）。

---

## 4. Retention / Eviction 路径（路径 3）

### 4.1 保存位置

| 层 | 位置 | 管理者 |
|---|---|---|
| device mamba | `MambaPool.temporal/conv` 的 slot | `MambaSlotAllocator` + radix `mamba_lru_list` |
| host mamba | `MambaPoolHost.temporal_buffer/conv_buffer` | `mamba_host_lru_list`（`HostLRUList`，`hi_mamba_radix_cache.py:62-94`） |
| 树记账 | `TreeNode.mamba_value`（device slot idx）/ `mamba_host_value`（host slot idx） | LRU: `mamba_evictable_size_/mamba_protected_size_` |

### 4.2 谁决定 eviction

- **FA-KV 满时**：`token_to_kv_pool_allocator.alloc` 失败 → scheduler `evict` 回调（`evictable_full_device_leaves` 堆按 `last_access_time` 弹出，`hi_mamba_radix_cache.py:711-743`）。
- **mamba slot 满时**：`_alloc_with_evict(self.req_to_token_pool.mamba_allocator, n, self.evict_mamba, ...)`（`hi_mamba_radix_cache.py:1075-1081` match-COW 分配、`:2157-2163` restore 分配；unified 版 `mamba_component.py:186-198` alloc 失败 → `cache.evict(EvictParams(num_tokens=0, mamba_num=1))`）。另外 admission 预留：`alloc_req_slots` 按 `MAMBA_STATE_PER_REQ_PREFIX_CACHE=3 / _LAZY=2 / _NO_CACHE=1` 为每 req 预留 mamba slot（`mem_cache/common.py:25-30`、`allocation.py:252-291`）。

### 4.3 真正释放 recurrent state 的函数

`_free_device_mamba`（`hi_mamba_radix_cache.py:529-542`）：`req_to_token_pool.mamba_allocator.free(node.mamba_value)` + 摘出 `mamba_lru_list` + `node.mamba_value = None`。三个上游：

```
evict(EvictParams)  :711-743
├─ full_num_tokens>0: _evict_device_leaf :695-709
│   ├─ backuped → _evict_to_host :544-562
│   │     cache_controller.evict_device(node.value)   [FA-KV D 释放]
│   │   + _free_device_mamba(node)                    [mamba 释放]
│   └─ 未 backuped:
│       ├─ write_back → write_backup(write_back=True) 后 _evict_to_host
│       └─ write_through → _evict_regular :564-601（连 host mamba 一起 free :580-584，整节点从树中删除）
└─ mamba_num>0: evict_mamba :798-840
    ├─ 内部节点: 仅 free GPU mamba（tombstone），FA-KV 留在 GPU
    │    req_to_token_pool.mamba_allocator.free + _tombstone_internal_node
    │    (mamba_radix_cache.py:1333; unified: mamba_component.py evict_component :308)
    └─ 叶子: 与 FA-KV 原子共存亡 → _evict_device_leaf
```

host 侧：`evict_host`（:745-759，host KV+mamba 一起删节点）、`evict_mamba_host`（:761-796，**只 free host mamba**，内部节点 tombstone / 叶子走 `_evict_host_leaf`）。

### 4.4 FA-KV 与 mamba 能否独立 eviction？

**能**。证据链：
1. TreeNode 四属性正交（§2.4），`evict_mamba` 对内部节点只 free mamba、KV 留 device（:818-824 注释 "Internal: free device mamba only, KV stays on device (tombstone)"）。
2. `evict_mamba_host` 只 free host mamba（:761-796）。
3. match 侧兼容半驱逐状态：`child.evicted and not child.backuped` 才断链（`:991` / `unified_tree_core.py:682`）；mamba validator 接受 device 或 host 上的 mamba（`mamba_component.py:137-141`）。
4. 限制：**叶子节点上两者原子绑定**（`evict_mamba` 叶子分支 assert `full_lock_ref == 0` 后走 `_evict_device_leaf`，KV 与 mamba 一起 demote/delete）。
5. 另有 unified 栈特有 per-path 配额：`mamba_max_states_per_path` → `_evict_excess_path_states`（`mamba_component.py:238-296`）。

---

## 5. Restore 路径（路径 4）

restore 分两个正交场景：

### 5.1 mamba 在 device 上（host backup 不涉及）——COW 复制进请求 slot

```
match_prefix(cow_mamba=True)（Req.init_next_round_input, schedule_batch.py:1310-1319;
  cow_mamba 默认 = tree_cache.supports_mamba()）
→ [LEGACY] _match_post_processor :1072-1084
   req.mamba_pool_idx = _alloc_with_evict(mamba_allocator, 1, evict_mamba)
   req.mamba_cow_src_index = mamba_node.mamba_value
   req.mamba_needs_clear = False
  [LIVE] MambaComponent.finalize_match_result_in_cache (mamba_component.py:173-202)
→ ScheduleBatch._collect_deferred_mamba_cow_and_clear (schedule_batch.py:2640-2660)
   → batch.mamba_cow_src_indices / mamba_cow_dst_indices / mamba_clear_indices
→ forward 前 model_runner._maybe_execute_deferred_mamba_cow_and_clear
   (model_runner.py:1478-1524, 调用点 :1575; "COW/clear only happen at prefix match on extend")
→ pool.mamba_pool.copy_from(translate(src), translate(dst))  :1518-1521
   copy_from = conv+temporal 全层 D2D copy + 重置 replayssm 游标 (memory_pool.py:939-986)
→ int8 变体（--enable-int8-mamba-checkpoint, 仅非 hierarchical 路径）:
   COW 源是 int8 checkpoint slot → mamba_ckpt_pool.load_to_active(mamba_pool, src, dst)
   (model_runner.py:1509-1515; 逆量化+conv cast, mamba_checkpoint_pool.py:253-259;
    checkpoint 侧写入 = store_from_active :245-251: 全 L 层 temporal int8 量化 + conv 原生 dtype;
    池结构 Int8CheckpointStore qdata[L,slots,H,d_v,d_k] int8 + per-channel scale,
    mamba_checkpoint_pool.py:60-138; 与 --enable-hierarchical-cache 互斥 server_args.py:6005-6022)
```

### 5.2 mamba 只在 host 上（mamba_evicted && mamba_backuped）——H2D load_back

```
match 结果: mamba_host_hit_length >= 1
  [LEGACY] _match_post_processor :1068-1070; [LIVE] mamba_component.py:164-169
→ req.needs_host_load_back() (schedule_batch.py:1177-1183: 三个 host_hit 之一 > 0)
→ PrefillAdder.add_one_req (schedule_policy.py:1159-1167)
   tree_cache.init_load_back(InitLoadBackParams(best_match_node, host_hit_length, req))
→ [LEGACY] init_load_back :351-374
   best_match_node.evicted 或 (mamba_evicted && mamba_backuped) → load_back(node, mem_quota, req)
→ load_back :254-349
   ① 收集 evicted 祖先链 nodes_to_load（KV host indices）
   ② mamba_restore_nodes = [last_hit_node] if mamba_backuped and mamba_evicted  :268-270
   ③ 小额/超配额 且无 mamba restore → 放弃（return None，回退到更浅前缀 :280-294）
     （即: mamba restore 需求会强制 load_back，即使 KV host indices 为空）
   ④ mamba_restore_transfers(last_hit_node, mamba_restore_nodes, req) :2128-2172
      - transfers[0]: MAMBA host_indices=cat(mamba_host_value), device None（controller 分配）
      - req.mamba_pool_idx 未分配时再补一条 host→req.mamba_pool_idx 的定向 transfer :2151-2170
   ⑤ cache_controller.load(host_indices, extra_pools=mamba_pools) :302
      （H2D，专用 load_stream: managers/cache_controller.py:299；
        实际逐层执行 start_loading → HostPoolGroup.load_to_device_per_layer
        (hybrid_cache_controller.py:492-548) → MambaPoolHost.load_to_device_per_layer
        (memory_pool_host.py:428-473) → JIT CUDA kernel transfer_kv_mamba_pf_lf
        (kernels/ops/mamba/transfer_mamba.py:41-60)；
        逐层流水同步 LayerDoneCounter (cache_controller.py:70-117)；
        失败先 evict 再重试 :307-318）
   ⑥ mamba_restore_commit(restored_nodes, transfers) :2174-2187
      把 controller 分配的 device slot 写回 node.mamba_value，挂回 mamba_lru_list :337-342
   ⑦ KV 部分: nodes_to_load 的 host_value 换回 device value :326-335
→ scheduler 侧: req.prefix_indices = cat([prefix_indices, new_indices]) (schedule_policy.py:1167)
→ 异步完成回收: loading_check (ack_load_queue, CUDA event sync) :432-464
   （metrics: increment_load_back_num_tokens / observe_load_back_duration :457-463）
```

[LIVE] 对应物：`MambaComponent.prepare_load_back`（`mamba_component.py:634-654`，含 dst slot 分配 assert）/ `finalize_load_back`（:656）。

**restore 哪个 checkpoint 由谁决定**：由 match walk 决定——`best_match_node` = 最深的「所有组件 validator 都通过」的节点（unified：`_update_best_if_valid`，`unified_tree_core.py:662-676`；LEGACY：`_match_prefix_helper` 中 `node.mamba_value is not None or node.mamba_backuped` 处更新，`hi_mamba_radix_cache.py:994-1015`）。即 **match 的终点本身就是 mamba checkpoint 边界**。

---

## 6. Replay 路径（路径 5，最重要）

### 6.1 场景设定

FA-KV 命中至 30K，最近的 recurrent checkpoint 在 20K（例如 20K 处节点有 `mamba_value`，20K-30K 节点 KV 在、mamba 被 `evict_mamba` tombstone，或从未产生）。新请求共享 0..30K 前缀。

### 6.2 实际执行链（已沿 definition→caller→caller's caller 逐步核实）

```
(1) Req.init_next_round_input (schedule_batch.py:1249)
    → match_prefix(MatchPrefixParams(key=token_ids, req, cow_mamba=True))
(2) match walk:
    [LEGACY] _match_prefix_helper (hi_mamba_radix_cache.py:977-1017)
      value[] 沿途收集全部 FA-KV（累计 30K）;
      仅在 node.mamba_value is not None or node.mamba_backuped 的节点更新
      best_value_len/best_last_node → best_value_len = 20000
    [LIVE] unified_tree_core._match_prefix_helper (unified_tree_core.py:624-709)
      value[] 同样累计 30K; _update_best_if_valid 要求 MambaComponent validator
      (mamba_component.py:137-141: mamba device 或 host 值存在) 通过
      → best_match_device_value_len = 20000; full_kv_hit_length = 30000 (:686)
(3) 截断 —— 关键一步:
    [LEGACY] value = value[:best_value_len] (hi_mamba_radix_cache.py:1086)
    [LIVE]   device_indices = cat(value[:best_match_device_value_len])
             (unified_tree_core.py:741-742)
    → req.prefix_indices = 20000 个 KV index（不是 30000!）
(4) branching 计算:
    [LEGACY] :1043-1056; [LIVE] mamba_component.py:152-171
    mamba_branching_seqlen = (30000 // mamba_cache_chunk_size) * chunk_size = 30000
    （"the longest page-aligned position that could've been cache hit if there
      exists a mamba state" — base_prefix_cache.py:188-190）
(5) COW restore（§5.1）: mamba@20K slot → req.mamba_pool_idx（forward stream 上 copy）
    [若 mamba 在 host: §5.2 load_back 先行, schedule_policy.py:1159-1167]
(6) PrefillAdder.add_one_req (schedule_policy.py:1079-1081)
    cand_extend_input_len = len(fill_ids) - len(prefix_indices) = N - 20000
    → **replay workload = 一个普通的 (N-20K) token prefill**，
      scheduler 无任何 mamba 特殊分支（截断已在 cache 内部完成）
(7) prepare_for_extend (schedule_batch.py:2281-2423)
    input_ids = fill_ids[len(prefix_indices):] (:2291) — 20K..N 的 token
    KV 分配: alloc_for_extend → out_cache_loc（20K..N 的**全新** KV slot）
    prefix KV 写 req_to_token: write_cache_indices (allocation.py:63-91)
    track entry 构建 (:2419-2423 → 2557-2638):
      若 mamba_branching_seqlen(30K) ∈ (prefix_len, mamba_track_seqlen) 且 chunk 对齐
      → 强制 mamba_track_seqlen = _force_track_h(30K), last_track_seqlen = 30000 (:2616-2631)
(8) forward (ForwardMode.EXTEND):
    - attention 层: 0..20K 的 KV 从树命中读取（不进 input_ids、不重算 QKV，
      只作为被 attend 的 KV）; 20K..N 的 token 重算 QKV 写入新 slot
    - mamba 层: 从 COW 恢复的 state@20K 起步，递推 20K..N
    - forward 途中在 30K 处快照 recurrent state 进 ping-pong slot
      （track_ssm_*, hybrid_linear_attn_backend.py:212-222/313-360）
(9) 请求完成 → cache_finished_req (§3.1):
    cache_len = mamba_last_track_seqlen（含 branching 强制的 30K 或更后的边界）
    → insert 时把 checkpoint 挂到对应位置节点; 树在 30K 处的 mamba 空洞被本次
      replay 顺带补上（自愈）
    重复 KV 行回收: 走树命中段上 req 新分配的 20K..30K 行被 free
    [LEGACY] _insert_helper free_segment (hi_mamba_radix_cache.py:896-903)
    [LIVE]   FreeDeviceKV([value_slice[dup_start:consumed_from]])
             (unified_tree_core.py:948-952)
```

### 6.3 四个关键问题的直接回答

| 问题 | 答案 | 证据 |
|---|---|---|
| replay token range 谁计算？ | 没有独立的 "replay range" 概念。cache 的 match 把 `prefix_indices` 截断到 mamba 边界（20K），scheduler 用既有公式 `len(fill_ids) - len(prefix_indices)` 得出 extend 范围 20K..N | `hi_mamba_radix_cache.py:1086` / `unified_tree_core.py:741-742`；`schedule_policy.py:1079-1081`；`schedule_batch.py:2291` |
| replay 是否走 Prefill？ | **是，完整走普通 extend prefill（ForwardMode.EXTEND）**，所有层（含 attention）都跑 20K..N。没有 layer-selective replay | `schedule_batch.py:2281+`；`model_runner.py:1484-1491`（COW 仅在 extend 触发） |
| FA-KV 已命中部分如何处理？ | 0..20K：作为 prefix KV 被 attend，不重算。**20K..30K：完全不复用**——重新计算 QKV 写入新 slot，insert 时新行被 free、树保留旧行。既有的 20K..30K FA-KV 对本请求零收益 | `allocation.py:63-91`；`hi_mamba_radix_cache.py:896-903` / `unified_tree_core.py:948-952` |
| scheduler 如何表示 workload？ | 一个普通的 (N−20K)-token prefill 请求 + 两个附加信号：`forward_batch.mamba_track_*`（30K 强制 track）与 `mamba_cow/clear_indices`（restore）。metrics 上 `#cached-token` 只计 20K | `forward_batch_info.py:442-450`；`schedule_batch.py:2405-2414` |

### 6.4 executable prefix 的确定：变量级追踪（Q1 变量表）

match 阶段各长度的存放位置与消费方式（两栈对照，符号 = LIVE / [LEGACY]）：

| 量 | 产生处 | 存放处 | 消费 |
|---|---|---|---|
| match 输入上限 `key_limit` | `schedule_batch.py:1281` → `_compute_max_prefix_len = input_len − 1`（:1367-1372；保证最后一个 token 一定被计算以产 logits） | `RadixKey(limit=)`（`radix_cache.py:62-85`，`_raw_len` 生效） | match walk 的 key 长度 |
| FA-KV matched length（未截断） | walk 中累计：`unified_tree_core.py:686 full_kv_hit_length += prefix_len` / [LEGACY] 未截断 `value` 列表总长（`hi:1046-1049` 计算 branching 时用） | `MatchResult.full_kv_hit_length`（LIVE）/ [LEGACY] 仅局部变量 | 计算 `mamba_branching_seqlen` |
| Mamba checkpoint matched position | walk 中只在该节点有 mamba 值时更新：`unified_tree_core.py:662-676 _update_best_if_valid`（Mamba validator `mamba_component.py:137-141`）/ [LEGACY] `hi:994-1015`（`node.mamba_value is not None or node.mamba_backuped`） | `best_match_node`（节点即位置：node.key 长度）+ `best_match_device_value_len` / [LEGACY] `best_value_len` | 截断长度 |
| `mamba_boundary_len` | `mamba_component.py:152`（= len(device_indices)+host_hit_length）/ [LEGACY] `hi:1086` 之前的 `best_value_len` | 局部 | branching 判据 |
| **executable prefix（最终）** | 截断：`unified_tree_core.py:741-742 device_indices = cat(value[:best_match_device_value_len])` / [LEGACY] `hi:1086 value = value[:best_value_len]` | **`req.prefix_indices`**（`schedule_batch.py:1324-1337` 解包） | 一切下游（extend 长度、req_to_token 写入、cached_tokens） |
| `mamba_branching_seqlen` | `mamba_component.py:154-162` / [LEGACY] `hi:1043-1056`（full hit 向下按 `mamba_cache_chunk_size` 对齐，>boundary 才非 None） | `req.mamba_branching_seqlen`（`schedule_batch.py:1333`） | 仅用于 replay forward 的强制 track（`schedule_batch.py:2616-2631`） |
| host 命中 | `mamba_host_hit_length`（`mamba_component.py:164-169` / `hi:1068-1070`） | `req.mamba_host_hit_length` | `needs_host_load_back()`（`schedule_batch.py:1177-1183`）→ init_load_back |

**结论（Q1）**：不存在单独的 "executable prefix length" 字段——`len(req.prefix_indices)` 本身就是它，且它在 match 内部已被 mamba 边界截断；FA-KV 的真实命中长度只以 `full_kv_hit_length`（LIVE）或局部变量（LEGACY）形式短暂存在，唯一持久化用途是 branching track。

### 6.5 具体数字例子（Context=30K, FA-KV cached=30K, mamba ckpt=20K）

设请求输入 30,000 tokens，树中 0..30K FA-KV 完整，最近 mamba checkpoint 在 20,000（`mamba_cache_chunk_size` 记为 C）：

```
input_len = 30000
key_limit = 30000 − 1 = 29999                    (schedule_batch.py:1369)
match walk: value 累计 29999, full_kv_hit_length = 29999
mamba 边界: best_match_device_value_len = 20000  (20K 节点有 mamba_value)
→ req.prefix_indices = 20000                     (utc:741 / hi:1086 截断)
→ mamba_branching_seqlen = (29999 // C) · C      (C=256 → 29696)
PrefillAdder: cand_extend_input_len = 30000 − 20000 = 10000   (schedule_policy.py:1079-1081)
             real_input_tokens = 10000 (无 host hit, :1088-1089)
req.set_extend_range(20000, 30000)               (schedule_policy.py:1196-1199)
prepare_for_extend: extend_lens = 10000, prefix_lens = 20000  (schedule_batch.py:2291-2297)
ForwardBatch: extend_seq_lens = 10000, extend_prefix_lens = 20000 (forward_batch_info.py:737-738)
forward: 处理 10,000 tokens, 从位置 20,000 起（含位置 29,999 以产首个输出 token）
cached_tokens += 20000（计 20K 而非 30K）        (schedule_batch.py:2389-2390:
                                                  new_cached = pre_len − already_computed)
replay forward 中: 强制 track @29696（branching 点, schedule_batch.py:2616-2631）
```

**答案（Q5）**：scheduler / model forward 实际处理 **10,000 tokens，从位置 20,000 开始**。原因：mamba recurrent state 不可由 KV 重构，只能从 checkpoint 起步串行重放；match 把 `prefix_indices` 截到 checkpoint 边界，使这 10K 落入普通 extend 范围；最后一个 token（29,999）必须被计算以产生 logits（故有 N−1 上限）。

### 6.6 Motivation 视角的结构性事实（本调查的核心产出）

1. **Gap = 纯浪费的 FA-KV**：mamba 边界(20K)到 FA-KV 边界(30K)之间已物化的 KV 无法为当前请求节省任何计算；replay 的代价是 **全模型 FLOPs**（不只 mamba 层）。
2. **自愈是顺带的**：replay forward 在 branching 点(30K)强制 track，把 checkpoint 空洞补齐——但只补这一个点，且只有当该请求真的被调度才发生。
3. **checkpoint 粒度由 `mamba_track_interval`（默认 256）决定**，且每次 extend forward 只 track 一个位置；树内 checkpoint 只在请求边界（finished/unfinished-chunked）落地。
4. **位置不对称**：FA-KV 是 per-token page 粒度复用；mamba 是 per-prefix 单 slot（一个 checkpoint = 树上一个节点挂一个 slot）。两者的边界错位是结构性的、由 eviction/insert 的不同步造成。
5. **eviction 独立性已有基础设施**（`evict_mamba`/`evict_mamba_host` 对内部节点独立 tombstone），但 insert 侧 mamba 只在请求边界发生——一快一慢造成 §6.1 场景。

---

## 7. 完整调用图

```
Request 到达
│
├─ scheduler.get_new_batch_prefill (scheduler.py:3228)
│    └─ Req.init_next_round_input (schedule_batch.py:1249)
│         └─ match_prefix(cow_mamba=True)                        [MATCH]
│              [LIVE] UnifiedRadixCache.match_prefix (unified_radix_cache.py:382)
│                → UnifiedTreeCore._match_prefix_helper (unified_tree_core.py:624)
│                → device_indices 截断@mamba边界 (utc.py:741)
│                → MambaComponent.finalize_match_result_in_tree_core (branching, mc.py:143)
│                → MambaComponent.finalize_match_result_in_cache (COW 标记, mc.py:173)
│              [LEGACY] HiMambaRadixCache.match_prefix (hi:958)
│                → _match_prefix_helper (hi:977) → _match_post_processor (hi:1019, 截断:1086)
│
├─ PrefillAdder.add_one_req (schedule_policy.py:1059)
│    ├─ needs_host_load_back? → init_load_back (sp.py:1159)      [RESTORE-H2D]
│    │    [LEGACY] load_back (hi:254) → mamba_restore_transfers (hi:2128)
│    │              → cache_controller.load → mamba_restore_commit (hi:2174)
│    └─ extend_len = fill - prefix(=mamba 边界)                   [REPLAY 范围隐式确定]
│
├─ ScheduleBatch.prepare_for_extend (schedule_batch.py:2281)     [STATE UPDATE]
│    ├─ input_ids = fill[prefix:] (2291)
│    ├─ alloc_for_extend + write_cache_indices (allocation.py:303/63)
│    ├─ _collect_deferred_mamba_cow_and_clear (2640)             [RESTORE-COW 标记]
│    └─ track entry + branching 强制 track (2419→2557-2638)      [CHECKPOINT 计划]
│
├─ model_runner forward (EXTEND)
│    ├─ _maybe_execute_deferred_mamba_cow_and_clear (mr.py:1478) [RESTORE-COW 执行]
│    │    mamba_pool.copy_from / ckpt_pool.load_to_active
│    ├─ conv/ssm kernels in-place 更新 (gdn_backend.py:563/668; mamba.py:577/632)
│    ├─ track 快照@30K (hybrid_linear_attn_backend.py:313-360)   [CHECKPOINT 产生(中途)]
│    └─ attention: prefix KV 命中部分只读不算; 20K..N 全重算
│
├─ decode 循环
│    ├─ 每步 in-place 更新 (selective_state_update / decode kernels)
│    ├─ kv_committed_len % mamba_track_interval == 0 → track     [CHECKPOINT 产生(decode)]
│    │    (batch_result_processor.py:1170-1195; mamba_lazy_prealloc schedule_batch.py:2856)
│    └─ scheduler 事件循环 check_hicache_events (scheduler.py:3104) [backup 完成回收]
│
├─ 请求完成/chunked
│    ├─ cache_unfinished_req (mamba_radix_cache.py:666)          [CHECKPOINT 入树]
│    └─ release_kv_cache → cache_finished_req (common.py:149 → mr.py:541)
│         donate ping-pong keep slot @mamba_last_track_seqlen
│         → insert → 尾节点挂 mamba_value (mamba_component.py:204)
│         → _inc_hit_count → write_backup                        [CHECKPOINT D2H]
│              (hi:381-383 → mamba_backup_transfers hi:2069
│               → cache_controller.write → mamba_backup_commit hi:2081)
│
├─ 内存压力
│    └─ evict (hi:711 / unified evict)
│         ├─ FA-KV 满: _evict_device_leaf → _evict_to_host(552) | _evict_regular(573)
│         │    └─ _free_device_mamba (hi:529)                    [EVICT device mamba]
│         ├─ mamba slot 满: evict_mamba (hi:798)                 [EVICT 仅 mamba]
│         │    内部节点 tombstone / 叶子与 KV 原子
│         └─ host 满: evict_mamba_host (hi:761)                  [EVICT host mamba]
│
└─ 下一个请求命中该前缀 → 回到 MATCH（若 mamba 已 tombstone → 边界回退 30K→20K → REPLAY）
```

（`hi:` = `hi_mamba_radix_cache.py`；`mr.py` = `mamba_radix_cache.py`；`mc.py` = `mamba_component.py`；`utc.py` = `unified_tree_core.py`；`sp.py` = `schedule_policy.py`）

---

## 8. Motivation Experiment 建议插桩位置

按「测什么」组织，全部为只读插桩（log/counter），不改语义：

### 8.1 Gap 分布（核心指标：mamba 边界与 FA-KV 边界的错位量）

| 插什么 | 哪里 | 记什么 |
|---|---|---|
| match 截断点 | `mamba_component.py:143-171 finalize_match_result_in_tree_core`（LIVE）/ `hi_mamba_radix_cache.py:1043-1056`（LEGACY） | `full_kv_hit_length`、`mamba_boundary_len`、`branching_seqlen`、`gap = full_kv_hit_length − boundary` |
| branching 自愈事件 | `schedule_batch.py:2616-2631` 强制 track 分支 | req id、branching 位置、extend 范围（replay 长度） |
| 实际 prefix | `schedule_batch.py:1324-1337`（match 解包处）或 `:2291` | `len(prefix_indices)` vs `req.mamba_branching_seqlen`（§6.5 端到端确认） |
| cached_tokens 口径 | `schedule_batch.py:2389-2390`（`new_cached = pre_len − already_computed`，pre_len 为截断后 prefix） | 与 `#cached-token` 输出对齐——应等于 20K 而非 30K |

### 8.2 Replay 代价

| 插什么 | 哪里 | 记什么 |
|---|---|---|
| replay workload | `schedule_policy.py:1079-1081` | `cand_extend_input_len`（= replay 长度），关联 `mamba_branching_seqlen` 区分「正常 extend」与「gap replay」 |
| 重算后丢弃的 KV | `unified_tree_core.py:948-952 FreeDeviceKV` / `hi_mamba_radix_cache.py:896-903 free_segment` | 被.free 的新行数（= 对应树已有 KV 的重复计算量） |
| cached_tokens 分解 | `schedule_batch.py:2405-2414` | device/host/storage 分解已内置，直接采信 |

### 8.3 Checkpoint 生命周期

| 插什么 | 哪里 | 记什么 |
|---|---|---|
| decode track 触发 | `batch_result_processor.py:1170-1195 _mamba_check_track_boundary` | 触发时刻、track_seqlen |
| checkpoint 入树 | `mamba_component.py:204-236 commit_insert_component_data` / `mr.py:628-635 insert` | 位置（key 长度）、slot、来源（donate/copy/int8） |
| D2H backup 字节与时延 | `memory_pool_host.py:475-520 backup_from_device_all_layer` + `hi:457-463 loading_check` 内置 metrics（`increment_load_back_num_tokens`/`observe_load_back_duration` 对 H2D） | bytes、duration、node 位置 |
| H2D restore | `memory_pool_host.py:428-473 load_to_device_per_layer` | bytes、duration |

### 8.4 Eviction 压力

| 插什么 | 哪里 | 记什么 |
|---|---|---|
| 仅-mamba eviction | `hi_mamba_radix_cache.py:798-840 evict_mamba` / `mamba_component.py:308 evict_component` | 内部节点 tombstone 次数、位置、触发者（match-COW alloc vs admission 预留 vs restore alloc） |
| host mamba eviction | `hi:761-796 evict_mamba_host` | 次数、位置 |
| tombstone 后的 gap 生成率 | 在 evict_mamba 处记 node key_len，与 §8.1 的 gap 分布对齐 | 因果链：evict@P → 后续命中 gap=P−边界 |

### 8.5 COW/restore 时延

| 插什么 | 哪里 | 记什么 |
|---|---|---|
| COW copy | `model_runner.py:1505-1521` | src/dst slot、（可选）event 计时 |
| host load_back | `schedule_policy.py:1159-1167`（发起）+ `hi:432-464 loading_check`（完成） | 与 §8.3 H2D 合并观测 |

### 8.6 建议的最小运行验证（对应 NEEDS_RUNTIME_VALIDATION）

1. **两请求共享 32K 前缀**（hybrid-SSM 模型，如 Qwen3-Next）：req2 的 `#cached-token` 应等于 mamba 边界而非 FA-KV 命中长度——直接验证 §6.2(3)。
2. **人为制造 gap**：等 req1 完成后压力触发 `evict_mamba`（或调小 mamba pool），再发 req2，观察 replay 长度 = FA-KV 命中 − 幸存 checkpoint 边界。
3. **branching 自愈**：req2 完成后检查树在原 branching 点是否出现新 mamba checkpoint（log insert 位置）。
4. **impl 确认**：启动日志 `Tree cache initialized: ... impl=`（`registry.py:233-242`）确认 UnifiedRadixCache。

---

## 9. NEEDS_RUNTIME_VALIDATION 清单

> **2026-08-19 更新：#1/#2/#5 已由一次最小 runtime validation 关闭（VERDICT PASS）**，详见
> `flowstate/motivation/artifacts/runtime_validation_gap_replay_20260819/`（findings.md + server_full.log + instrumentation.patch）。
> 实测（Qwen3.5-9B, v0.5.17 镜像, extra_buffer, `--max-mamba-cache-size 16`）：启动日志 `impl=UnifiedRadixCache hybrid_ssm=True`（#1✅）；
> 容量 LRU 逐出深 mamba ckpt（internal 节点 tombstone，16 次 `mamba_evict fa_kept=True`，#5✅）后，R2 请求
> `full_kv_hit=32767 / exec_prefix=16384 / gap=16383`，extend_len=16384 全量 re-prefill，insert 时旧 FA 行
> `FreeDeviceKV` 释放 16320 行（#2✅），branching forced-track@32704 自愈新 ckpt（R3 探测 exec=32704，仅 64-token extend）。

| # | 结论 | 静态依据 | 验证方法 | 状态 |
|---|---|---|---|---|
| 1 | v0.5.17 默认 dispatch 下 LIVE 实现是 UnifiedRadixCache+MambaComponent；HiMambaRadixCache 不可达 | `registry.py:114-115`；全树无实例化点；`server_args.py:6007` 注释与之矛盾 | 启动日志看 `impl=`（`registry.py:233-242`） | ✅ 2026-08-19 |
| 2 | replay 时 20K..30K FA-KV 对当前请求零复用、重算后被 free | `hi:896-903`/`utc:948-952` 静态读 | §8.6-1/2：对比 `#cached-token` 与观测 FreeDeviceKV 行数 | ✅ 2026-08-19（gap=16383, extend=16384, freed=16320） |
| 3 | `mamba_track_interval` 默认 256 且 decode 每 boundary 产生 track | `server_args.py:2518`；`batch_result_processor.py:1184-1188` | log `_mamba_check_track_boundary` 触发序列 | 未验 |
| 4 | 每次 extend forward 至多一个 track 位置（branching 可抢占该位置） | `schedule_batch.py:2572-2632` 单 entry 结构 | log track entry per extend | ✅（佐证）2026-08-19：R2 单次 extend 仅 branching 抢占 track@32704 |
| 5 | `evict_mamba` 内部节点分支真的保留 FA-KV | `hi:818-824` assert 与注释 | 压力实验后检查节点 `evicted=False, mamba_evicted=True` | ✅ 2026-08-19（tombstone 后 full_kv_hit=32767 仍全命中） |
| 6 | COW copy 发生在 forward stream、extend-only | `model_runner.py:1484-1491` | torch profiler / cuda event | 未验 |
| 7 | mamba_size（`max_mamba_cache_size`）数值来源 | `kv_cache_configurator.py:779`（运行时求解） | `--debug-memory-pool` 启动日志（`memory_pool.py:788-814` 打印 conv/ssm size） | 未验 |

---

## 附：与既有实验的衔接

- exp-2 harness 的 offload/restore RPC（`exp-2/harness/offload_rpc.py`）操作的是 §2.3 末尾的 `get_cpu_copy/load_cpu_copy` 血缘（进程内显式控制），与本文的树内自动 checkpoint（§3）是**平行机制**：前者 bypass radix，后者走 TreeNode。FlowState 的 Motivation 实验应盯后者。
- 2026-08-16 hybrid consistency 审计发现的 "component lifecycle versioning drift" 与本文 §2.4 四态正交性 + §4.4 叶子原子性约束直接相关：独立 eviction 的自由度只存在于内部节点。
