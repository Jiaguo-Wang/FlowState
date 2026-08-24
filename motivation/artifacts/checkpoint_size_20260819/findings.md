# WP1 — Recurrent Checkpoint Size Characterization

本报告只回答 SGLang v0.5.17 + Qwen3.5-9B 的 recurrent-state/checkpoint HBM 大小与 slot 语义；没有测 latency、replay、Host Offload，也没有修改 checkpoint policy 或实现 FlowState。

结论先行：当前普通（non-int8）state 是 mixed dtype——temporal 为 FP32、conv 为 BF16。一个完整 slot、一个 active recurrent state、一个 no-overlap temporary state、以及一个普通 retained checkpoint 的逻辑 payload 都是：

```text
51,511,296 bytes = 49.125 MiB
```

它们是不同的 slot ownership/生命周期角色，不是同一时刻的四份固定副本。启动日志中的 `4 state slots per request` 是并发容量规划口径，不是 checkpoint size。

## 1. Environment

| Item | Value |
|---|---|
| SGLang | `v0.5.17`, image `lmsysorg/sglang:v0.5.17-cu129-runtime`, build commit `29481685462732237d80d86076d6563e1f658102` |
| Model | `/home/wjg/models/qwen3.5-9b`, `Qwen3_5ForConditionalGeneration` |
| GPU | NVIDIA H100 PCIe, nominal 80 GiB HBM |
| Parallelism | TP=1, PP=1 |
| Cache implementation | `UnifiedRadixCache + MambaComponent`, hierarchical cache off |
| Mamba settings | `extra_buffer`, `disable_overlap_schedule=True`, `max_mamba_cache_size=16`, `mamba_track_interval=256`, `mamba_max_states_per_path=-1`, int8 checkpoint off |
| Other relevant settings | `enable_linear_replayssm=False`, page-major/envelope layout off |

本轮新做了一次最小 runtime probe：只启动 server，没有运行本任务的 benchmark、replay 或用户 workload。`launch_server` 默认的内置 warmup/readiness check 在启动后执行过一次 80-token prefill（`server_probe.log:684-686`）；Tensor probe 已在其之前的 cache-allocation 阶段完成，且本报告不使用该 warmup 的 latency。一次性 instrumentation 只读取并打印 Tensor 元数据，不改变 Tensor 或执行路径：

- raw log：`flowstate/motivation/artifacts/checkpoint_size_20260819/server_probe.log`
- patch：`flowstate/motivation/artifacts/checkpoint_size_20260819/runtime_tensor_probe.patch`
- instrumentation 位置：`MambaPool.__init__` 的普通 allocation-log 分支之后，`python/sglang/srt/mem_cache/memory_pool.py` 原始约 `:812-814`；每个 MambaPool construction 只打印一次
- runtime 参数：`server_probe.log:11`
- 模型实际类型：`server_probe.log:663`
- probe 数值：`server_probe.log:664-668`
- 实际 tree-cache implementation：`server_probe.log:677`
- Uvicorn startup：`server_probe.log:682`；内置 warmup 与 SGLang ready：`:684-686`

该结果还与此前同一 SGLang 版本、同一模型 config 的真实 state fingerprint artifact 交叉一致：`artifacts/exp2-b-PASS-20260812T130323Z/strict_L2048.json:2023-2056` 给出 temporal `50,331,648` bytes、conv `1,179,648` bytes、总计 `51,511,296` bytes；其逐层 runtime shape/dtype 从 `:2057`（temporal layer 0）持续到 `:2657`（conv layer 23）。

下文未写绝对路径的 SGLang 源码引用均相对于冻结快照 `/tmp/sglang0517_src/sglang/srt/`。

## 2. Runtime State Layout

### 2.1 Number of Linear/Recurrent Layers

模型 config 有 32 层，`full_attention_interval=4`；每 4 层为 3 个 linear-attention + 1 个 full-attention，因此：

```text
linear/recurrent layers = 32 × 3/4 = 24
full-attention layers   = 32 × 1/4 = 8
```

这不是仅由配置名字推断：

1. `Qwen3_5TextConfig` 继承 `Qwen3NextConfig`（`configs/qwen3_5.py:15-29`）。
2. Qwen3.5 outer config 把 `text_config` 构造成该类型（同文件 `:78-103`）。
3. hybrid dispatch 取得 text config，并走 GDN/mambaish 分支（`configs/hybrid_arch.py:48-60,116-126`）。
4. `Qwen3NextConfig.mamba2_cache_params` 明确返回 `Mamba2CacheParams(shape=Mamba2StateShape, layers=linear_layer_ids, ...)`（`configs/qwen3_next.py:259-277,287-313`），而不是 `KimiLinearStateShape`。
5. configurator 把这些 params 和 PP-filtered layer IDs 传入 `HybridReqToTokenPool`（`mem_cache/kv_cache_configurator.py:777-800`）；TP=1/PP=1 保留全部 24 层。
6. runtime 直接打印 `num_mamba_layers=24`（`server_probe.log:666`）。

### 2.2 Temporal State

Qwen3.5 的实际 builder 输入为：

```text
num_heads = linear_num_value_heads = 32
head_dim  = linear_value_head_dim  = 128
state_size = linear_key_head_dim   = 128
TP = 1
```

`Mamba2StateShape.create` 的公式是 `(num_heads / TP, head_dim, state_size)`（`configs/mamba_utils.py:188-237`），所以每层、每 slot：

```text
temporal_state_shape = (32, 128, 128)
维度含义 = (value heads per TP, value-head dim, key/state dim)
dtype = torch.float32
```

dtype resolver 默认 FP32，读取 config 的 `mamba_ssm_dtype=float32`，再允许环境变量覆盖（`configs/mamba_utils.py:63-107`）。本轮 runtime 最终确认是 `torch.float32`，不是从 config 猜测。

整个预分配 Tensor 包含 24 层与 `16 + reserved slot 0`：

```text
shape        = (24, 17, 32, 128, 128)
dtype        = torch.float32
numel        = 213,909,504
element_size = 4 bytes
total bytes  = 855,638,016 = 816 MiB
```

证据：`server_probe.log:667`。分配代码读取 `cache_params.shape.temporal/dtype.temporal` 并构造 `(num_mamba_layers, size + 1) + temporal_shape`（`mem_cache/memory_pool.py:457-480,564-568`）。

### 2.3 Conv State

GDN conv state 是 packed `Q + K + V` window：

```text
Q = 16 key/query heads × 128 = 2,048
K = 16 key heads       × 128 = 2,048
V = 32 value heads     × 128 = 4,096
conv_dim = 2,048 + 2,048 + 4,096 = 8,192
window = linear_conv_kernel_dim - 1 = 4 - 1 = 3
```

这与 builder 的 `intermediate_size + 2 × n_groups × state_size` 公式一致（`configs/qwen3_next.py:294-309`; `configs/mamba_utils.py:200-227`）。每层、每 slot：

```text
conv_state_shape = (8192, 3)
维度含义 = (packed Q/K/V channels per TP, conv history window)
dtype = torch.bfloat16
```

整个 `mamba_cache.conv` 只有一个 Tensor，即 `conv[0]`：

```text
shape        = (24, 17, 8192, 3)
dtype        = torch.bfloat16
numel        = 10,027,008
element_size = 2 bytes
total bytes  = 20,054,016 = 19.125 MiB
```

证据：`server_probe.log:668`。分配代码位于 `mem_cache/memory_pool.py:540-547`。

## 3. Active Recurrent State Size

### Theoretical Calculation

一个 active state 包含同一 slot 上全部 24 个 recurrent layers 的 temporal + conv；不包含模型权重、full-attention KV、CUDA workspace，也不包含已关闭的 ReplaySSM ring。

```text
temporal_bytes_per_slot
  = 24 × 32 × 128 × 128 × 4
  = 50,331,648 bytes
  = 48 MiB

conv_bytes_per_slot
  = 24 × 8192 × 3 × 2
  = 1,179,648 bytes
  = 1.125 MiB

active_state_bytes
  = 50,331,648 + 1,179,648
  = 51,511,296 bytes
  = 49.125 MiB
  = 0.0479736328125 GiB
```

源码自带的 `mamba_cache_per_req` 也按 `shape product × dtype itemsize × number of layers` 计算（`configs/mamba_utils.py:110-125`）。

### Runtime Tensor Calculation

runtime Tensor 的 slot 轴是 17，因此直接用 `numel() × element_size()` 后除以 17：

```text
temporal = 213,909,504 × 4 / 17
         = 50,331,648 bytes = 48 MiB

conv     = 10,027,008 × 2 / 17
         = 1,179,648 bytes = 1.125 MiB

total    = 51,511,296 bytes = 49.125 MiB
```

probe 还直接打印了 `per_slot_numel` 和 `per_slot_bytes`（`server_probe.log:667-668`），所以这里没有用启动日志的两位小数反推精确值。

### Cross Validation

| Component | Shape × dtype | Runtime `numel × element_size / 17` | Match |
|---|---:|---:|---:|
| Temporal | 50,331,648 B | 50,331,648 B | exact |
| Conv | 1,179,648 B | 1,179,648 B | exact |
| Total | 51,511,296 B | 51,511,296 B | exact |

active execution 从 `req_index_to_mamba_index_mapping` 取得一个 `mamba_cache_indices`（`layers/attention/hybrid_linear_attn_backend.py:93-102`），GDN 再用同一 index 读写该层的 conv 与 temporal（`layers/attention/linear/gdn_backend.py:379-437`），所以 active state 不是两个独立 slot。

## 4. Extra-buffer / Temporary State

当前 `disable_overlap_schedule=True` 被传为 `enable_overlap_schedule=False`（`mem_cache/kv_cache_configurator.py:792-797`），因此：

```text
mamba_ping_pong_track_buffer_size = 1
```

源码在 `mem_cache/memory_pool.py:1170-1172` 直接给出该分支；普通 non-lazy `extra_buffer` 随后恰好 `alloc(1)`（同文件 `:1400-1423`）。track 时 conv 与 temporal 都写到这个 tracked destination（`layers/attention/hybrid_linear_attn_backend.py:791-822`; `layers/attention/linear/gdn_backend.py:555-561,698-701`）。

因此当前配置的 request-owned steady state 是：

| Role | Slots/request | Footprint |
|---|---:|---:|
| Mutable active | 1 | 49.125 MiB |
| Temporary tracked/extra buffer | 1 | 49.125 MiB |
| Steady private total | 2 | 98.25 MiB |

`extra_buffer` 的 temporary 是完整 state slot，不是只存 temporal 或只存 conv。

启动时的 `4 state slots per request`（`server_probe.log:664`）必须分开解释。静态 concurrency solver 的 ratio 是 base `3` 加 no-overlap additional `1`（`mem_cache/kv_cache_configurator.py:113-123,1659-1688`），再用 `max_mamba_cache_size // ratio` 限制并发（同文件 `:1734-1750`）。它覆盖 hit/COW/unfinished-cache 的 worst-case non-reclaimable/transitional working set：可能同时存在仍锁住的 incoming retained checkpoint、active、当前 temporary，以及 donation 前先分配的 replacement slot。它不是四个常驻 private slots，更不是“一个 checkpoint = 四个 states”。当前 `mamba_max_states_per_path=-1` 时，更早的 retained checkpoints 是独立、可 LRU eviction 的历史 occupancy，单条 path 的历史总数可超过 4（`mem_cache/unified_cache/components/mamba_component.py:253-261`）；`4` 只约束 running-request capacity charge，并不限制历史 checkpoint 数。

另有动态 admission 的 `3 × requests` eviction-headroom heuristic（`mem_cache/common.py:26-30`; `mem_cache/allocation.py:264-284`）；这同样是调度保守量，不改变物理 temporary 数量为 1。

## 5. Retained Recurrent Checkpoint Size

### Lifecycle Evidence

当前 runtime 路径是 `UnifiedRadixCache + MambaComponent`，不是 legacy `HiMambaRadixCache`（`server_probe.log:677`）。普通 non-int8 checkpoint 的生命周期如下：

1. **CREATE / active alloc**：cache miss 时给 `req.mamba_pool_idx` 分配一个 active slot；extra-buffer 同时给 request 分配一个 tracked slot（`mem_cache/memory_pool.py:1294-1341,1400-1423`）。slot allocator 只管理 ID `1..size`，slot 0 保留（`mem_cache/allocator/mamba.py:30-35,74-97`）。
2. **TRACK**：在 boundary 把 temporal 与 conv 的快照写入该 single temporary slot（上述 track 代码路径）。
3. **DONATE / associate with radix node**：finished request 从 ping-pong buffer 取一个 one-element slot index；int8 关闭时直接设为 `insert_params.mamba_value`（`mem_cache/unified_cache/components/mamba_component.py:556-571`）。unfinished request 会先分配 replacement，再把旧 tracked slot donate 给树（同文件 `:582-617`; `mem_cache/memory_pool.py:1438-1461`）。
4. **RETAIN**：新 radix node 直接保存这个 `mamba_value`，并按 `len(mamba_value)=1` 计入 Mamba LRU/evictable size（`mem_cache/unified_cache/components/mamba_component.py:219-251`）。finished cleanup 释放 active，但在 insert 成功时保留 chosen tracked slot；本配置 buffer size=1 时该 slot 不被 free（同文件 `:619-652`; `mem_cache/memory_pool.py:1463-1508`）。
5. **RESTORE / reuse**：prefix hit 不原地修改 retained source；它另分配 active destination 并记录 COW（`mem_cache/unified_cache/components/mamba_component.py:188-217`），forward 前 `MambaPool.copy_from` 复制全部 conv + temporal 到 active slot（`model_executor/model_runner.py:1478-1521`; `mem_cache/memory_pool.py:939-978`）。
6. **FREE / evict**：Mamba component tombstone node 的单个 value（`mem_cache/unified_cache/components/mamba_component.py:323-366`）；Unified cache 把 `device_frees` 转成 `FreeComponentDeviceSlot`（`mem_cache/unified_radix_cache.py:443-469,840-847`），Mamba component 最终调用 `_free_mamba_value`，把该 ID 返回同一个 `MambaSlotAllocator`（`mem_cache/unified_cache/components/mamba_component.py:525-529,920-937`）。已有 runtime eviction 日志直接显示 `freed=1` 且 FA-KV 仍保留（`flowstate/motivation/artifacts/runtime_validation_gap_replay_20260819/server_full.log:768,775,782`）。

### Per-checkpoint Footprint

因此，在当前 `extra_buffer`、`enable_int8_mamba_checkpoint=False` 下：

```text
1 retained checkpoint
  = 1 ordinary MambaPool slot
  = all 24 layers' temporal + conv
  = 51,511,296 bytes
  = 49.125 MiB
```

“ordinary/full-precision”在这里表示没有 int8 checkpoint quantization；实际 native layout 是 FP32 temporal + BF16 conv。

## 6. Int8 Checkpoint Size

本轮按要求没有启动 int8 serving。以下由已经 runtime-verified 的 Qwen3.5 shape/dtype 和 int8 pool 的真实源码 layout 严格计算。

`Int8CheckpointStore` 的 layout 是：

```text
qdata [L, slots, H, d_v, d_k]     int8
scale [L, slots, H, 1,   d_k]     temporal dtype
conv  [L, slots, 8192, 3]         conv dtype
```

证据：`mem_cache/mamba_checkpoint_pool.py:60-113,186-225`；静态 estimator 的同一公式位于 `:261-293`。当前真实 dtype 决定 scale 为 FP32、conv 为 BF16。

| Component | Formula per checkpoint | Bytes | MiB |
|---|---:|---:|---:|
| qdata | `24 × 32 × 128 × 128 × 1` | 12,582,912 | 12.000 |
| scale | `24 × 32 × 1 × 128 × 4` | 393,216 | 0.375 |
| conv | `24 × 8192 × 3 × 2` | 1,179,648 | 1.125 |
| **total** | qdata + scale + conv | **14,155,776** | **13.500** |

```text
1 int8 checkpoint = 14,155,776 bytes
                    = 13.5 MiB
                    = 0.01318359375 GiB
```

该 footprint 是普通 checkpoint 的 `27.48%`，节省 `72.52%`；在只比较 checkpoint payload 的同等 HBM 下，理论容量约为 `3.64×`。该数字不包含 int8 checkpoint pool 的 reserved slot 0；和普通 checkpoint 一样，它是 per-usable-checkpoint payload。

## 7. Capacity Scaling

普通 retained checkpoint 的理论 footprint：

```text
bytes_per_checkpoint = 51,511,296
total_bytes(N) = N × 51,511,296
HBM_percent(N) = 100 × total_bytes(N) / (80 × 2^30)
```

| checkpoints | total bytes | footprint | % of H100 80 GiB |
|---:|---:|---:|---:|
| 1 | 51,511,296 | 49.125 MiB = 0.047974 GiB | 0.0600% |
| 8 | 412,090,368 | 393 MiB = 0.383789 GiB | 0.4797% |
| 16 | 824,180,736 | 786 MiB = 0.767578 GiB | 0.9595% |
| 32 | 1,648,361,472 | 1,572 MiB = 1.535156 GiB | 1.9189% |
| 64 | 3,296,722,944 | 3,144 MiB = **3.0703125 GiB** | **3.8379%** |
| 128 | 6,593,445,888 | 6,288 MiB = **6.140625 GiB** | **7.6758%** |
| 256 | 13,186,891,776 | 12,576 MiB = **12.28125 GiB** | **15.3516%** |
| 512 | 26,373,783,552 | 25,152 MiB = 24.5625 GiB | 30.7031% |

这张表只描述 N 个 checkpoint payload，按要求不加入模型权重、full-attention KV、CUDA context 或 workspace。它是扩容外推；当前配置只有 16 个 usable Mamba slots，不能实际同时容纳表中大于 16 的 state slots，除非扩大 pool。

## 8. Preallocation vs Occupancy

### Reserved Capacity

`MambaPool` 在启动时一次性分配 `(num_layers, size + 1, ...)` Tensor（`mem_cache/memory_pool.py:540-568`）。`max_mamba_cache_size=16` 时 backing 为 17 slots，因为 slot 0 是 padded-token dummy write target（`mem_cache/allocator/mamba.py:93-97`）。

本轮 runtime 的实际 backing allocation：

| Tensor | 17-slot backing bytes | Footprint |
|---|---:|---:|
| Temporal | 855,638,016 | 816 MiB = 0.796875 GiB |
| Conv | 20,054,016 | 19.125 MiB = 0.018677 GiB |
| **Total** | **875,692,032** | **835.125 MiB = 0.815552 GiB** |

这精确解释启动日志粗粒度的 `ssm_state size: 0.80GB`、`conv_state size: 0.02GB`（`server_probe.log:665`）。

其中 16 个 usable slots 的逻辑 payload capacity 是 `824,180,736 bytes = 786 MiB`；额外的 reserved slot 0 是 `49.125 MiB`，两者相加才是上表的 17-slot backing `835.125 MiB`。

### Occupied Slots

`MambaSlotAllocator.free_slots` 初始化为 `1..16`；alloc 从 free list 移除 ID，free 再放回（`mem_cache/allocator/mamba.py:74-97`）。运行时的 active、temporary、retained checkpoint 都只是对这块已分配 backing Tensor 中 slot 的 ownership。

所以正确表述是：

> A retained checkpoint consumes one preallocated Mamba-state slot and reduces available recurrent-state capacity by one slot. Its logical payload is 49.125 MiB, but checkpoint creation does not necessarily increase `torch.cuda.memory_allocated()` because the backing pool was reserved at server initialization.

反过来，eviction/free 归还 slot capacity，也不一定立刻降低 PyTorch CUDA allocation。在正常 checkpoint create/free 生命周期中 backing allocation 保持不变；需要调整、释放或重建 pool backing，才会改变这块预分配 Tensor 本身。

## 9. Key Findings

1. Qwen3.5-9B 在该 runtime 实际使用 `Mamba2CacheParams + Mamba2StateShape`，共 24 个 recurrent/GDN layers。
2. 一个 native state slot 是 FP32 temporal `48 MiB` 加 BF16 conv `1.125 MiB`，合计 `49.125 MiB`；理论与 runtime Tensor 逐 byte 相等。
3. 当前 no-overlap `extra_buffer` 的 steady temporary 数量是 1 slot/request；active + temporary 是 `98.25 MiB/request`。
4. 一个普通 retained checkpoint 是一个完整、预分配的 MambaPool slot，即 `49.125 MiB`；`4 slots/request` 是容量规划，不是 checkpoint size。
5. int8 checkpoint 的严格静态大小是 `13.5 MiB`，其中 scale 必须按当前 FP32 temporal dtype 计算。
6. 64/128/256 个普通 checkpoints 单独占 `3.07/6.14/12.28 GiB`；在固定 slot pool 中，active、temporary 与 retained checkpoints 竞争同一有限容量。

## 10. Answers to Q1–Q5

**Q1. 一个 active recurrent state 到底多大？**

`51,511,296 bytes = 49.125 MiB`。其中 temporal `48 MiB`（FP32），conv `1.125 MiB`（BF16）。这是 24 个 recurrent layers 的完整 state。

**Q2. 当前 extra_buffer 策略下，一个 retained recurrent checkpoint 到底多大？**

`51,511,296 bytes = 49.125 MiB`。它占一个 ordinary `MambaPool` slot；不是 4 slots。当前 no-overlap request 另有 1 个 active slot和 1 个 steady temporary slot，这些是不同角色。

**Q3. 一个 int8 checkpoint 理论多大？**

`14,155,776 bytes = 13.5 MiB`：qdata `12 MiB` + FP32 scale `0.375 MiB` + BF16 conv `1.125 MiB`。这是源码 + runtime-verified shape/dtype 的静态计算；按要求没有启动 int8 benchmark。

**Q4. 64 / 128 / 256 checkpoints 分别需要多少 GiB HBM？**

```text
64  =  3.0703125 GiB  (3.8379% of nominal 80 GiB)
128 =  6.140625  GiB  (7.6758%)
256 = 12.28125   GiB  (15.3516%)
```

**Q5. 这些结果是否说明 recurrent checkpoint memory 足以成为一个需要优化的有限资源？**

是，但结论应限定为“显著的容量资源”，而不是“已经证明是端到端瓶颈”。单个 checkpoint 只有约 49 MiB，64 个也只是 80 GiB 的 3.84%；但到 128/256 个就分别是 6.14/12.28 GiB（7.68%/15.35%），已经不可忽略。更直接的是，本轮配置只有 16 个 usable Mamba slots，并且并发 Agent 的 active/temporary、prefix hit 的锁定 state、多个 workflow boundaries 的 retained checkpoints 共同竞争这些 slots；容量压力会在达到 GPU 总 HBM 百分比之前先以 slot exhaustion/LRU eviction 出现。因此 checkpoint memory 是合理且可量化的优化对象，但 WP1 数字本身不证明某种具体 policy、int8 或 offload 一定带来端到端收益。

本轮没有剩余 `NEEDS_RUNTIME_VALIDATION`：native runtime Tensor 已在当前配置下直接验证；int8 项按任务要求是源码/shape 静态计算，而非待补的 serving benchmark。
