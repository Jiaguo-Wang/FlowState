# Runtime Validation: Physical FA-KV coverage > Executable Hybrid Prefix ⇒ gap 走完整 EXTEND/re-prefill

**日期**: 2026-08-19 · **VERDICT: PASS（全部 4 项预期验证通过）**

## 环境

| 项 | 值 |
|---|---|
| Docker image | `lmsysorg/sglang:v0.5.17-cu129-runtime` (1dfce9033670) |
| SGLang source commit | 29481685462732237d80d86076d6563e1f658102 (editable, /sgl-workspace/sglang/python) |
| 模型 | Qwen3.5-9B (`Qwen3_5ForConditionalGeneration`, hybrid GDN+FA) |
| GPU | 1× H100 PCIe (device=1) |
| 关键 server args | `--max-mamba-cache-size 16 --mamba-radix-cache-strategy extra_buffer --context-length 45056 --mem-fraction-static 0.40 --chunked-prefill-size 45056 --disable-cuda-graph --disable-overlap-schedule --page-size 1`（无 hierarchical cache） |
| Cache impl 确认 | 启动日志：`Tree cache initialized: source=default impl=UnifiedRadixCache hybrid_swa=False hybrid_ssm=True hierarchical=False streaming_wrapped=False` ✅（静态分析结论 NEEDS_RUNTIME_VALIDATION #1 关闭：is_hybrid_ssm 走 UnifiedRadixCache，非 MambaRadixCache/HiMambaRadixCache） |
| 派生参数 | `mamba_track_interval=256`，mamba chunk=128（由 branching=32768//128*128=32704 推得），`max_total_num_tokens=403915`（无 FA 容量压力） |

## 人为制造 recurrent checkpoint gap（未修改任何 matching/replay 语义）

纯容量+LRU 自然路径（全部走既有接口）：

1. **S1**（16K head of P32）→ 浅层 ckpt@16384。
2. **R1**（P32, 32K）→ 深层 ckpt@32704（chunk 边界，node 18）。
3. **EXT**（P32+2048 unique）→ 使 R1 的 ckpt 节点变 **internal**（关键：mamba LRU 逐出到 internal 节点 = tombstone 只释放 mamba value，FA 行保留；若为 leaf 则整节点删除）。
4. **F00–F23**（HEAD+2048 unique ×24）→ mamba pool（16 slots）耗尽 → 每个新请求的 COW `alloc(1)` 失败 → `evict(mamba_num=1)` → LRU 尾逐出。日志证实 **16 次 `mamba_evict ... fa_kept=True`**，首个被逐 = node 18（R1 的深 ckpt）。filler 每次 match 都刷新 S1 的 ckpt（MATCH_END reset MRU），故浅层 ckpt 存活。
5. **R2**（同 P32）→ **测量对象**。
6. **R3**（同 P32，探测）→ 测 R2 自愈后的新 ckpt 位置。

## 结果表（来自只读 [FSVAL] 探针，server_full.log）

| Req | FA-KV physical hit | Executable prefix | Gap | Extend tokens | Duplicate KV freed | New ckpt position |
|---|---|---|---|---|---|---|
| S1 (build) | 0 | 0 | – | 16384 | – | 16384（由 R1 match 证实） |
| R1 (build) | 16384 | 16384 | 0 | 16384 | 0 | 32704（由 R2 前 evict + R3 愈后推断，chunk 边界） |
| **R2 (gap 后测量)** | **32767** | **16384** | **16383** | **16384** | **16320** | **32704（由 R3 证实）** |
| R3 (探测) | 32767 | 32704 | 63 | 64 | – | –（确认自愈） |

关键原始日志行：

```
match_end req=R2 full_kv_hit=32767 exec_prefix=16384 mamba_boundary=16384 branching=32704 gap=16383
extend    req=R2 fill_ids=32768 prefix_len=16384 extend_start=16384 extend_len=16384 extend_end=32768
forced_track req=R2 prefix_len=16384 extend_len=16384 branching_seqlen=32704 last_track_seqlen=32704
dup_free  node=47 dup_start=0 consumed_from=16320 freed=16320 step_prefix_len=16320 total_prefix=16384
match_end req=R3 full_kv_hit=32767 exec_prefix=32704 mamba_boundary=32704 gap=63
extend    req=R3 fill_ids=32768 prefix_len=32704 extend_start=32704 extend_len=64
```

## 预期逐项核对

| 预期 | 实测 | 判定 |
|---|---|---|
| `full_kv_hit_length > len(prefix_indices)`，gap>0 | 32767 > 16384，gap=16383 | ✅ |
| `extend_tokens ≈ gap_tokens`（N-1 语义允许差 1） | 16384 = gap+1 | ✅ |
| `freed_duplicate_kv_tokens ≈ gap_tokens` | 16320 = [16384,32704) 旧 FA 行；差 63 = [32704,32767) 尾段（由 R3 的 full_kv_hit=32767 证实该段旧行仍保留、未被 dup-free 路径处理） | ✅（差异已解释） |
| 记录新 checkpoint 位置（自愈靠近原 FA-KV hit 边界） | 32704 = 32767 按 mamba chunk 128 对齐；R3 仅需 64-token extend（0.20s vs R2 的 0.88s） | ✅ |

## 语义结论（Motivation 实验直接可用的量化事实）

1. **Physical FA-KV 覆盖 ≠ 可执行前缀**：FA 行完整保留（full hit 32767）但深 mamba ckpt 被容量逐出后，请求只能从浅 ckpt（16384）执行，**gap 段 16383 tokens 全部走完整 EXTEND re-prefill（全层全 FLOPs）**——即静态分析 §6 的 replay 语义在运行时成立。
2. **Gap 段既有 FA-KV 不被复用**：R2 在 gap 段重新计算写入新 slots；insert 时旧 rows 作为 duplicate 被 `FreeDeviceKV` 释放（16320 行）——"先重算再替换"的浪费被直接量化。
3. **Branching 自愈**：extend 内 forced-track 在 chunk 对齐的 branching 点（32704）打新 ckpt，下一个同前缀请求（R3）恢复近全命中（gap 63 ≈ N-1+对齐余量）。
4. **Side observation**：`insert_ckpt` 探针在 cache_finished 路径打印 `ckpt_pos=0`——extra_buffer 策略下有效 ckpt 多在 chunked cache_unfinished 边界捐出，finish 时 `mamba_last_track_seqlen` 常为 0；真实 ckpt 位置需由后续请求的 match（R3 法）或 unfinished 路径探针确认。

## 文件

- `server_full.log` — 完整服务器日志（141 行 [FSVAL]）
- `driver.py` / `driver_results.json` — workload 驱动与结果
- `instrumentation.patch` — 4 个只读探针的完整 diff（对 pristine 镜像）

## 复现

```bash
docker run -d --name fsval-gap --gpus device=1 --network host --shm-size 32g \
  -v /home/wjg/models/qwen3.5-9b:/model:ro -e HF_HUB_OFFLINE=1 lmsysorg/sglang:v0.5.17-cu129-runtime sleep infinity
# apply instrumentation.patch to /sgl-workspace/sglang/python/sglang/srt/... (3 files), clear __pycache__
docker exec -d fsval-gap bash -c "python -m sglang.launch_server --model-path /model \
  --context-length 45056 --mem-fraction-static 0.40 --chunked-prefill-size 45056 \
  --max-mamba-cache-size 16 --mamba-radix-cache-strategy extra_buffer \
  --disable-cuda-graph --disable-overlap-schedule --port 49930 --log-level info > /srv.log 2>&1"
python3 driver.py   # 注意清空 http_proxy/https_proxy 或设 no_proxy=127.0.0.1
```
