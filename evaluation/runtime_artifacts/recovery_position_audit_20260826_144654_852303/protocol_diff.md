# Step 9D 与 Step 10D.1 协议差异

| 项目 | Step 9D | Step 10D.1 |
|---|---|---|
| 模型 | Qwen3.5-9B | Qwen3.5-9B |
| 模型路径 | /model | /model |
| 模型 revision | artifact 中不可用 | artifact 中不可用 |
| SGLang | 0.5.17 | 0.5.17 |
| 容器镜像 | artifact 中不可用 | artifact 中不可用 |
| GPU | NVIDIA H100 PCIe 80 GiB | NVIDIA H100 PCIe |
| TP | 1 | 1 |
| 输出 token | 1 | 1 |
| 请求入口 | 进程内 Engine.generate(stream=True) | 进程内 Engine.generate(stream=True) |
| 计时边界 | 复用 Step 9B 首个流式 token client-side 计时 | 复用 Step 9D 的首个流式 token client-side 计时 |
| warmup | 2 | 2 |
| measured | 12 | 12 |
| prompt/token 构造 | make_tokens 的固定算术序列；checkpoint 基种子 51001，target suffix 种子 541019，suffix 长度 63 | make_tokens 的固定算术序列；checkpoint 基种子 51001，target suffix 种子 741019，suffix 长度 63 |
| checkpoint 构造 | 按候选 token_pos 递增构建同 lineage 节点 | 按候选 token_pos 递增构建，并显式清理 chunk-boundary 循环状态 |
| FA-KV 构造 | 构建后保留全部 FA-KV | 构建后保留全部 FA-KV |
| 循环状态驱逐 | flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only | flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only |
| 服务重置 | 每个 trial 前 flush_cache 并验证空缓存 | 每个 trial 前 flush_cache 并验证空缓存 |

## 引擎参数

| 参数 | Step 9D | Step 10D.1 |
|---|---:|---:|
| `chunked_prefill_size` | 45056 | 45056 |
| `context_length` | 45056 | 131200 |
| `disable_cuda_graph` | True | True |
| `disable_overlap_schedule` | True | True |
| `enable_request_time_stats_logging` | True | True |
| `log_level` | info | info |
| `mamba_max_states_per_path` | -1 | -1 |
| `mamba_radix_cache_strategy` | extra_buffer | extra_buffer |
| `mamba_track_interval` | 256 | 256 |
| `max_mamba_cache_size` | 24 | 24 |
| `mem_fraction_static` | 0.4 | 0.4 |
| `model_path` | /model | /model |
| `stream_interval` | 1 | 1 |
| `tp_size` | 1 | 1 |

## T / E / G

| G | Step 9D T/E | Step 10D.1 T/E |
|---:|---:|---:|
| 0 | 32768 / 32768 | 131072 / 131072 |
| 4096 | 32768 / 28672 | 131072 / 126976 |
| 8192 | 32768 / 24576 | 131072 / 122880 |
| 16384 | 32768 / 16384 | 131072 / 114688 |
| 32768 | 32768 / 0 | 131072 / 98304 |

## 关键差异

- T 从固定 32768 改为固定 131072
- context_length 从 45056 提高到 131200
- target suffix 的确定性种子从 541019 改为 741019
- 长上下文构造显式清理 chunk-boundary 循环状态
