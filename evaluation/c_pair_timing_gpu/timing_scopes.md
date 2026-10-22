# 计时口径表：每个指标覆盖的事件范围

client 与 engine 同用 CLOCK_MONOTONIC，跨进程时间戳直接衔接；分解恒等式 TTFT ≈ queue + execwin + delivery（容差 5ms，runner 内闭合断言）。

| 指标 | 覆盖事件 | 本轮中位数(ms) |
|---|---|---|
| TTFT（ttft_ms） | client 发送请求 → 首个含 Token 的流式 chunk 到达；含 RPC/排队/准入/编译间隙/前向/采样/解词全部 | 3578.1 |
| queue（queue_ms） | client 发送 → 引擎首 batch 开始（execution[0].ns_before）；含调度等待、radix/FA 匹配、mamba 准入（E>0 时含 COW 暂存与 payload 哈希审计） | 2541.3 |
| pregap（pregap_ms） | 首 batch 开始 → 首个 kernel CUDA 事件；host 侧 kernel 编译/autotune 主要落在此段 | 474.9 |
| execwin（execwin_ms） | 首 batch 整窗（execution[0] 事件 elapsed）；事件间 GPU 空闲（等 host 编译）计入 | 944.0 |
| compute（compute_cuda_ms） | 紧凑 kernel 事件窗（compute[0].cuda_elapsed_ms）；含事件间由编译造成的空隙，非纯 kernel 执行时间之和 | 378.2 |
| postgap（postgap_ms） | 末 kernel 事件 → batch 结束 | 85.2 |
| delivery（delivery_ms） | batch 结束 → 首 chunk 到达；含采样、解词、流式下发 | 96.2 |
| cuda_prefill_ms | 全部 is_extend compute 事件 elapsed 之和（=compute_cuda_ms，单 extend batch） | 378.2 |
| first_decode（first_decode_cuda_ms） | 首个 decode 步事件窗；首 decode 形状编译证据（引擎 disable_cuda_graph=true，无 Graph Capture） | 1517.4 |
| control_rpc_wall_ms | b5d4_apply RPC 墙钟；审计增强型研究仪表（重型观察+FA 内容审计），与请求计时完全分离 | 11386.2 |
| cow_copy_cuda_ms | COW 设备拷贝事件 elapsed（宿主暂存与哈希审计不计入，落在 queue 内） | 0.6 |
