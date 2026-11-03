# 计时口径表（三口径分层）

| 指标 | 覆盖事件 | 本轮中位数(ms) |
|---|---|---|
| 口径A cuda_prefill_ms（compute_cuda_ms 同值） | 首 extend 批 CUDA 事件窗（含 host 提交空隙；形状预热后无编译/autotune） | 97.3 |
| 口径A cow_copy_cuda_ms / cow_copy_host_ms | COW 设备拷贝事件 / 原生宿主调用（B 类保留项） | — |
| 口径B ttft_ms | client 发送 → 首个含 Token 流式 chunk；测量模式：窗口内零 A 类审计零落盘 | 105.7 |
| 口径B queue/pregap/execwin/postgap/delivery | 六段分解（闭合断言 5ms）；pregap 对 E>0 含原生 mamba 绑定（B 类） | — |
| 口径B request_wall_ms / stream_tpot_ms | 整请求墙钟与逐 Token 间隔（含 8 Token 解码） | 254.0 |
| 口径C control_rpc_wall_ms | b5d4_apply RPC 墙钟（审计增强研究仪表：重型观察+FA 内容审计+apply 基线快照） | 11302.2 |
| 口径C ctrl_auth_plan_ms / ctrl_reconcile_ms | 生产控制核心近似（认证+计划 / 真实驱逐+合同后检），单列供净收益核算 | — |
