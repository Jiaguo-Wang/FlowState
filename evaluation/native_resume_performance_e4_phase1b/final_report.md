# E4-Phase 1B：w1/K2轻量恢复性能Smoke

**Gate：READY**。有效测量臂12/12，已验证严格恢复（含Smoke）16/16，清理记录18/18。

只使用固定w1/A与C，四个轻量Smoke通过后才测量。每个臂独立Engine、八条相同历史；每个Pending独立Cold，temperature=0/max_new_tokens=8/ignore_eos=True。没有复用被其他Pending污染的状态。

## 实际恢复与计算区间

|Pending|策略|H/E/G|实际Prefill Tokens|
|---|---|---|---:|
|IMW1_A_TURN_003|Greedy|[(3648, 2432, 1216)]|[1818]|
|IMW1_A_TURN_003|Exact|[(3648, 3392, 256)]|[858]|
|IMW1_C_TURN_003|Greedy|[(4096, 3712, 384)]|[785]|
|IMW1_C_TURN_003|Exact|[(4096, 3712, 384)]|[785]|

## 三对实测结果

所有值为毫秒；差值为Exact减Greedy，负值表示Exact较短。均值、中位数、范围及全部逐次原始数据分别见timing_summary.csv、gpu_raw_timings.csv、paired_differences.csv。

|Pending|指标|Greedy均值|Exact均值|配对差值均值|Exact较短/相同/较长|
|---|---|---:|---:|---:|---|
|IMW1_A_TURN_003|observation_inline_ms|43.275|42.800|-0.475|2/0/1|
|IMW1_A_TURN_003|solve_ms|0.110|0.145|0.034|0/0/3|
|IMW1_A_TURN_003|reconciliation_ms|138.172|138.857|0.686|2/0/1|
|IMW1_A_TURN_003|control_rpc_ms|775.516|824.074|48.558|0/0/3|
|IMW1_A_TURN_003|prefill_cuda_ms|73.026|42.618|-30.408|3/0/0|
|IMW1_A_TURN_003|ttft_ms|102.601|74.258|-28.343|3/0/0|
|IMW1_A_TURN_003|control_to_completion_ms|7527.624|7352.864|-174.760|2/0/1|
|IMW1_A_TURN_003|audit_inclusive_completion_ms|9422.254|9406.734|-15.520|2/0/1|
|IMW1_C_TURN_003|observation_inline_ms|45.690|43.225|-2.465|3/0/0|
|IMW1_C_TURN_003|solve_ms|0.109|0.152|0.043|0/0/3|
|IMW1_C_TURN_003|reconciliation_ms|138.695|139.288|0.593|2/0/1|
|IMW1_C_TURN_003|control_rpc_ms|838.413|819.970|-18.442|2/0/1|
|IMW1_C_TURN_003|prefill_cuda_ms|37.062|37.460|0.398|1/0/2|
|IMW1_C_TURN_003|ttft_ms|66.731|67.588|0.857|1/0/2|
|IMW1_C_TURN_003|control_to_completion_ms|7485.015|7511.830|26.815|2/0/1|
|IMW1_C_TURN_003|audit_inclusive_completion_ms|9436.906|9449.168|12.262|2/0/1|

## 方向性判断

A的三对Prefill CUDA与TTFT均改善，支持本次更深Native前沿能够减少实际恢复计算。A的Control RPC三对均更慢；完整Control-to-completion与Audit-inclusive各仅2/3对改善，不能证明控制后的稳定净收益。C保持相同前沿和Prefill Tokens，时间变化属于负对照波动，不是分配恢复收益。

当前Control RPC包含客户端请求/响应证据写盘与fsync，不能解释为纯生产控制成本。前置全内容审计及请求后对拍分别保存；完整路径没有静默扣除这些区间内实际发生的审计工作。

## 正确性和结论边界

有效臂均要求当前Strict Epoch与冻结输入及重型参考一致、Apply-time同Scheduler调用、5项真实驱逐、最终恰为计划选集、FA与非目标元数据不变、实际Native来源匹配、COW返回、8 Token与独立Cold精确一致、请求后保留组件及受保护FA内容不变。四个Smoke还执行控制后全内容对拍与COW源/目标内容检查。性能臂不在关键路径逐槽哈希，保留原E1安全检查及请求后内容审计。

A用于检验更深恢复位置，C为相同恢复位置负对照；C的任何时间变化不能归因于前沿改善。求解目标J为四Pending模型值，不能直接解释本次单请求CUDA或Wall差异。PROVENANCE_UNVERIFIED保持不变。

必须分别解释CUDA计算、客户端TTFT、控制到完成及含前置审计的完成时间。三对只提供受控方向性描述，不能证明稳定生产净收益、P95或显著性。K=2是逻辑预算，不是HBM容量压力；未扩展其他工作负载。

计时的父子区间与审计范围见measurement_validity.md。已保留全部尝试和失败；若未完成，不补造缺失数据。完成后停止。
