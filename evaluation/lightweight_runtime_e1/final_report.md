# FlowState E1 最终报告

**Gate：READY**。CPU门禁见cpu_regression_results.json。独立Smoke/Cold及性能尝试见execution_ledger.json；完成性能臂12/12，未替换失败记录。

## 实现与安全

见implementation.md。生产代码、算法、Phi、Strict Compatibility未修改。轻量路径保留真实Scheduler连续隔离、完整元数据、FA前沿、组件切片身份、全域引用/别名/记账、Apply重读和原生产逐项安全协调。没有未经认证的generation。重型路径原样保留；Smoke核对内容及COW，各次输入/计划/状态一致性见state_consistency.json。

CPU模拟证明合同拒绝，不等于GPU认证；轻量性能路径并不每步对张量内容哈希，实际安全范围依赖已审计Native原语和独占窗口。完整动态结论以本报告门禁及原始记录为准，不能从CPU通过直接升级。

## 四臂全部测量

|指标（ms，均值）|A 重型Greedy|B 重型Exact|C 轻量Greedy|D 轻量Exact|
|---|---:|---:|---:|---:|
|observation_stage_ms|1074.589|1169.963|44.582|43.139|
|reference_observation_rpc_wall_ms|1985.939|1946.518|1877.401|2031.953|
|epoch_authentication_ms|539.734|532.904|104.987|104.316|
|solve_ms|0.112|0.137|0.111|0.141|
|plan_validation_ms|8.321|8.334|8.643|8.666|
|apply_check_ms|553.761|549.289|37.980|37.518|
|reconciliation_ms|175.868|189.283|140.759|139.460|
|control_rpc_wall_ms|4682.064|4926.071|788.566|806.993|
|prefill_cuda_event_ms|79.263|45.478|76.846|43.647|
|ttft_ms|1077.949|1079.284|107.556|73.169|
|request_wall_ms|7824.606|8429.864|6440.403|6372.788|
|control_to_completion_ms|12791.662|13670.431|7449.515|7385.488|
|audit_inclusive_to_completion_ms|14856.747|15709.050|9418.211|9510.282|

逐次值见gpu_raw_timings.csv；均值、中位数、范围见four_arm_summary.csv；分阶段控制及差值分别在control_cost_breakdown.csv、paired_differences.csv。UNKNOWN表示未完成，不从旧B5-D4b填补。

## 两个问题分别判断

以下差值为后者减前者，负值表示所测区间较短；逐次胜负全部保留。

- 轻量减重型_Greedy / control_rpc_wall_ms：差值均值-3893.498 ms；较短/相同/较长=3/0/0。
- 轻量减重型_Greedy / prefill_cuda_event_ms：差值均值-2.417 ms；较短/相同/较长=3/0/0。
- 轻量减重型_Greedy / ttft_ms：差值均值-970.393 ms；较短/相同/较长=3/0/0。
- 轻量减重型_Greedy / control_to_completion_ms：差值均值-5342.147 ms；较短/相同/较长=3/0/0。
- 轻量减重型_Greedy / audit_inclusive_to_completion_ms：差值均值-5438.536 ms；较短/相同/较长=3/0/0。
- 轻量减重型_Exact / control_rpc_wall_ms：差值均值-4119.078 ms；较短/相同/较长=3/0/0。
- 轻量减重型_Exact / prefill_cuda_event_ms：差值均值-1.831 ms；较短/相同/较长=3/0/0。
- 轻量减重型_Exact / ttft_ms：差值均值-1006.114 ms；较短/相同/较长=3/0/0。
- 轻量减重型_Exact / control_to_completion_ms：差值均值-6284.942 ms；较短/相同/较长=3/0/0。
- 轻量减重型_Exact / audit_inclusive_to_completion_ms：差值均值-6198.768 ms；较短/相同/较长=3/0/0。
- 重型_Exact减Greedy / control_rpc_wall_ms：差值均值244.007 ms；较短/相同/较长=1/0/2。
- 重型_Exact减Greedy / prefill_cuda_event_ms：差值均值-33.784 ms；较短/相同/较长=3/0/0。
- 重型_Exact减Greedy / ttft_ms：差值均值1.335 ms；较短/相同/较长=1/0/2。
- 重型_Exact减Greedy / control_to_completion_ms：差值均值878.768 ms；较短/相同/较长=1/0/2。
- 重型_Exact减Greedy / audit_inclusive_to_completion_ms：差值均值852.303 ms；较短/相同/较长=1/0/2。
- 轻量_Exact减Greedy / control_rpc_wall_ms：差值均值18.427 ms；较短/相同/较长=2/0/1。
- 轻量_Exact减Greedy / prefill_cuda_event_ms：差值均值-33.199 ms；较短/相同/较长=3/0/0。
- 轻量_Exact减Greedy / ttft_ms：差值均值-34.387 ms；较短/相同/较长=3/0/0。
- 轻量_Exact减Greedy / control_to_completion_ms：差值均值-64.026 ms；较短/相同/较长=2/0/1。
- 轻量_Exact减Greedy / audit_inclusive_to_completion_ms：差值均值92.071 ms；较短/相同/较长=2/0/1。

1. **在本受控范围内保持了严格安全合同**：66/66 CPU门禁、独立Smoke及12个性能臂全部通过。每次7候选、4Pending、13准入/15拒绝/0未知；Greedy保留公共及AB状态，Exact保留AB及CD状态；每次实际5项驱逐，空闲槽21→26，最终2项驻留。原生产Bind/Preflight/Postcheck及Adapter未修改。来源血缘仍未认证。
2. **轻量化明显降低本次控制成本**：Greedy观察阶段1074.589→44.582 ms（−95.85%），Exact 1169.963→43.139 ms（−96.31%）；控制RPC分别降低83.16%和83.62%。这是实际新路径测量，不是旧耗时扣除审计估计。
3. **路径轻量化与策略收益分开**：轻量Exact相对轻量Greedy，Prefill均值76.846→43.647 ms、TTFT 107.556→73.169 ms，三次均较短。H均为3904，Greedy E/G=2432/1472，Exact E/G=3520/384；模型四Pending J分别130.803与48.229 modeled ms，不等于实测C请求耗时。
4. **轻量化有受控总路径改善，Exact没有稳定完整路径优势**：相同策略轻量相对重型的Control-to-Completion三次均降低，均值降幅41.76%/45.97%；计入前置重型参考审计后仍三次均降低，均值降幅36.61%/39.46%。但轻量Exact相对轻量Greedy的Control-to-Completion只减少64.026 ms，2次更短、1次更长；含前置审计的均值反而增加92.071 ms。不能声称Exact已获得稳定净收益，更不能推广为生产收益。
5. 无法证明生产吞吐、并发、多工作负载、硬deadline、P95或统计显著性；Node27继续PROVENANCE_UNVERIFIED。逻辑K2不代表物理HBM压力。

## 失败、完整性与清理

失败：[]。
冻结文件/锁变化：[]。
清理：{"smoke": "PASS", "cold": "PASS", "r01_A": "PASS", "r01_B": "PASS", "r01_C": "PASS", "r01_D": "PASS", "r02_D": "PASS", "r02_C": "PASS", "r02_B": "PASS", "r02_A": "PASS", "r03_C": "PASS", "r03_A": "PASS", "r03_D": "PASS", "r03_B": "PASS"}。
任务结束，不进入其他阶段。
