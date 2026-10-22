# Phase B5-D4b 最终报告

**Gate：READY**。有效配对5/5；已完成策略臂10；失败记录0。Gate不以Exact获胜为条件。

## 恢复与状态证据

固定目标B5C2_C_TURN_003；实际当前选集、来源、H/E/G、工作槽及8Token对照见paired_gpu_timings.csv、native_resume_verification.json及cold_reference_correctness.json。各臂由当前Epoch重新求解，Exact与当前Oracle核对，非选状态通过原生产严格协调移除。严格兼容与Native实际来源通过Token、当前绑定、Match/COW、输出及组件后验分别核验。首次Census仅为来源线索，血缘仍为PROVENANCE_UNVERIFIED，不称作已认证跨Workflow恢复。

## 模型与真实恢复前沿

- Greedy：四Pending J=[130.80308383] modeled ms；C成本=[55.951289486] modeled ms；H/E/G=[(3904, 2432, 1472)]；实际Prefill Token数=[2003]。
- Exact：四Pending J=[48.228728759] modeled ms；C成本=[14.658225263] modeled ms；H/E/G=[(3904, 3520, 384)]；实际Prefill Token数=[915]。

## 全部成对观察

|Pair|Greedy Prefill事件ms|Exact Prefill事件ms|Exact−Greedy完整路径ms|
|---|---:|---:|---:|
|1|78.443|44.615|77.791|
|2|78.075|46.346|99.044|
|3|78.435|47.954|-288.426|
|4|79.010|46.125|-218.206|
|5|77.775|45.457|-89.636|

负值表示Exact在该实测区间更短，正值更长。以上Prefill为COW之后的Eager设备事件时间线，不等同恢复缺口G的纯计算。完整输入后缀还包含T之后的新输入，实际Prefill Token数单独保存。

- Greedy prefill_cuda_event_ms：均值78.348 ms，中位78.435 ms，范围[77.775,79.010]，n=5。
- Greedy ttft_client_ms：均值1039.121 ms，中位1045.505 ms，范围[1020.942,1047.736]，n=5。
- Greedy request_wall_ms：均值8258.730 ms，中位8248.036 ms，范围[8004.662,8435.760]，n=5。
- Greedy control_rpc_wall_ms：均值4614.229 ms，中位4657.374 ms，范围[4454.284,4680.872]，n=5。
- Greedy control_through_completion_ms：均值13219.006 ms，中位13264.530 ms，范围[13055.289,13381.751]，n=5。
- Exact prefill_cuda_event_ms：均值46.099 ms，中位46.125 ms，范围[44.615,47.954]，n=5。
- Exact ttft_client_ms：均值1078.325 ms，中位1078.688 ms，范围[1055.712,1089.536]，n=5。
- Exact request_wall_ms：均值8116.660 ms，中位8171.120 ms，范围[7891.945,8255.323]，n=5。
- Exact control_rpc_wall_ms：均值4690.122 ms，中位4691.098 ms，范围[4563.199,4769.726]，n=5。
- Exact control_through_completion_ms：均值13135.119 ms，中位13133.081 ms，范围[13046.324,13214.584]，n=5。

## 解释范围

本次五对均无失败或补跑，10个策略臂各执行5项真实Recurrent-only驱逐，共50项；另有1个独立Cold生命周期。10份输出均与Cold的8个Token逐项一致。全部当前Native Match、COW完成、所选历史组件保护与FA验证通过；来源血缘等级仍未认证。这些是三个分开的证据结论。

预测改善与实测的关系：C的模型成本由55.951降至14.658 modeled ms；实测Prefill事件均值由78.348降至46.099 ms，五对均缩短，方向一致但数值不可互换。实际Prefill包含目标T之后的新输入：2003→915 Token，而不是直接执行G个Token。

不利结果：Exact客户端TTFT五对均更长，均值1039.121→1078.325 ms；控制RPC均值4614.229→4690.122 ms。含已测控制到请求完成的总路径均值13219.006→13135.119 ms，仅减少83.886 ms，且逐对为3次更短、2次更长，不构成稳健或生产端到端收益证据。请求Wall均值8258.730→8116.660 ms。上述均为单一受控组合的5次描述性重复，未进行显著性推断。

完整控制路径包含重型观察、GPU哈希、当前Epoch认证、Oracle、规划、Apply检查和协调；成本详见control_cost_breakdown.csv。独立求解时间不能直接与GPU恢复成本抵消。不得从剔除审计成本后的数字声称轻量生产E2E净收益。所有5对或失败臂保留，未按性能筛选。

本数据仅为一个固定四Session受控组合中单Pending C的成对实验，重复运行不是多个独立Workflow。即使实际恢复前沿更深且某个计时区间更短，也不能推广为四Pending全局吞吐或生产性能优势；首次编译、事件同步、重型审计和环境波动限制见measurement_validity.md。

## 完整性、失败与清理

冻结来源及执行锁变化：[]。清理：{"p01_Greedy": "PASS", "p01_Exact": "PASS", "cold": "PASS", "p02_Exact": "PASS", "p02_Greedy": "PASS", "p03_Greedy": "PASS", "p03_Exact": "PASS", "p04_Exact": "PASS", "p04_Greedy": "PASS", "p05_Greedy": "PASS", "p05_Exact": "PASS"}。
失败与异常：[]。原始运行日志保留所有警告；不只依靠进程返回码判断成功。未修改生产算法、Phi、Compatibility及历史冻结证据。任务结束，不扩大工作负载、不进入下一阶段。
