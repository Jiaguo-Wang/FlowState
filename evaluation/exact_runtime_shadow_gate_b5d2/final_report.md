# Phase B5-D2 最终报告

**Gate：READY**。

## 当前真实Epoch

一个新Engine生命周期完成固定八条历史回放；四个Controlled Pending均未提交。当前候选7，Pending4，完整配对28：{'EPOCH_ADMISSIBLE': 13, 'REJECTED': 15, 'UNKNOWN': 0}。配置、实际接收Token与冻结B5-C2输入核验通过。Node/Slot均来自当前观察，未复用历史绑定。保持Trace-derived Controlled Composition语义。

主要且唯一规划预算K=2。新Epoch身份：`b5c3-191-655747993937905`。完整StrictAllocationEpoch摘要、来源字节SHA和两份原StrictAllocationPlan见shadow_plan_comparison.json。两份计划通过既有validate_plan；无PhysicalBinding。

|求解器|选集|J（模型化ms）|最优性认证|
|---|---|---:|---|
|Greedy|B5C3_NODE_20,B5C3_NODE_18|130.803083830|False|
|Exact|B5C3_NODE_18,B5C3_NODE_27|48.228728759|True|


独立Oracle直接读取当前Token位置及严格边，用Decimal冻结Phi穷举所有大小≤2子集，目标与代表集合一致，见current_epoch_oracle.json。没有以历史B5-C4目标替代本次认证。详细每Pending的T/E/G及成本均保存在两份Plan中。

## 分段耗时

|阶段|归属|实测ms|
|---|---|---:|
|runtime_observation_freeze_rpc|共同|1102.325|
|epoch_construction_authentication|共同|110.831|
|benefit_matrix|共同|0.041|
|solve|Greedy|0.187|
|plan_construction_validation|Greedy|13.553|
|solve|Exact|0.225|
|plan_construction_validation|Exact|13.747|
|independent_oracle|审计|0.404|


这是一次实际调用的耗时，不是线上P95或重复统计。共同预处理只执行一次；观察RPC含GPU同步、摘要、传输与解码，Epoch构建认证含序列化、来源整理、严格验证及快照写出。CPU规划在前观察RPC返回后、后观察RPC之前执行，不占Scheduler连续控制窗口。Oracle为独立审计开销，不归入任一求解器。两个求解器连续运行的总时间不能作为单个规划器独立耗时；需要计算单个逻辑路径时仅加共同段和对应solve/plan两段。

## 无副作用与清理

两次完整物理Census、内容摘要及Match/COW/Eviction列表核验全部通过；候选和FA不变，四个Pending未派发。源文件摘要不变=True，Engine shutdown返回=True，进程返回码=0，GPU清理=PASS。专用容器移除及连续资源检查见gpu_cleanup.json；未终止其他用户进程。

退出日志中若存在CUDA IPC释放警告，保留runtime.log原文；清理判断以实际进程、容器和显存检查为准，不仅依据退出码。实际警告摘录见runtime_integration_risks.md。

本次证明新真实观察可进入纯规划合同，不证明未来应用时状态新鲜、Reconciliation安全、请求恢复正确或GPU净收益。EPOCH_ADMISSIBLE不是RESUME_VALIDATED。未调用驱逐、恢复或保留策略，不进入下一阶段。
