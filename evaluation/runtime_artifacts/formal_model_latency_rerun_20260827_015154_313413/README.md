# 正式位置感知模型定向 H100 latency 重跑

本目录只包含 Scalable N16 K4 与 SOTA-signal K8 两个代表点。
三个 baseline 的选择来自 Step 9B 冻结制品；FlowState 的选择来自
Step 10D.4 正式位置感知恢复模型审计。所有策略在本次运行中使用同一
runtime、同一 fresh-snapshot 协议和同一计时边界。

`class_multiplicity` 用于恢复逻辑 workload 权重；warmup 不进入统计。
checkpoint rebuild、flush 与 reconcile 均不计入目标请求 TTFT。
旧 FlowState latency 只作为历史参考，正式 baseline 对比必须使用本次
同源重跑数据。真实 latency 不由正式恢复模型预测值替代。
