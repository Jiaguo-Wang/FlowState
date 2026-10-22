# Phase B5-D3 最终报告

**Gate：READY**。

## 实际结果

当前新Runtime自然捕获7个候选、4个Pending。Exact K2选择 `['B5C3_NODE_18', 'B5C3_NODE_27']`；模型化J=48.228728759ms。当前Epoch独立Decimal Oracle一致，没有使用历史Node/Slot硬编码选择。

单次Scheduler同步控制函数内完成当前观察、规划、Apply-time重验、bind/preflight及StateController.reconcile_strict。实际执行5项Recurrent-only Eviction；最终Selected驻留及全部后置条件通过。控制函数没有返回调度循环后再使用旧绑定执行。

正式恢复仅 `B5C2_B_TURN_003`，实际来源 `B5C3_NODE_18`、Node 18、Slot [3]，COW工作槽 [19]。实际H/E/G=4096/3712/384。逐Token 8 Token输出与独立Cold Reference对照见output_correctness.json。

FA协调前后索引及实际K/V内容不变；恢复后Selected组件内容与非重算区域旧FA内容保持。详见state_integrity_report.md。没有部分驱逐失败或静默异常成功；如最终复核错误，原因列表为[]。

## 耗时边界

control_overhead_breakdown.csv保留重型观察/GPU同步、Epoch认证、Exact+Plan校验、Oracle、Apply-time校验、严格协调及每次Native run_batch宿主调用区间。原Adapter操作耗时也单列，但包含原安全检查，不是裸GPU内核时延。request wall包含观测写盘、匹配与COW审计；run_batch调用亦不等同CUDA event测量。不能把这些数值称生产轻量控制、线上P95或TTFT收益。

结构/身份/保护检查可能进入真实生产关键路径；全树内容哈希、独立Oracle、详细日志等是本轮实验审计成本。没有实现删除审计后的轻量版本，更没有通过相减宣称净性能改善。J是模型化成本，与实测时间分开。

## 完整性与清理

冻结来源及生产源码摘要不变=True。Strict与Cold各独立生命周期、各仅一次正式Pending，GPU清理见gpu_cleanup.json。CUDA IPC退出警告如存在保留runtime.log原文；资源清理以连续GPU与容器观测判断，不能只依据返回码。首次CPU夹具失败已保留，修正后真实接口CPU门禁通过，再启动GPU。

工作负载是Trace-derived Controlled Composition；K2不是物理显存压力。本轮只验证功能正确性，不进行Greedy/Exact性能A/B，不推广所有兼容边。完成后停止。
