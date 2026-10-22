# E4-Phase 1A：w1/K2独立来源Native Resume门禁

**Gate：READY**。本轮仅做正确性验证，不报告性能优劣。

实际认证8/8个策略—Pending恢复；Cold完成4/4；生命周期清理记录12份。

旧BLOCKED目录完整保留。本次只改变输出目录，复用已通过10项新增CPU测试及54项Strict Epoch/Control测试的入口。每个Pending使用独立Cold、Greedy、Exact生命周期，严格臂各重放八条相同历史；没有在被前一请求污染的状态上验证后续Pending。

## 冻结逻辑选集与实际执行范围

| 策略 | 冻结候选 | Model J ms |
|---|---|---:|
| Greedy | ['B5C3_NODE_20', 'B5C3_NODE_21'] | 126.590493308 |
| Exact | ['B5C3_NODE_18', 'B5C3_NODE_21'] | 53.770758869 |

Greedy+1-Swap在本条件与Exact同集，因此没有重复物理Arm。上述ID仅用于标识冻结候选；每个新生命周期按完整Token前缀双射重新映射，当前Node/Slot/组件由真实Observe、Bind、Preflight和Apply-time检查认证。

## 逐Pending真实结果

| Pending | 策略 | 模型T/E/G | 实际H/E/G | 当前来源Node/Slot | 工作Slot | Gate |
|---|---|---|---|---|---|---|
| IMW1_A_TURN_003 | Greedy | 3648/2432/1216 | 3648/2432/1216 | 20/[8] | [19] | READY |
| IMW1_A_TURN_003 | Exact | 3648/3392/256 | 3648/3392/256 | 18/[3] | [19] | READY |
| IMW1_B_TURN_003 | Greedy | 3712/2432/1280 | 3712/2432/1280 | 20/[8] | [19] | READY |
| IMW1_B_TURN_003 | Exact | 3712/3392/320 | 3712/3392/320 | 18/[3] | [19] | READY |
| IMW1_C_TURN_003 | Greedy | 4096/3712/384 | 4096/3712/384 | 21/[10] | [19] | READY |
| IMW1_C_TURN_003 | Exact | 4096/3712/384 | 4096/3712/384 | 21/[10] | [19] | READY |
| IMW1_D_TURN_003 | Greedy | 4160/3712/448 | 4160/3712/448 | 21/[10] | [19] | READY |
| IMW1_D_TURN_003 | Exact | 4160/3712/448 | 4160/3712/448 | 21/[10] | [19] | READY |

未完成项不以模型预测填充Runtime值，None表示没有相应认证。完整输出Token与Cold逐Token对照见resume_correctness.json；全部原始Match、Admission、COW、执行和状态摘要保留在各Arm子目录。

## 安全合同与证据等级

已通过行的认证要求：单Scheduler同步控制窗口内重建/认证当前Epoch并确认冻结计划；Bind、Preflight和Apply-time内容/身份一致；正式StateController.reconcile_strict执行5项Recurrent-only Eviction；控制后7→2个候选驻留、空闲槽21→26；保留的Conv/Temporal内容不变；全部原FA索引与内容不变。请求实际Admission再次检查状态，不能把前一控制窗口称作跨RPC租约。

Native实际匹配必须等于当前物理绑定且H/E符合计划；原生COW源/目的切片隔离，源前后与目的复制后内容摘要一致；实际Prefill/Decode完成，8个确定性Token及停止行为与独立Cold完全一致；请求后保留状态内容及非重算FA仍受保护。正常请求会重算E之后的目标FA并创建新状态，不能机械要求请求后整棵树不变。

- EPOCH_ADMISSIBLE：当前完整严格准入合同，不单独证明实际恢复。
- RECONCILIATION_VALIDATED：仅对应真实物理控制及后置检查已通过的Arm。
- RESUME_VALIDATED：仅对应Native来源/COW/执行/Cold输出及状态保护均通过的Pending。
- PROVENANCE_UNVERIFIED：全部维持原等级；Node匹配和Token共享不认证张量跨Workflow血缘。

## 资源、完整性和结论边界

每个生命周期启动前检查GPU0无其他计算任务；只清理本轮专用容器，保留逐次退出及显存/进程记录。CPU协议、输入、模型和Runtime源码身份逐项核对，不改生产算法、Phi、严格兼容矩阵或旧实验。日志可能包含既有CUDA事件与Wall计时，但本轮不据此进行性能比较，不将modelled ms解释为GPU时延；K=2仍是逻辑保留预算，不是HBM压力。

本轮完成后停止，不扩展w2–w4，不进入E4-Phase 1B或性能实验。
