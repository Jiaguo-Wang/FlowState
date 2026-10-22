# FlowState E2 最终报告

Gate：**READY**。4/4组完成严格合同和分配比较；4/4组形成真实驻留且满足EPOCH_ADMISSIBLE的跨Pending共享。

| 组 | 候选 | 准入/拒绝/未知 | 共享候选 | K_min | 结果 |
|---|---:|---|---:|---:|---|
| w1 | 7 | 12/16/0 | 3 | 4 | READY |
| w2 | 7 | 12/16/0 | 3 | 4 | READY |
| w3 | 7 | 14/14/0 | 5 | 4 | READY |
| w4 | 7 | 13/15/0 | 4 | 4 | READY |

16个Snapshot—预算条件中，16个条件至少两种方法的最终J严格不同。四预算是同一组的重复条件，不作为独立工作负载样本。

| 方法 | 达到Exact | 严格落后Exact |
|---|---:|---:|
| FlowState-Exact | 16 | 0 |
| FlowState-Greedy | 4 | 12 |
| StandaloneTopK | 4 | 12 |
| PCB | 4 | 12 |
| LDR | 4 | 12 |
| WFC | 4 | 12 |

| 对比（前者相对后者） | 胜 | 平 | 负 |
|---|---:|---:|---:|
| FlowState-Exact / FlowState-Greedy | 12 | 4 | 0 |
| FlowState-Greedy / StandaloneTopK | 3 | 13 | 0 |
| FlowState-Greedy / PCB | 12 | 0 | 4 |
| FlowState-Greedy / LDR | 12 | 0 | 4 |
| FlowState-Greedy / WFC | 0 | 16 | 0 |

## 能证明什么

在四组固定真实来源受控组合中，历史请求可自然物化共享Recurrent状态，严格矩阵可支持统一模型目标下的分配比较。结果完整展示Exact、Greedy与简单规则的相同和差异；支持重叠收益与预算截断会影响集合分配质量，而非仅比较选集名称。详见workload_structure_analysis.md逐组全部成本与选集。

## 不能证明什么

本轮未派发任何Pending，Native Pending匹配和逐边恢复均未测试，不能把准入升级为RESUME_VALIDATED。四组物理来源血缘均为PROVENANCE_UNVERIFIED。没有策略GPU执行、驱逐或性能配对，不能从modeled ms推断TTFT或净端到端收益。逻辑K不是HBM压力。全部组合是Trace-derived Controlled Composition，不是自然并发Agent工作负载。

## 完整性与复现

协议在GPU前冻结，CPU回归复现B5-C4的24个历史策略条件。每组capture_manifest.json锁定原始采集文件；每组allocation_results.json保存支持集、Exact全部同分集合、选择步骤及原生Baseline记录。source_identity.json和execution_lock.json记录执行前来源/代码身份，gpu_cleanup.json逐组保留资源清理。旧cpu_run.log末行追加差异已在来源报告中以精确字节证明解释，未改旧数据。

执行结束后停止；未进入E3/E4或扩充来源。

## 结构解释与简单规则适用条件

四组K1时公共共享状态已经最优；K2/K3共8个条件Exact严格优于本轮全部其他规则。Greedy首步保留公共祖先，此后不能撤回，在K3该祖先已变为零删除贡献却仍占预算。w3/K3、w3/K4、w4/K4进一步出现静态TopK重复覆盖，动态更新可以避免该冗余；但Greedy与WFC始终持平。K4时四组PCB/LDR均达到Exact，说明预算充足时简单方法已经足够。

四组均28槽、7驻留、21空闲，未发生策略驱逐。全部32条历史请求完成，16个Pending均未提交；清理后各GPU检查无计算进程、显存14 MiB。日志中的退出CUDA IPC告警原样保留，未据此省略独立容器/GPU清理检查。
