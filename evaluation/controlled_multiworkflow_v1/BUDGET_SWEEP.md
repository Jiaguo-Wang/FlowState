# 受控多工作流 v1 预算扫描

## 研究问题

本扫描要回答：随着 recurrent-state memory budget 增加，不同 allocation policy 如何在 executable coverage 与 recovery cost 间形成不同 memory–recovery tradeoff？

扫描直接复用 `scenario.py` 中冻结的七个待续请求、五个等大小候选和 evaluation 层策略。检查点大小固定为 51,511,296 bytes，预算依次取 `K=1、2、3、4、5`。本步骤不重新定义 workload，也不改变任何策略规则。

## 指标定义

每个 `K × policy` 组合记录所选检查点、总恢复间隔、每请求平均恢复间隔、planning executable prefix ratio 和 `sum Phi(G)`。

planning executable prefix ratio 定义为 `sum(E_p) / sum(T_p)`。这里的分母是 planning target，不是真实运行时 physical hit，因此字段固定命名为 `planning_executable_prefix_ratio`，不得与 GPU runtime EPR 混用。

FlowState 相对每个基线还记录 absolute gap reduction、relative gap reduction 和 estimated recovery cost reduction。基线总 gap 为零时，相对 gap reduction 没有有效分母，记录为 `None`，不伪造百分比。

## 解释边界

本扫描是 planning/offline sanity，用于挑选后续最有信息量的 GPU budget 点，并验证成本随预算增加的基本趋势。它不是最终 GPU 实验结果，也不能用于报告真实时延提升。

该扫描固定单个 decision epoch，不包含请求完成后产生新检查点的 Online Sequential 状态演化。
