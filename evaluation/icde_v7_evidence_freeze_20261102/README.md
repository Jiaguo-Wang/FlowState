# v7.0证据审计

本目录只写本轮派生产物。用户提供全文身份在`paper_version.json`，原始实验和论文没有修改。

复算需指定**空的新目录**，不要覆盖本目录已封存结果：

```bash
export FLOWSTATE_AUDIT_OUTPUT=/tmp/flowstate_v7_evidence_recheck
python3 -B -m evaluation.icde_v7_evidence_freeze_20261102.audit
python3 -B -m evaluation.icde_v7_evidence_freeze_20261102.complete_audit
python3 -B -m evaluation.icde_v7_evidence_freeze_20261102.finalize
python3 -B -m evaluation.icde_v7_evidence_freeze_20261102.parent_links
python3 -B -m evaluation.icde_v7_evidence_freeze_20261102.build_report
```

所有步骤只读已有数据，不导入生产Solver执行，不运行GPU，不重新拟合模型或运行策略。`K_min`位掩码DP只是结构统计；Exact最优性读取已存枚举及独立复核证书，不新增枚举实验。旧Bootstrap区间直接引用冻结产物，未重新抽样。

`evidence_master.csv`逐指标标注正文位置、原始来源、统计分母及限制。`workload_structure_summary.csv`的1357行分成三域：573原比例、764固定、20 Strict；其`same_set_all`对应各域已声明方法，不能混成总体胜率。Formal两域为FlowState/PCB/LDR/WFC，Strict20为七方法。

审计裁决为NEEDS_TARGETED_FIX：三项CUDA汇总数字、短上下文误差分母需要更正；8张表格不可见，不能签署全表一致。唯一补证是提供可见表格静态导出后做同源对拍；本目录没有执行该补证或新实验。
