# Step 13G-A · FlowState RQ3 Same-Snapshot Sanity/Structure/Mechanism Audit 最终报告

**状态：** `RQ3_SANITY_AUDIT_READY`  
**报告生成时间：** 2026-09-04  
**Audit Root：** `evaluation/runtime_artifacts/rq3_sanity_structure_audit_20260904_121500/`  
**消费对象：**
- Formal Population：`evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017/`
- Step 13F Evaluation：`evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/`

---

## 1. 执行摘要

本报告对 Step 13F 的正式评估结果执行只读、CPU-only 的形式化审计，目标是把 **结果正确性（correctness）** 与 **workload 简化（workload simplification）** 区分开来。所有 red-flag gate 均通过；未发现 13F 结果在算法实现、目标函数复现、selector 语义或 Exact OPT 最优性上存在错误。

核心发现：
- **结果复现 PASS**：504 个 snapshot×K case 的 C(S) 与 13F 报告完全一致，snapshot digest 未被修改。
- **Selector 语义独立审计 PASS**：LRU、LFU Adaptation、Marconi Adaptation 的独立实现输出与 13F 逐 case 一致。
- **Budget Monotonicity PASS**：168 个 snapshot 在 0.25/0.50/0.75 budget ratio 下均无 C(S) 随 K 增大而上升的违例。
- **Exact OPT 独立审计 PASS**：434 个 tractable cases 用独立 combinations loop 重新枚举，全部与 13F Exact OPT 结果一致；FlowState 在这些 cases 上 100% 达到最优。
- **Workload 结构高度简化**：99.5% 的 candidate 只 compatible 于一个 pending；168/168 snapshot 内每个 workflow 的 compatible candidates 形成严格单调 chain，无 branching。
- **机制解释**：FlowState 的 reduction 主要来自 (1) 目标函数直接优化 recovery gap，(2) 该 workload 的“一对一”兼容结构使 greedy 退化为 per-workflow best，(3) baseline 在 budget 宽松时大量选择 zero-marginal checkpoint（冗余），而 FlowState 的 selected set 零冗余。
- **诚实结论**：Step 13F 的结果在 same-snapshot objective 与给定 population 下是 **正确且最优的**，但其巨大的 magnitude（62–99% reduction）与 RQ3 Main Population 的 **特定结构** 强相关，不能无条件外推到更复杂的多 pending 共享候选或 branching 工作负载。

---

## 2. 审计范围与约束

| 项目 | 值 |
|---|---|
| 输入 | 已冻结的 168 个 ELIGIBLE snapshots + 504 个 13F per-snapshot results |
| 修改权限 | **只读**；未修改核心算法、selector、recovery model、population、evaluation artifact |
| 计算资源 | CPU only；未使用 GPU/SGLang |
| 新增代码 | `evaluation/rq3_sanity_structure_audit.py` |
| 新增测试 | `tests/test_rq3_sanity_structure_audit.py`（15 tests） |
| 审计 artifact root | `rq3_sanity_structure_audit_20260904_121500/` |

---

## 3. Red-Flag Gate 汇总

| Gate | 结果 | 关键指标 |
|---|---|---|
| Result Reproduction | **PASS** | 504 cases，mismatch = 0，snapshot mutation = 0 |
| Selector Semantics Audit | **PASS** | LRU/LFU/Marconi mismatch = 0；frequency boundary violation = 0 |
| Budget Monotonicity | **PASS** | violation = 0 |
| Exact OPT Independent Audit | **PASS** | 434 tractable cases，mismatch = 0 |
| Formal Population Modified | **NO** | — |
| Formal Evaluation Modified | **NO** | — |
| Core Code Modified | **NO** | — |
| GPU Used | **NO** | — |
| **Overall Status** | **RQ3_SANITY_AUDIT_READY** | — |

---

## 4. 结果复现（Result Reproduction）

使用公共 `evaluate_objective(snapshot, selected_ids)` 对每个 policy 与 Exact OPT 的 selected set 重新计算 `C(S)`：

- 总 case 数：504
- mismatch：0
- snapshot digest 运行前后不一致：0

结论：13F 输出的 `total_recovery_cost_ms`、`total_benefit_ms`、`empty_selection_cost_ms` 与独立复现完全一致。

---

## 5. Selector 语义独立审计

| Selector | Mismatch | 说明 |
|---|---|---|
| LRU | 0 | 使用 `select_global_lru`，按 `-last_access_order, -creation_order, checkpoint_id` 排序 |
| LFU Adaptation | 0 | 使用 `select_lfu`，按 `-frequency, -last_access_order, checkpoint_id` 排序 |
| Marconi Adaptation | 0 | 使用 `MarconiStylePolicy`，alpha=1.0，归一化 recency + 归一化 FLOP-saved/memory |

Marconi score 统计（eligible candidate 级别）：
- score mean / median：0.915 / 0.895
- 归一化 recency mean：0.573
- 归一化 efficiency mean：0.342
- tie count：6,828（在约 2,276 candidates 上存在 utility tie，tie-break 完全依赖 checkpoint_id）

---

## 6. Budget Monotonicity

对每个 snapshot，比较相邻 budget ratio 下同一 policy 的 C(S)：
- violation：0

结论：K 增大时 C(S) 不增，符合 submodular greedy 的预期。

---

## 7. Exact OPT 独立审计

- tractable cases：434 / 504
- 独立 combinations loop 与 13F Exact OPT 输出 mismatch：0
- FlowState 在这些 cases 上 exact match rate：1.0

结论：13F 的 Exact OPT 实现正确；FlowState 在所有可解 case 上达到最优。

---

## 8. Workload 结构诊断

### 8.1 Candidate Compatibility Structure

| 指标 | 值 |
|---|---|
| 总 candidate 数 | 2,276 |
| d=1（exactly one pending compatible） | 2,264（99.47%） |
| d=0（无 pending compatible） | 12（0.53%） |
| max compatibility degree | 1 |
| cross-pending candidate 数 | 0 |

**结论**：在 168 个 snapshots 中，几乎不存在跨 pending 共享的 candidate。这意味着选择问题在很大程度上可以按 pending/workflow 分解。

### 8.2 Per-Workflow Chain Structure

| 指标 | 值 |
|---|---|
| strict chain snapshots | 168 / 168（100%） |
| non-decreasing benefit snapshots | 168 / 168（100%） |
| branching structure snapshots | 0 / 168（0%） |

**结论**：每个 workflow 内的 compatible candidates 按 `token_pos` 严格递增，且 standalone benefit 单调非减；没有 lineage branching。这进一步降低了选择问题的复杂度。

---

## 9. Mechanism 诊断

### 9.1 Zero-Marginal Redundancy

| Ratio | Policy | mean zero-marginal count | mean redundancy ratio | mean useful selected |
|---|---|---|---|---|
| 0.25 | FlowState | 0.00 | 0.00% | 3.18 |
| 0.25 | LRU | 1.46 | 43.75% | 1.92 |
| 0.25 | LFU | 0.21 | 4.17% | 3.18 |
| 0.25 | Marconi | 0.92 | 22.98% | 2.46 |
| 0.50 | FlowState | 0.00 | 0.00% | 4.00 |
| 0.50 | LRU | 3.60 | 52.08% | 3.18 |
| 0.75 | FlowState | 0.00 | 0.00% | 4.00 |
| 0.75 | LRU | 6.45 | 60.52% | 3.71 |
| 0.75 | LFU | 6.16 | 55.75% | 4.00 |
| 0.75 | Marconi | 6.17 | 55.95% | 3.99 |

**关键观察**：
- FlowState 在所有 ratio 下 **零冗余**；每个 selected checkpoint 都降低总 recovery cost。
- Baseline 在 budget 越宽松时冗余越严重：LRU 在 0.75 budget 下平均 6.45 个 selected checkpoints 是 zero-marginal（占所选 60.5%）。
- LFU/Marconi 在 0.75 下冗余比例也超过 55%。

### 9.2 Marginal Dependency

| 指标 | 值 |
|---|---|
| greedy 步骤总数 | 21,504 |
| marginal 发生变化（相对 empty set）的步骤 | 6,708（31.2%） |
| 同 pending 冗余导致的 marginal 变化 | 6,708（100%） |
| 多 pending overlap 导致的 marginal 变化 | 0 |
| 其它变化 | 0 |

**结论**：FlowState greedy 确实存在 set dependency，但其来源全部是 **同一 pending 内部多个 compatible candidates 之间的冗余**；没有跨 pending 的边际交互。

### 9.3 PerWorkflowBest Diagnostic Probe

| 指标 | 值 |
|---|---|
| cases | 504 |
| selected set 精确匹配率 | 100.0% |
| C(S) 精确匹配率 | 100.0% |
| 最大 C 差异 | 0.0 ms |
| 结论 | equivalent |

**结论**：在这个 workload 上，FlowState greedy 的输出与“每个 pending 选 standalone benefit 最大的兼容候选，再按 benefit Top-K”完全一致。这再次说明 cross-pending 交互几乎不存在。

### 9.4 StandaloneTopK Diagnostic Probe

| 指标 | 值 |
|---|---|
| cases | 504 |
| selected set 精确匹配率 | 14.5% |
| C(S) 精确匹配率 | 71.4% |
| mean FlowState − StandaloneTopK（ms） | −59.3 ms |
| 结论 | different |

**解读**：StandaloneTopK（全局按 standalone score Top-K，不更新 marginal）在约 29% 的 cases 上比 FlowState 更优或等价差异。这说明：
- 31% 的 greedy 步骤存在 set dependency（见 9.2），因此简单 standalone score 不是完全等价；
- 但该 set dependency 主要表现为止步/冗余修正，而非跨 pending 协同，因此 StandaloneTopK 仍能取得相近甚至更好的成本。

---

## 10. Selected-Set Size 与 Budget Saturation

### 10.1 平均 |S| 与 unused budget

| Ratio | mean K | mean \|S\|_FlowState | fraction \|S\| < K | mean unused budget | pending count |
|---|---|---|---|---|---|
| 0.25 | 3.39 | 3.18 | 20.8% | 0.21 | 4 |
| 0.50 | 6.77 | 4.00 | 71.4% | 2.77 | 4 |
| 0.75 | 10.16 | 4.00 | 100.0% | 6.16 | 4 |

### 10.2 Full Coverage 与 Zero Cost Rate

| Ratio | FlowState full coverage | FlowState zero cost | Marconi full coverage | Marconi zero cost |
|---|---|---|---|---|
| 0.25 | 42.3% | 42.3% | 2.4% | 2.4% |
| 0.50 | 92.9% | 92.9% | 19.0% | 19.0% |
| 0.75 | 92.9% | 92.9% | 41.1% | 41.1% |

**关键观察**：
- FlowState 在 0.50 与 0.75 ratio 下平均只选 4 个 checkpoints（等于 pending count = 4），即可覆盖全部 pending 并实现 zero recovery cost。
- K 在这些 ratio 下远超 4，但 FlowState 因当前边际收益 ≤ 0 而提前停止，导致大量 unused budget。
- Baseline（LRU/LFU/Marconi）把 budget 填满，但很多选择对当前 objective 无贡献，造成高冗余。

---

## 11. FlowState vs Marconi Overlap

| Ratio | mean Jaccard | exact set match rate |
|---|---|---|
| 0.25 | 0.20 | 0.6% |
| 0.50 | 0.27 | 0.0% |
| 0.75 | 0.29 | 0.0% |

**结论**：FlowState 与 Marconi 的 selected set 差异显著，差异不是由实现误差造成，而是 selector 目标不同导致。

---

## 12. Win / Tie / Loss（FlowState vs Baseline）

| Ratio | vs LRU | vs LFU | vs Marconi |
|---|---|---|---|
| 0.25 | 168 win / 0 tie / 0 loss | 168 win / 0 tie / 0 loss | 163 win / 5 tie / 0 loss |
| 0.50 | 90 win / 78 tie / 0 loss | 168 win / 0 tie / 0 loss | 131 win / 37 tie / 0 loss |
| 0.75 | 48 win / 120 tie / 0 loss | 168 win / 0 tie / 0 loss | 92 win / 76 tie / 0 loss |

说明：
- FlowState 从未输给任何 baseline。
- 与 LRU 的 tie 随 budget 增大而增加，因为 LRU 在 budget 足够大时也能覆盖所有 pending。
- 与 LFU 始终 win，反映 LFU Adaptation 的排序与 recovery cost 最不相关。

---

## 13. Per-Round / Candidate-Count Stratification

### 13.1 By Allocation Round（相对 Marconi 的 reduction）

| Round | n | mean reduction | 95% CI |
|---|---|---|---|
| 2 | 144 | 71.9% | 65.2%–78.5% |
| 3 | 126 | 80.9% | 76.4%–85.4% |
| 4 | 129 | 63.1% | 54.7%–71.1% |
| 5 | 105 | 33.2% | 24.6%–42.3% |

### 13.2 By Candidate Count

| \|C\| | n | mean reduction | 95% CI |
|---|---|---|---|
| 8 | 144 | 71.9% | 65.2%–78.5% |
| 12 | 126 | 80.9% | 76.4%–85.4% |
| 16 | 129 | 63.1% | 54.7%–71.1% |
| 20 | 105 | 33.2% | 24.6%–42.3% |

**观察**：reduction 随 candidate count 增加而下降。原因：候选越多，baseline 也越可能偶然覆盖到高 benefit checkpoint；同时 FlowState 的绝对优势空间被更多候选稀释。

---

## 14. Absolute Cost Distribution

| Ratio | C_empty mean | C_FlowState mean | C_Marconi mean | FlowState − Marconi mean |
|---|---|---|---|---|
| 0.25 | 902.2 ms | 129.5 ms | 321.1 ms | −191.6 ms |
| 0.50 | 902.2 ms | 3.4 ms | 121.4 ms | −118.1 ms |
| 0.75 | 902.2 ms | 3.4 ms | 29.7 ms | −26.4 ms |

- 0.50/0.75 下 FlowState 平均 C(S) 仅 3.4 ms，约 99.6% 的 empty cost 被消除。
- 此时即使相对 reduction 很大，**绝对改进的基数在缩小**，因此百分比数字会进一步放大。

---

## 15. Current-Set Myopia（局限性声明）

| 指标 | 值 |
|---|---|
| FlowState mean unused budget | 3.05 |
| FlowState stops before K 比例 | 64.1% |
| mean zero-current-marginal unselected | 9.20 |
| median zero-current-marginal unselected | 8.00 |

**重要声明**：RQ3 当前只评估 **single-allocation-epoch objective**。大量未选中的 candidate 在当前 objective 下边际收益为零，但这 **不能证明** 它们在后续 allocation rounds 中没有价值。因此，把本次结果外推到 multi-round online setting 时需要额外谨慎。

---

## 16. 诚实科学结论

### 16.1 正确性（Correctness）

Step 13F 的结果在本审计下 **全部通过**：
1. 目标函数复现无误；
2. Baseline selector 语义与 13F 实现逐 case 一致；
3. Budget monotonicity 无违例；
4. Exact OPT 独立验证通过；FlowState 在所有 tractable cases 上达到最优。

因此，**FlowState 在 same-snapshot objective 与给定 Main Population 下确实优于 LRU/LFU/Marconi Adaptation，且优于不是由实现 bug 造成**。

### 16.2 Workload 简化（Workload Simplification）

巨大的 reduction magnitude（62–99%）与以下结构特征强相关：
1. **99.5% candidate 只 compatible 于一个 pending**，不存在跨 pending 共享候选；选择问题几乎可分解为 per-pending/workflow 子问题。
2. **所有 snapshot 内 chain 结构严格单调**，无 branching；每个 pending 的兼容候选形成一条清晰的 benefit 非减链。
3. **Pending count = 4**，而 0.50/0.75 budget 下的 K 已大于 4；FlowState 只需选 4 个 useful checkpoints 即可完全覆盖并达到 zero cost，随后因边际收益为零而停止。
4. **Baseline 的代理目标与 recovery cost 不一致**：LRU（recency）、LFU（frequency）、Marconi（FLOP-saved/memory）都倾向于填满 budget，导致大量 zero-marginal 选择；FlowState 直接优化 recovery gap，因此零冗余。
5. **PerWorkflowBest probe 与 FlowState 100% 等价**，进一步说明在此 workload 上 FlowState greedy 本质上执行的是 per-pending best，而非复杂的全局 set-dependent 优化。

### 16.3 不能过度推广的部分

- 本审计确认的是 **给定 population + same-snapshot objective** 下的正确性，不是 FlowState 在所有 recurrent-state allocation 场景下的普适优越性。
- 如果未来工作负载出现 (a) 跨 pending 共享候选增多、(b) branching lineage、(c) multi-round 未来信息价值显著，baseline 与 FlowState 的相对差距可能缩小。
- 当前未评估 multi-round online regret；`current_set_myopia` 显示大量候选在当前 round 零边际，但可能在多 round 中有价值。

### 16.4 最终判定

**Step 13F 结果可用于论文 RQ3，但结论应表述为**：

> “在 OpenHands Main Population 的 same-snapshot 设置下，FlowState 相对 LRU、LFU Adaptation、Marconi Adaptation 显著降低 aggregate executable recovery cost；在所有 tractable Exact OPT cases 上达到最优。该优势与该 workload 中候选几乎唯一对应单个 pending、无 branching 的结构特征一致，且随着 candidate count 增加而减弱。”

---

## 17. Artifact 清单

`evaluation/runtime_artifacts/rq3_sanity_structure_audit_20260904_121500/` 下包含：

- `validation_report.json`
- `result_reproduction.json`
- `selector_semantics_audit.json`
- `budget_monotonicity.json`
- `exact_structure_audit.json`
- `compatibility_structure.json`
- `chain_structure.json`
- `marginal_dependency.json`
- `per_workflow_best_probe.json`
- `standalone_topk_probe.json`
- `marconi_overlap.json`
- `win_tie_loss.json`
- `saturation_explanation.json`
- `selected_size_and_saturation.json`
- `redundancy_analysis.json`
- `workflow_coverage.json`
- `workflow_distribution.json`
- `absolute_cost_distribution.json`
- `current_set_myopia.json`
- `zero_cost_and_gap.json`
- `per_round_analysis.json`
- `INPUTS.json`
- `SANITY_PROTOCOL.json`
- `STEP_13G_A_FINAL_REPORT.md`

---

## 18. 测试环境

| 项目 | 值 |
|---|---|
| Audit interpreter | `/tmp/flowstate-test-venv/bin/python` |
| Python version | 3.12.13 |
| Full CPU suite command | `/tmp/flowstate-test-venv/bin/python -m pytest tests/ -q` |
| Full CPU suite result | **749/749 PASS** |
| Warnings | 2（pytest UnknownMarkWarning for `slow` / `timeout`） |

---

*报告生成于 2026-09-04 · FlowState RQ3 Step 13G-A 审计*
